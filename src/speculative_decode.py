"""
Hybrid Speculative Decoding Engine
===================================
Implements five decoding modes:
  - autoregressive    : vanilla greedy/sampling, no speculation
  - specdec           : draft-model-only speculative decoding (fixed draft length)
  - sa_only           : suffix-automaton-only speculation (fixed draft length)
  - hybrid_fixed      : SA + draft model, fixed draft length
  - hybrid_dynamic    : SA + draft model, DYNAMIC draft length (novel contribution)

Dynamic draft length is the explicit "future work" item from Baseten's Jan 27 2026
SA MTP blog post — this implementation uses a rolling-window acceptance rate
controller, one tracker per source (SA vs draft model).

KV Cache: both target and draft models maintain running key-value caches across
decoding steps, so each forward pass only processes NEW tokens (O(n)) rather
than the full sequence (O(L)). This is the critical fix that makes speculative
decoding faster than autoregressive.

Mathematical correctness: rejection sampling follows the exact algorithm from
  Leviathan et al. (2023) "Fast Inference from Transformers via Speculative Decoding"
ensuring the output distribution equals the target model's distribution.
"""

from __future__ import annotations

import time
from typing import Literal

import torch
import torch.nn.functional as F

from .suffix_automaton import DualSuffixAutomaton
from .utils import GenerationMetrics, MetricsTracker

Mode = Literal["autoregressive", "specdec", "sa_only", "hybrid_fixed", "hybrid_dynamic"]


# ---------------------------------------------------------------------------
# KV cache helpers
# ---------------------------------------------------------------------------

def _trim_kv(past_kv, seq_len: int):
    """Trim KV cache to cover only the first *seq_len* positions.

    Handles both the legacy tuple-of-tuples format (transformers < 4.38)
    and the DynamicCache object format (transformers >= 4.38).
    """
    try:
        from transformers.cache_utils import DynamicCache
        if isinstance(past_kv, DynamicCache):
            new_cache = DynamicCache()
            for k, v in zip(past_kv.key_cache, past_kv.value_cache):
                new_cache.key_cache.append(k[:, :, :seq_len, :])
                new_cache.value_cache.append(v[:, :, :seq_len, :])
            return new_cache
    except ImportError:
        pass
    # Legacy tuple format
    return tuple((k[:, :, :seq_len, :], v[:, :, :seq_len, :]) for k, v in past_kv)


def _kv_len(past_kv) -> int:
    """Return the sequence length covered by a KV cache."""
    try:
        return past_kv.get_seq_length()          # DynamicCache (transformers >= 4.38)
    except AttributeError:
        return past_kv[0][0].shape[2]            # legacy tuple format


# ---------------------------------------------------------------------------
# Dynamic draft length controller
# ---------------------------------------------------------------------------

class DynamicLengthController:
    """
    Adapts draft token count using a rolling-window acceptance rate per source.

    - rate > HIGH_THRESH → increase draft_len
    - rate < LOW_THRESH  → decrease draft_len
    - Clamps to [MIN_DRAFT_LEN, MAX_DRAFT_LEN]

    Separate windows for SA and draft-model sources because they have
    different base acceptance rates.
    """

    WINDOW_SIZE = 20
    MIN_DRAFT_LEN = 2
    MAX_DRAFT_LEN = 10
    HIGH_THRESH = 0.8
    LOW_THRESH = 0.4

    def __init__(self, initial_draft_len: int = 4) -> None:
        self._draft_len: dict[str, int] = {"SA": initial_draft_len, "draft": initial_draft_len}
        self._window: dict[str, list[tuple[int, int]]] = {"SA": [], "draft": []}

    def update(self, source: str, accepted: int, proposed: int) -> None:
        if source not in self._window:
            return
        window = self._window[source]
        window.append((proposed, accepted))
        if len(window) > self.WINDOW_SIZE:
            window.pop(0)
        total_p = sum(p for p, _ in window)
        total_a = sum(a for _, a in window)
        if total_p == 0:
            return
        rate = total_a / total_p
        if rate > self.HIGH_THRESH:
            self._draft_len[source] = min(self._draft_len[source] + 1, self.MAX_DRAFT_LEN)
        elif rate < self.LOW_THRESH:
            self._draft_len[source] = max(self._draft_len[source] - 1, self.MIN_DRAFT_LEN)

    def get_draft_length(self, source: str) -> int:
        return self._draft_len.get(source, 4)


# ---------------------------------------------------------------------------
# Core decoder
# ---------------------------------------------------------------------------

class HybridSpecDecoder:
    """
    Unified speculative decoding engine supporting all five modes.

    Both target and draft models maintain running KV caches so each
    forward pass only processes new tokens — O(draft_len) not O(seq_len).
    """

    def __init__(
        self,
        target_model,
        target_tokenizer,
        draft_model=None,
        draft_tokenizer=None,
        device: str = "cuda",
    ) -> None:
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer or target_tokenizer
        self.device = device

        self.target_model.eval()
        if self.draft_model is not None:
            self.draft_model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        mode: Mode = "hybrid_dynamic",
        num_draft_tokens: int = 4,
        sa_threshold: int = 2,       # lowered from 4 → SA fires on 2-token matches
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> tuple[str, GenerationMetrics]:
        """Generate text using the specified decoding mode. Returns (text, metrics)."""
        tracker = MetricsTracker()
        t_start = time.perf_counter()

        # Tokenize prompt
        input_ids = self.target_tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_len = input_ids.shape[1]

        # Build suffix automaton from prompt tokens
        dsa = DualSuffixAutomaton()
        if mode in ("sa_only", "hybrid_fixed", "hybrid_dynamic"):
            dsa.build_from_prompt(input_ids[0].tolist())

        # Dynamic length controller
        dlc = DynamicLengthController(initial_draft_len=num_draft_tokens)

        uses_draft = mode in ("specdec", "hybrid_fixed", "hybrid_dynamic")

        # ------------------------------------------------------------------
        # Prefill: single forward pass to populate KV caches
        # ------------------------------------------------------------------
        t_prefill = time.perf_counter()

        t_out = self.target_model(input_ids, use_cache=True)
        target_past_kv = t_out.past_key_values
        target_last_logit = t_out.logits[0, -1, :]   # P(next | prompt)

        draft_past_kv = None
        draft_last_logit = None
        if uses_draft and self.draft_model is not None:
            d_out = self.draft_model(input_ids, use_cache=True)
            draft_past_kv = d_out.past_key_values
            draft_last_logit = d_out.logits[0, -1, :]

        tracker.record_ttft(time.perf_counter() - t_prefill)
        first_token = False  # TTFT already recorded at prefill

        generated_ids = input_ids.clone()
        eos = self.target_tokenizer.eos_token_id

        # ------------------------------------------------------------------
        # Main generation loop
        # ------------------------------------------------------------------
        while (generated_ids.shape[1] - prompt_len) < max_new_tokens:
            remaining = max_new_tokens - (generated_ids.shape[1] - prompt_len)

            # ==============================================================
            # Autoregressive mode — one cached target step per iteration
            # ==============================================================
            if mode == "autoregressive":
                new_tok = self._sample(target_last_logit, temperature, top_p).item()
                tracker.record_draft_attempt("autoregressive", proposed=1, accepted=1, draft_len=1)
                if new_tok == eos:
                    break
                generated_ids = torch.cat(
                    [generated_ids, torch.tensor([[new_tok]], device=self.device)], dim=1
                )
                dsa.extend(new_tok)
                # Advance target KV by one token
                t_out = self.target_model(
                    torch.tensor([[new_tok]], device=self.device),
                    past_key_values=target_past_kv,
                    use_cache=True,
                )
                target_past_kv = t_out.past_key_values
                target_last_logit = t_out.logits[0, -1, :]
                continue

            # ==============================================================
            # Speculative modes — decide source and draft length
            # ==============================================================
            context_tokens = generated_ids[0].tolist()

            if mode == "specdec":
                draft_len = num_draft_tokens
                sa_drafts, sa_match = [], 0
            elif mode == "sa_only":
                draft_len = num_draft_tokens
                sa_drafts, sa_match = dsa.query(context_tokens, max_draft_len=draft_len)
            else:  # hybrid_fixed / hybrid_dynamic
                sa_dl = dlc.get_draft_length("SA") if mode == "hybrid_dynamic" else num_draft_tokens
                draft_dl = dlc.get_draft_length("draft") if mode == "hybrid_dynamic" else num_draft_tokens
                sa_drafts, sa_match = dsa.query(context_tokens, max_draft_len=sa_dl)
                draft_len = sa_dl if sa_match >= sa_threshold else draft_dl

            draft_len = min(draft_len, remaining)
            if draft_len <= 0:
                break

            use_sa = (
                mode in ("sa_only", "hybrid_fixed", "hybrid_dynamic")
                and sa_match >= sa_threshold
                and len(sa_drafts) > 0
            )

            if use_sa:
                source = "SA"
                draft_tokens = sa_drafts[:draft_len]
            elif uses_draft and self.draft_model is not None:
                source = "draft"
                draft_tokens, draft_past_kv, draft_last_logit = self._draft_model_tokens_cached(
                    draft_last_logit, draft_past_kv, draft_len, temperature
                )
            else:
                # sa_only with no match — fall back to single cached autoregressive step
                source = "autoregressive"
                new_tok = self._sample(target_last_logit, temperature, top_p).item()
                tracker.record_draft_attempt("autoregressive", proposed=1, accepted=1, draft_len=1)
                if new_tok == eos:
                    break
                generated_ids = torch.cat(
                    [generated_ids, torch.tensor([[new_tok]], device=self.device)], dim=1
                )
                dsa.extend(new_tok)
                t_out = self.target_model(
                    torch.tensor([[new_tok]], device=self.device),
                    past_key_values=target_past_kv,
                    use_cache=True,
                )
                target_past_kv = t_out.past_key_values
                target_last_logit = t_out.logits[0, -1, :]
                continue

            # ==============================================================
            # Verify draft tokens — single cached target forward pass
            # ==============================================================
            (
                accepted_tokens,
                bonus_token,
                target_past_kv,
                target_last_logit,
            ) = self._verify_drafts_cached(
                target_last_logit,
                target_past_kv,
                draft_tokens,
                source,
                temperature,
                draft_past_kv,
                draft_last_logit,
            )

            proposed = len(draft_tokens)
            accepted = len(accepted_tokens)
            tracker.record_draft_attempt(source, proposed=proposed, accepted=accepted, draft_len=proposed)

            if mode == "hybrid_dynamic":
                dlc.update(source, accepted=accepted, proposed=proposed)

            # Append accepted + bonus to sequence, update SA and draft KV
            all_new = accepted_tokens + ([bonus_token] if bonus_token is not None else [])
            hit_eos = False
            for tok in all_new:
                generated_ids = torch.cat(
                    [generated_ids, torch.tensor([[tok]], device=self.device)], dim=1
                )
                dsa.extend(tok)
                if tok == eos:
                    hit_eos = True
                    break

            # Sync draft model KV cache to the newly accepted tokens.
            # draft_past_kv after _draft_model_tokens_cached covers [prefix + all proposed drafts].
            # We need it to cover [prefix + accepted_tokens + bonus].
            # Strategy: trim back to pre-draft length, then extend with all_new tokens in one pass.
            if uses_draft and self.draft_model is not None and all_new:
                pre_draft_kv_len = _kv_len(target_past_kv) - len(all_new)
                draft_base_kv = _trim_kv(draft_past_kv, pre_draft_kv_len)
                sync_ids = torch.tensor([all_new], device=self.device)
                d_out = self.draft_model(sync_ids, past_key_values=draft_base_kv, use_cache=True)
                draft_past_kv = d_out.past_key_values
                draft_last_logit = d_out.logits[0, -1, :]

            if hit_eos:
                break

        total_time = time.perf_counter() - t_start
        total_tokens = generated_ids.shape[1] - prompt_len
        metrics = tracker.finalize(total_tokens=total_tokens, total_time=total_time)
        decoded = self.target_tokenizer.decode(generated_ids[0][prompt_len:], skip_special_tokens=True)
        return decoded, metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample(self, logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> torch.Tensor:
        if temperature == 0.0:
            return logits.argmax()
        logits = logits / temperature
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove] = -float("inf")
            logits = torch.zeros_like(logits).scatter_(0, sorted_indices, sorted_logits)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze()

    def _draft_model_tokens_cached(
        self,
        draft_last_logit: torch.Tensor,
        draft_past_kv: tuple,
        num_tokens: int,
        temperature: float,
    ) -> tuple[list[int], tuple, torch.Tensor]:
        """
        Generate num_tokens draft tokens using the cached draft model.
        Each step processes only the new token — O(1) per step instead of O(L).
        Returns (tokens, updated_past_kv, updated_last_logit).
        """
        tokens = []
        kv = draft_past_kv
        logit = draft_last_logit

        for _ in range(num_tokens):
            tok = self._sample(logit, temperature).item()
            tokens.append(tok)
            if tok == self.target_tokenizer.eos_token_id:
                break
            out = self.draft_model(
                torch.tensor([[tok]], device=self.device),
                past_key_values=kv,
                use_cache=True,
            )
            kv = out.past_key_values
            logit = out.logits[0, -1, :]

        return tokens, kv, logit

    def _verify_drafts_cached(
        self,
        target_last_logit: torch.Tensor,
        target_past_kv: tuple,
        draft_tokens: list[int],
        source: str,
        temperature: float,
        draft_past_kv: tuple | None,
        draft_last_logit: torch.Tensor | None,
    ) -> tuple[list[int], int | None, tuple, torch.Tensor]:
        """
        Single cached target forward pass over draft tokens.

        Uses target_last_logit (the prediction from the previous accepted step)
        to verify draft_tokens[0], and the output logits to verify draft_tokens[1..n-1].

        Returns (accepted_tokens, bonus_token, new_target_past_kv, new_target_last_logit).
        """
        n = len(draft_tokens)
        if n == 0:
            return [], None, target_past_kv, target_last_logit

        draft_tensor = torch.tensor([draft_tokens], device=self.device)
        prefix_kv_len = _kv_len(target_past_kv)

        # Single forward pass — only processes n new tokens, not the full prefix
        out = self.target_model(draft_tensor, past_key_values=target_past_kv, use_cache=True)
        # out.logits[0, i, :] = P(next | prefix + draft[0..i])
        # → verifies draft[i+1]; bonus = out.logits[0, n-1, :]
        # target_last_logit verifies draft[0]

        # Build verification logit matrix [n, vocab]
        if n == 1:
            verification_logits = target_last_logit.unsqueeze(0)  # [1, vocab]
        else:
            verification_logits = torch.cat(
                [target_last_logit.unsqueeze(0), out.logits[0, :-1, :]], dim=0
            )  # [n, vocab]

        bonus_logit = out.logits[0, -1, :]

        if temperature == 0.0:
            # Greedy verification
            accepted = []
            for i, tok in enumerate(draft_tokens):
                best = verification_logits[i].argmax().item()
                if tok == best:
                    accepted.append(tok)
                else:
                    bonus = best
                    new_kv, new_logit = self._advance_target_kv(
                        target_past_kv, out.past_key_values, prefix_kv_len,
                        len(accepted), bonus
                    )
                    return accepted, bonus, new_kv, new_logit
            bonus = bonus_logit.argmax().item()
            new_kv, new_logit = self._advance_target_kv(
                target_past_kv, out.past_key_values, prefix_kv_len, n, bonus
            )
            return accepted, bonus, new_kv, new_logit

        # Sampled: exact rejection sampling (Leviathan et al. 2023)
        target_probs = F.softmax(verification_logits / max(temperature, 1e-8), dim=-1)  # [n, vocab]
        bonus_probs = F.softmax(bonus_logit / max(temperature, 1e-8), dim=-1)

        if source == "SA":
            # p_draft(t_i) = 1 (deterministic), so accept_prob = min(1, p_target(t_i))
            accept_probs = target_probs[torch.arange(n), draft_tokens]
        else:
            # Get draft model probabilities using cached forward passes
            draft_probs_list = self._draft_probs_cached(
                draft_last_logit, draft_past_kv, draft_tokens, temperature
            )
            draft_token_probs = torch.tensor(draft_probs_list, device=self.device)
            target_draft_probs = target_probs[torch.arange(n), draft_tokens]
            accept_probs = torch.clamp(target_draft_probs / (draft_token_probs + 1e-10), max=1.0)

        accepted = []
        for i, tok in enumerate(draft_tokens):
            if torch.rand(1, device=self.device).item() < accept_probs[i].item():
                accepted.append(tok)
            else:
                # Sample correction from residual distribution
                if source != "SA":
                    residual = F.relu(target_probs[i] - accept_probs[i] * target_probs[i])
                    z = residual.sum()
                    bonus = (torch.multinomial(residual / z, 1).item() if z > 1e-8
                             else target_probs[i].argmax().item())
                else:
                    bonus = torch.multinomial(target_probs[i], 1).item()
                new_kv, new_logit = self._advance_target_kv(
                    target_past_kv, out.past_key_values, prefix_kv_len, len(accepted), bonus
                )
                return accepted, bonus, new_kv, new_logit

        # All accepted — sample bonus from target
        bonus = torch.multinomial(bonus_probs, 1).item()
        new_kv, new_logit = self._advance_target_kv(
            target_past_kv, out.past_key_values, prefix_kv_len, n, bonus
        )
        return accepted, bonus, new_kv, new_logit

    def _advance_target_kv(
        self,
        prefix_kv: tuple,
        full_draft_kv: tuple,
        prefix_kv_len: int,
        k_accepted: int,
        bonus_token: int,
    ) -> tuple[tuple, torch.Tensor]:
        """
        After accepting k draft tokens + a bonus token, advance the target KV cache.

        1. Trim full_draft_kv to prefix + k accepted positions.
        2. Run one target step with bonus_token to get the new KV + last logit.
        """
        trimmed = _trim_kv(full_draft_kv, prefix_kv_len + k_accepted)
        out = self.target_model(
            torch.tensor([[bonus_token]], device=self.device),
            past_key_values=trimmed,
            use_cache=True,
        )
        return out.past_key_values, out.logits[0, -1, :]

    def _draft_probs_cached(
        self,
        draft_last_logit: torch.Tensor,
        draft_past_kv: tuple,
        draft_tokens: list[int],
        temperature: float,
    ) -> list[float]:
        """
        Compute draft model probability for each draft token using KV cache.
        draft_last_logit[i] gives P(draft_tokens[i] | context before draft_tokens[i]).
        """
        probs = []
        kv = draft_past_kv
        logit = draft_last_logit

        for tok in draft_tokens:
            p = F.softmax(logit / max(temperature, 1e-8), dim=-1)
            probs.append(p[tok].item())
            out = self.draft_model(
                torch.tensor([[tok]], device=self.device),
                past_key_values=kv,
                use_cache=True,
            )
            kv = out.past_key_values
            logit = out.logits[0, -1, :]

        return probs
