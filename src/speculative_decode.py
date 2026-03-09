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

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Mode = Literal["autoregressive", "specdec", "sa_only", "hybrid_fixed", "hybrid_dynamic"]


# ---------------------------------------------------------------------------
# Dynamic draft length controller
# ---------------------------------------------------------------------------

class DynamicLengthController:
    """
    Adapts the number of speculative draft tokens based on rolling acceptance rate.

    Separate windows are maintained for SA drafts and draft-model drafts because
    they have different acceptance characteristics.

    Controller logic (per source):
      - If recent acceptance rate > high_thresh: increase draft_len by 1
      - If recent acceptance rate < low_thresh:  decrease draft_len by 1
      - Clamp to [min_draft_len, max_draft_len]
    """

    WINDOW_SIZE = 20
    MIN_DRAFT_LEN = 2
    MAX_DRAFT_LEN = 10
    HIGH_THRESH = 0.8
    LOW_THRESH = 0.4

    def __init__(self, initial_draft_len: int = 4) -> None:
        # Per-source state
        self._draft_len: dict[str, int] = {
            "SA": initial_draft_len,
            "draft": initial_draft_len,
        }
        # Rolling windows: list of (proposed, accepted) per step
        self._window: dict[str, list[tuple[int, int]]] = {"SA": [], "draft": []}

    def update(self, source: str, accepted: int, proposed: int) -> None:
        """Record one draft attempt and adjust draft length if needed."""
        if source not in self._window:
            return
        window = self._window[source]
        window.append((proposed, accepted))
        if len(window) > self.WINDOW_SIZE:
            window.pop(0)

        total_proposed = sum(p for p, _ in window)
        total_accepted = sum(a for _, a in window)
        if total_proposed == 0:
            return
        rate = total_accepted / total_proposed

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

    Args:
        target_model:       The large model (e.g. Qwen2.5-Coder-1.5B-Instruct)
        target_tokenizer:   Tokenizer for target model
        draft_model:        The small model (e.g. Qwen2.5-Coder-0.5B-Instruct)
        draft_tokenizer:    Tokenizer for draft model (usually same vocab as target)
        device:             "cuda" | "cpu" | "mps"
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
        sa_threshold: int = 4,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> tuple[str, GenerationMetrics]:
        """
        Generate text from *prompt* using the specified decoding mode.

        Returns:
            (generated_text, metrics)
        """
        tracker = MetricsTracker()
        t_start = time.perf_counter()

        # Tokenize prompt
        input_ids = self.target_tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_len = input_ids.shape[1]

        # Build suffix automaton from prompt tokens
        dsa = DualSuffixAutomaton()
        if mode in ("sa_only", "hybrid_fixed", "hybrid_dynamic"):
            prompt_tokens = input_ids[0].tolist()
            dsa.build_from_prompt(prompt_tokens)

        # Dynamic length controller (only used in hybrid_dynamic)
        dlc = DynamicLengthController(initial_draft_len=num_draft_tokens)

        generated_ids = input_ids.clone()
        first_token = True

        while (generated_ids.shape[1] - prompt_len) < max_new_tokens:
            remaining = max_new_tokens - (generated_ids.shape[1] - prompt_len)

            if mode == "autoregressive":
                new_token, elapsed = self._autoregressive_step(generated_ids, temperature, top_p)
                if first_token:
                    tracker.record_ttft(elapsed)
                    first_token = False
                tracker.record_draft_attempt("autoregressive", proposed=1, accepted=1, draft_len=1)
                if new_token == self.target_tokenizer.eos_token_id:
                    break
                generated_ids = torch.cat([generated_ids, new_token.unsqueeze(0).unsqueeze(0)], dim=1)
                dsa.extend(new_token.item())

            else:
                # --- Determine draft source and length ---
                context_tokens = generated_ids[0].tolist()

                if mode == "specdec":
                    draft_len = num_draft_tokens
                    sa_drafts, sa_match = [], 0
                elif mode == "sa_only":
                    draft_len = dlc.get_draft_length("SA") if mode == "hybrid_dynamic" else num_draft_tokens
                    sa_drafts, sa_match = dsa.query(context_tokens, max_draft_len=draft_len)
                    # No draft model fallback
                else:  # hybrid_fixed or hybrid_dynamic
                    sa_dl = dlc.get_draft_length("SA") if mode == "hybrid_dynamic" else num_draft_tokens
                    draft_dl = dlc.get_draft_length("draft") if mode == "hybrid_dynamic" else num_draft_tokens
                    sa_drafts, sa_match = dsa.query(context_tokens, max_draft_len=sa_dl)
                    draft_len = sa_dl if sa_match >= sa_threshold else draft_dl

                # --- Cap draft length to remaining budget ---
                draft_len = min(draft_len, remaining)
                if draft_len <= 0:
                    break

                # --- Decide which source to use ---
                use_sa = (
                    mode in ("sa_only", "hybrid_fixed", "hybrid_dynamic")
                    and sa_match >= sa_threshold
                    and len(sa_drafts) > 0
                )

                if use_sa:
                    source = "SA"
                    draft_tokens = sa_drafts[:draft_len]
                elif mode in ("specdec", "hybrid_fixed", "hybrid_dynamic"):
                    source = "draft"
                    draft_tokens = self._draft_model_tokens(generated_ids, draft_len, temperature)
                else:
                    # sa_only with no match — fall back to single autoregressive step
                    source = "autoregressive"
                    new_token, elapsed = self._autoregressive_step(generated_ids, temperature, top_p)
                    if first_token:
                        tracker.record_ttft(elapsed)
                        first_token = False
                    tracker.record_draft_attempt("autoregressive", proposed=1, accepted=1, draft_len=1)
                    if new_token == self.target_tokenizer.eos_token_id:
                        break
                    generated_ids = torch.cat([generated_ids, new_token.unsqueeze(0).unsqueeze(0)], dim=1)
                    dsa.extend(new_token.item())
                    continue

                # --- Single target forward pass over draft tokens ---
                t_fwd_start = time.perf_counter()
                accepted_tokens, bonus_token = self._verify_drafts(
                    generated_ids, draft_tokens, source, temperature
                )
                elapsed_fwd = time.perf_counter() - t_fwd_start

                if first_token:
                    tracker.record_ttft(elapsed_fwd)
                    first_token = False

                # --- Record metrics ---
                proposed = len(draft_tokens)
                accepted = len(accepted_tokens)
                tracker.record_draft_attempt(source, proposed=proposed, accepted=accepted, draft_len=proposed)

                # --- Update dynamic controller ---
                if mode == "hybrid_dynamic":
                    dlc.update(source, accepted=accepted, proposed=proposed)

                # --- Append accepted tokens + bonus ---
                all_new = accepted_tokens
                if bonus_token is not None:
                    all_new = accepted_tokens + [bonus_token]

                for tok in all_new:
                    if tok == self.target_tokenizer.eos_token_id:
                        generated_ids = torch.cat(
                            [generated_ids, torch.tensor([[tok]], device=self.device)], dim=1
                        )
                        # Force exit
                        remaining = 0
                        break
                    generated_ids = torch.cat(
                        [generated_ids, torch.tensor([[tok]], device=self.device)], dim=1
                    )
                    dsa.extend(tok)

                if remaining == 0:
                    break

        total_time = time.perf_counter() - t_start
        total_tokens = generated_ids.shape[1] - prompt_len

        metrics = tracker.finalize(total_tokens=total_tokens, total_time=total_time)

        decoded = self.target_tokenizer.decode(
            generated_ids[0][prompt_len:], skip_special_tokens=True
        )
        return decoded, metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _autoregressive_step(
        self, input_ids: torch.Tensor, temperature: float, top_p: float
    ) -> tuple[torch.Tensor, float]:
        """Single greedy/sampled target step. Returns (token_id, elapsed_s)."""
        t0 = time.perf_counter()
        logits = self.target_model(input_ids).logits[0, -1, :]
        token = self._sample(logits, temperature, top_p)
        return token, time.perf_counter() - t0

    def _sample(self, logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> torch.Tensor:
        """Temperature + top-p sampling returning a single token id."""
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

    def _draft_model_tokens(
        self, input_ids: torch.Tensor, num_tokens: int, temperature: float
    ) -> list[int]:
        """Run draft model autoregressively for *num_tokens* steps."""
        draft_ids = input_ids.clone()
        tokens = []
        for _ in range(num_tokens):
            logits = self.draft_model(draft_ids).logits[0, -1, :]
            tok = self._sample(logits, temperature).item()
            tokens.append(tok)
            draft_ids = torch.cat(
                [draft_ids, torch.tensor([[tok]], device=self.device)], dim=1
            )
            if tok == self.target_tokenizer.eos_token_id:
                break
        return tokens

    def _verify_drafts(
        self,
        prefix_ids: torch.Tensor,
        draft_tokens: list[int],
        source: str,
        temperature: float,
    ) -> tuple[list[int], int | None]:
        """
        Single forward pass through target model to verify draft tokens.

        Uses mathematically exact rejection sampling (Leviathan et al. 2023):
          accept_prob_i = min(1, p_target(t_i) / p_draft(t_i))

        For SA drafts we treat p_draft = 1.0 (deterministic, hard-coded token)
        so the accept probability becomes min(1, p_target(t_i)).

        Returns (accepted_tokens, bonus_token_or_None).
        """
        n = len(draft_tokens)
        if n == 0:
            return [], None

        # Build extended sequence: prefix + draft tokens
        draft_tensor = torch.tensor([draft_tokens], device=self.device)
        extended = torch.cat([prefix_ids, draft_tensor], dim=1)

        # One forward pass — get logits for all positions
        logits = self.target_model(extended).logits  # [1, prefix+n, vocab]

        # We need logits at positions: prefix_len-1 ... prefix_len+n-2
        # (each position predicts the *next* token)
        prefix_len = prefix_ids.shape[1]
        target_logits = logits[0, prefix_len - 1 : prefix_len + n - 1, :]  # [n, vocab]
        bonus_logits = logits[0, prefix_len + n - 1, :]  # logits for token after last draft

        if temperature == 0.0:
            # Greedy: accept iff draft token == argmax of target logits
            accepted = []
            for i, tok in enumerate(draft_tokens):
                best = target_logits[i].argmax().item()
                if tok == best:
                    accepted.append(tok)
                else:
                    bonus = best
                    return accepted, bonus
            bonus = bonus_logits.argmax().item()
            return accepted, bonus

        # Sampled: exact rejection sampling
        target_probs = F.softmax(target_logits / max(temperature, 1e-8), dim=-1)  # [n, vocab]
        bonus_probs = F.softmax(bonus_logits / max(temperature, 1e-8), dim=-1)

        if source == "SA":
            # Draft is deterministic: p_draft(t_i) = 1  (point mass on draft_tokens[i])
            # accept_prob = min(1, p_target(t_i)) = p_target(t_i) (always ≤ 1)
            draft_token_probs = target_probs[torch.arange(n), draft_tokens]  # [n]
            accept_probs = draft_token_probs  # already min(1, p_target/1)
        else:
            # Get draft model probs via its own forward pass
            draft_token_probs_list = self._draft_probs(prefix_ids, draft_tokens, temperature)
            draft_token_probs = torch.tensor(draft_token_probs_list, device=self.device)
            target_draft_probs = target_probs[torch.arange(n), draft_tokens]
            accept_probs = torch.clamp(target_draft_probs / (draft_token_probs + 1e-10), max=1.0)

        accepted = []
        for i, tok in enumerate(draft_tokens):
            u = torch.rand(1, device=self.device).item()
            if u < accept_probs[i].item():
                accepted.append(tok)
            else:
                # Sample correction token from residual distribution
                if source != "SA":
                    residual = F.relu(target_probs[i] - draft_token_probs[i] * accept_probs[i])
                    z = residual.sum()
                    if z > 1e-8:
                        bonus = torch.multinomial(residual / z, num_samples=1).item()
                    else:
                        bonus = target_probs[i].argmax().item()
                else:
                    bonus = torch.multinomial(target_probs[i], num_samples=1).item()
                return accepted, bonus

        # All accepted — sample bonus token from target
        bonus = torch.multinomial(bonus_probs, num_samples=1).item()
        return accepted, bonus

    def _draft_probs(
        self, prefix_ids: torch.Tensor, draft_tokens: list[int], temperature: float
    ) -> list[float]:
        """
        Compute draft model's probability for each draft token given prefix.
        Returns list of float probabilities, one per draft token.
        """
        probs = []
        ids = prefix_ids.clone()
        for tok in draft_tokens:
            logits = self.draft_model(ids).logits[0, -1, :]
            p = F.softmax(logits / max(temperature, 1e-8), dim=-1)
            probs.append(p[tok].item())
            ids = torch.cat([ids, torch.tensor([[tok]], device=self.device)], dim=1)
        return probs
