"""
Pure-Python Suffix Automaton (SAM) — O(n) construction via Blumer's algorithm.

Reimplements the core data structure from Baseten's sa_spec C++/CUDA repo
without any native dependencies, enabling Colab T4 compatibility.

References:
  - Blumer et al. (1985) "The smallest automaton recognizing the subwords of a text"
  - Baseten blog: "Suffix Automaton Multi-Token Prediction" (Jan 27, 2026)
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class SuffixAutomatonState:
    """A single state in the suffix automaton.

    Each state represents an equivalence class of substrings (an end-set class).

    Attributes:
        next:     Transition map token_id → state_id
        link:     Suffix link (parent in suffix link tree)
        len:      Length of the longest substring in this equivalence class
        cnt:      Number of times this equivalence class occurs (set during topological sort)
        last_tok: Most recently indexed outgoing token from this state.  Used by
                  query() to prefer the freshest continuation over min(next.keys()),
                  which matters most at temperature=0 where token selection is
                  deterministic: if context C was seen before and the model (greedily)
                  generated token X, last_tok=X and SA will propose X → accepted.
    """
    next: dict[int, int] = field(default_factory=dict)
    link: int = -1
    len: int = 0
    cnt: int = 0
    last_tok: int = -1


# ---------------------------------------------------------------------------
# Suffix Automaton
# ---------------------------------------------------------------------------

class SuffixAutomaton:
    """
    Online suffix automaton supporting:
      - O(n) bulk construction from a token list
      - O(1) amortized incremental extension (one token at a time)
      - O(k) query returning up to k draft tokens from a matched context suffix

    Usage:
        sa = SuffixAutomaton()
        sa.build([1, 2, 3, 1, 2])
        drafts, match_len = sa.query([1, 2], max_draft_len=4)
        # drafts == [3], match_len == 2
    """

    def __init__(self) -> None:
        self._reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        # State 0 is the initial (root) state
        self.states: list[SuffixAutomatonState] = [SuffixAutomatonState(len=0, link=-1)]
        self._last: int = 0   # index of state corresponding to last appended token
        self._size: int = 1

    def _new_state(self, length: int, link: int = -1) -> int:
        self.states.append(SuffixAutomatonState(len=length, link=link))
        idx = self._size
        self._size += 1
        return idx

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(self, tokens: list[int]) -> None:
        """Build the SA from scratch over *tokens* in O(n)."""
        self._reset()
        for tok in tokens:
            self.extend_one(tok)

    def extend_one(self, token: int) -> None:
        """Extend the automaton by one token in O(1) amortized.

        This is the core Blumer algorithm step.  After extension the
        automaton accepts exactly all substrings of the tokens seen so far.
        """
        # Check if transition already exists from _last (handles repeated tokens)
        if token in self.states[self._last].next:
            q = self.states[self._last].next[token]
            if self.states[q].len == self.states[self._last].len + 1:
                self._last = q
                return
            # Clone q
            clone = self._new_state(
                length=self.states[self._last].len + 1,
                link=self.states[q].link,
            )
            self.states[clone].next = dict(self.states[q].next)
            p = self._last
            while p != -1 and self.states[p].next.get(token) == q:
                self.states[p].next[token] = clone
                p = self.states[p].link
            self.states[q].link = clone
            self._last = clone
            return

        cur = self._new_state(length=self.states[self._last].len + 1)
        p = self._last
        while p != -1 and token not in self.states[p].next:
            self.states[p].next[token] = cur
            self.states[p].last_tok = token
            p = self.states[p].link
        if p == -1:
            self.states[cur].link = 0
        else:
            q = self.states[p].next[token]
            if self.states[q].len == self.states[p].len + 1:
                self.states[cur].link = q
            else:
                # Clone q to fix suffix links
                clone = self._new_state(
                    length=self.states[p].len + 1,
                    link=self.states[q].link,
                )
                self.states[clone].next = dict(self.states[q].next)
                while p != -1 and self.states[p].next.get(token) == q:
                    self.states[p].next[token] = clone
                    p = self.states[p].link
                self.states[q].link = clone
                self.states[cur].link = clone
        self._last = cur

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        context_tokens: list[int],
        max_draft_len: int = 8,
    ) -> tuple[list[int], int]:
        """Find draft tokens by matching the longest suffix of *context_tokens*.

        Algorithm:
          1. Walk from the initial state following *context_tokens* (the whole
             context, left-to-right).  On a mismatch, try progressively shorter
             suffixes via suffix links until we find a matching path or exhaust
             all options.
          2. From the matched state, greedily follow the first available forward
             edge up to *max_draft_len* hops, collecting draft token IDs.

        Returns:
            (draft_tokens, match_length)
              draft_tokens  — list of predicted next token IDs (may be empty)
              match_length  — length of the context suffix that was matched
        """
        if not context_tokens or self._size <= 1:
            return [], 0

        # Walk the automaton following context_tokens from state 0
        state = 0
        matched = 0
        for tok in context_tokens:
            if tok in self.states[state].next:
                state = self.states[state].next[tok]
                matched += 1
            else:
                # Fall back through suffix links until we can resume
                while state != -1 and tok not in self.states[state].next:
                    state = self.states[state].link
                if state == -1:
                    state = 0
                    matched = 0
                else:
                    state = self.states[state].next[tok]
                    matched = self.states[state].len

        if matched == 0 or not self.states[state].next:
            return [], matched

        # Greedily collect draft tokens from forward edges.
        # Prefer last_tok (most recently indexed outgoing transition) over min(keys).
        # At temperature=0 the target model is deterministic: if context C was seen
        # before and generated token X, then last_tok=X and SA will propose the
        # correct token → high acceptance.  min(keys) picks an arbitrary token by
        # vocab ID and is wrong most of the time when multiple transitions exist.
        draft_tokens: list[int] = []
        cur_state = state
        for _ in range(max_draft_len):
            if not self.states[cur_state].next:
                break
            s = self.states[cur_state]
            next_tok = s.last_tok if s.last_tok != -1 and s.last_tok in s.next else min(s.next.keys())
            draft_tokens.append(next_tok)
            cur_state = s.next[next_tok]

        return draft_tokens, matched

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def num_states(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"SuffixAutomaton(states={self._size}, last={self._last})"


# ---------------------------------------------------------------------------
# Dual automaton wrapper (prompt SA + dynamic generation SA)
# ---------------------------------------------------------------------------

class DualSuffixAutomaton:
    """
    Maintains two automata as described in the Baseten sa_spec architecture:

      - prompt_sa:  Built once from the prompt tokens (static).
      - live_sa:    Extended token-by-token during generation (dynamic).

    Query checks live_sa first (longer match wins), falls back to prompt_sa.
    """

    def __init__(self) -> None:
        self.prompt_sa = SuffixAutomaton()
        self.live_sa = SuffixAutomaton()

    def build_from_prompt(self, prompt_tokens: list[int]) -> None:
        self.prompt_sa.build(prompt_tokens)
        # Seed live_sa with the same tokens so early queries also match
        self.live_sa.build(prompt_tokens)

    def extend(self, token: int) -> None:
        """Extend the live automaton with a newly accepted generation token."""
        self.live_sa.extend_one(token)

    def query(
        self,
        context_tokens: list[int],
        max_draft_len: int = 8,
    ) -> tuple[list[int], int]:
        """Query both automata; return the result with the longer match."""
        live_drafts, live_match = self.live_sa.query(context_tokens, max_draft_len)
        prompt_drafts, prompt_match = self.prompt_sa.query(context_tokens, max_draft_len)
        if live_match >= prompt_match:
            return live_drafts, live_match
        return prompt_drafts, prompt_match
