# Rule-based global router — no learned weights, no HuggingFace, no training needed.
#
# Classifies the raw input text into one of four question types using
# keyword / regex patterns, then returns a hardcoded action plan for that type.
#
# Plans:
#   factual      → [rag, decoder]           look it up, then answer
#   calculation  → [tool, decoder]           compute it, then answer
#   uncertain    → [rag, decoder, rag]       retrieve, reason, verify
#   reasoning    → [decoder, mem, decoder]   think, remember context, conclude
#
# Priority (highest → lowest):  reasoning > calculation > uncertain > factual
# Unrecognised queries default to factual.

import re

# ── Pattern definitions ────────────────────────────────────────────────────────

_RE_REASONING = re.compile(
    r"\b("
    r"design|architect(ure)?|compare|contrast|analyz|analys|"
    r"reason|step.by.step|pros.and.cons|trade.?off|"
    r"how would you|think through|evaluate the|"
    r"difference between|similarities between|"
    r"best approach|optimal|tradeoff"
    r")\b",
    re.I,
)

_RE_CALCULATION = re.compile(
    r"("
    r"\d+\s*[\+\-\*\/\^%]\s*\d+"             # inline arithmetic: 3 + 4
    r"|\bcalculate\b|\bcompute\b|\bsolve\b"
    r"|\bevaluate\b|\bsum of\b|\bproduct of\b"
    r"|\bintegrate\b|\bderivative\b|\bequation\b"
    r"|\bhow many\b.{0,30}\d"                 # "how many X are there (with a number nearby)"
    r"|\bpercentage\b|\bconvert\b.{0,20}\bto\b"
    r")",
    re.I,
)

_RE_UNCERTAIN = re.compile(
    r"^("
    r"why\b|how does\b|how do\b|explain\b|"
    r"what would\b|what might\b|what if\b|"
    r"could (you|it|this|that)\b|"
    r"should (i|we|one)\b|"
    r"is it possible|"
    r"what causes\b|what leads\b|"
    r"in what way"
    r")",
    re.I,
)

_RE_FACTUAL = re.compile(
    r"^("
    r"what (is|are|was|were)\b|"
    r"who (is|are|was|were)\b|"
    r"where (is|are|was|were)\b|"
    r"when (did|was|were|is)\b|"
    r"which\b|define\b|tell me (about|what)\b|"
    r"what does\b|how (much|many)\b|"
    r"name (a|the|an)\b|list (the|a|some)\b"
    r")",
    re.I,
)

# ── Plan table ─────────────────────────────────────────────────────────────────

PLANS: dict[str, list[str]] = {
    "factual":     ["rag", "decoder"],
    "calculation": ["tool", "decoder"],
    "uncertain":   ["rag", "decoder", "rag"],    # last rag = verify step
    "reasoning":   ["decoder", "mem", "decoder"],
}


# ── Public classifier ──────────────────────────────────────────────────────────

def classify(text: str) -> str:
    """
    Return one of: 'factual', 'calculation', 'uncertain', 'reasoning'.
    Checked in priority order — reasoning wins if multiple patterns match.
    """
    t = text.strip()
    if _RE_REASONING.search(t):
        return "reasoning"
    if _RE_CALCULATION.search(t):
        return "calculation"
    if _RE_UNCERTAIN.match(t):
        return "uncertain"
    if _RE_FACTUAL.match(t):
        return "factual"
    return "factual"    # safe default


# ── Router class ───────────────────────────────────────────────────────────────

class RuleRouter:
    """
    Drop-in replacement for GlobalRouter in demo / no-checkpoint mode.
    Returns a deterministic action plan from regex classification alone.
    No nn.Module, no parameters, no training required.
    """

    def forward(self, raw_input: str, encoded_input=None):
        """
        raw_input    : original query string
        encoded_input: accepted but ignored — kept for API parity with GlobalRouter

        Returns
        -------
        action_plan  : list[str]   e.g. ['rag', 'decoder']
        tick_count   : int         length of the plan
        query_type   : str         one of the four category labels
        """
        qtype = classify(raw_input)
        plan  = list(PLANS[qtype])
        return plan, len(plan), qtype
