# Safe arithmetic calculator — no model, no eval().
#
# Two public functions:
#
#   calculate(expression)      — evaluate a math expression string and return
#                                 the result as a string.  Uses Python's AST
#                                 parser with a strict whitelist of nodes so
#                                 arbitrary code cannot be injected.
#
#   extract_expression(text)   — pull a computable expression out of a natural
#                                 language query so the loop can call calculate()
#                                 without any model involvement.
#
# Supported operations:
#   + - * / // % **              (binary arithmetic)
#   - + (unary)
#   sqrt, abs, ceil, floor,
#   sin, cos, tan, log, log10,
#   log2, factorial              (named functions)
#   pi, e                        (constants)
#   "square root of N"           (word-form rewrites → sqrt(N))
#   "N squared / cubed"
#   "N ^ M"   → N**M
#   "factorial of N"

import ast
import math
import operator
import re

# ── Safe node evaluator ────────────────────────────────────────────────────────

_OPERATORS = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod:      operator.mod,
    ast.Pow:      operator.pow,
    ast.USub:     operator.neg,
    ast.UAdd:     operator.pos,
}

_SAFE_NAMES: dict = {
    "pi":        math.pi,
    "e":         math.e,
    "sqrt":      math.sqrt,
    "abs":       abs,
    "ceil":      math.ceil,
    "floor":     math.floor,
    "round":     round,
    "sin":       math.sin,
    "cos":       math.cos,
    "tan":       math.tan,
    "log":       math.log,
    "log10":     math.log10,
    "log2":      math.log2,
    "factorial": math.factorial,
}


def _eval_node(node: ast.AST):
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Non-numeric constant: {node.value!r}")
        return node.value

    if isinstance(node, ast.Name):
        if node.id not in _SAFE_NAMES:
            raise ValueError(f"Unknown name: {node.id!r}")
        return _SAFE_NAMES[node.id]

    if isinstance(node, ast.BinOp):
        op = _OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))

    if isinstance(node, ast.UnaryOp):
        op = _OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
        return op(_eval_node(node.operand))

    if isinstance(node, ast.Call):
        func = _eval_node(node.func)
        if not callable(func):
            raise ValueError("Not callable")
        args = [_eval_node(a) for a in node.args]
        return func(*args)

    raise ValueError(f"Disallowed AST node: {type(node).__name__}")


# ── Public: calculate ──────────────────────────────────────────────────────────

def calculate(expression: str) -> str:
    """
    Evaluate a math expression string safely and return the result as a string.
    Returns an error string (prefixed 'Error:') on failure — never raises.

    Examples
    --------
    calculate("3 + 4 * 2")          → "11"
    calculate("sqrt(144)")           → "12"
    calculate("2 ** 10")             → "1024"
    calculate("factorial(6)")        → "720"
    calculate("sin(pi / 2)")         → "1.0"
    """
    if not expression or not expression.strip():
        return "Error: empty expression"

    expr = expression.strip().replace("^", "**")   # ^ → ** for convenience
    try:
        tree   = ast.parse(expr, mode="eval")
        result = _eval_node(tree.body)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as exc:
        return f"Error: {exc}"

    # Pretty-print: drop the ".0" suffix for whole-number floats
    if isinstance(result, float):
        if result.is_integer() and abs(result) < 1e15:
            return str(int(result))
        return f"{result:.10g}"          # up to 10 significant figures

    return str(result)


# ── Public: extract_expression ─────────────────────────────────────────────────

# Word-form → symbolic rewrites applied before pattern matching
_WORD_REWRITES: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\bsquare\s+root\s+of\s+([\d.]+)\b', re.I), r'sqrt(\1)'),
    (re.compile(r'\bsqrt\s+of\s+([\d.]+)\b',          re.I), r'sqrt(\1)'),
    (re.compile(r'([\d.]+)\s+squared\b',               re.I), r'(\1**2)'),
    (re.compile(r'([\d.]+)\s+cubed\b',                 re.I), r'(\1**3)'),
    (re.compile(r'([\d.]+)\s*\^\s*([\d.]+)',           re.I), r'(\1**\2)'),
    (re.compile(r'\bfactorial\s+of\s+(\d+)\b',        re.I), r'factorial(\1)'),
    (re.compile(r'([\d.]+)\s+percent\b',               re.I), r'(\1/100)'),
    (re.compile(r'\btimes\b',                          re.I), '*'),
    (re.compile(r'\bdivided\s+by\b',                   re.I), '/'),
    (re.compile(r'\bplus\b',                           re.I), '+'),
    (re.compile(r'\bminus\b',                          re.I), '-'),
]

# Matches a numeric expression after word rewrites
_RE_EXPR = re.compile(
    r'((?:sqrt|abs|ceil|floor|sin|cos|tan|log(?:10|2)?|factorial|round)'
    r'\s*\([^)]+\)'                              # named function call
    r'|[\d.]+\s*(?:[+\-*/%]|\*\*)\s*[\d.]+)'   # inline arithmetic
)


def extract_expression(text: str) -> str:
    """
    Extract a math expression from natural language.
    Returns the expression string ready for calculate(), or '' if none found.

    Examples
    --------
    "What is 12 * 7?"                    → "12 * 7"
    "Calculate the square root of 144"   → "sqrt(144)"
    "How much is 15 percent of 200?"     → "(15/100)"   (partial; caller combines)
    "What is 5 factorial?"               → "factorial(5)"  [if phrased that way]
    """
    t = text.strip()

    # Apply word rewrites first
    for pattern, repl in _WORD_REWRITES:
        t = pattern.sub(repl, t)

    # Try to find a recognised expression
    match = _RE_EXPR.search(t)
    if match:
        return match.group(1).strip()

    # Fallback: bare inline arithmetic with no function names
    bare = re.search(r'([\d.]+\s*[+\-*/%]\s*[\d.]+)', t)
    if bare:
        return bare.group(1).strip()

    return ""
