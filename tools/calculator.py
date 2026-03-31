import ast
import operator

from tools.base import BaseTool, ToolResult

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float | int:
    """Recursively walk the AST and compute numeric-only expressions."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_safe_eval(node.operand))

    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and abs(right) > 1000:
            raise ValueError(f"Exponent too large: {right}. Maximum allowed is 1000.")
        return _OPERATORS[type(node.op)](left, right)

    raise ValueError(
        f"Unsupported expression: {ast.dump(node)}. "
        "Only numbers and basic operators (+, -, *, /, //, %, **) are allowed."
    )


class CalculatorTool(BaseTool):
    """Safe math evaluator using AST — no eval() or exec()."""

    name: str = "calculator"
    description: str = (
        "Evaluates a mathematical expression and returns the result. "
        "Supports: +, -, *, /, //, %, ** and parentheses. "
        "Example: '(25 * 4) + 100' → 200"
    )

    def run(self, expression: str) -> ToolResult:
        if not expression or not expression.strip():
            return ToolResult(success=False, output=None, error="Empty expression provided.")

        try:
            tree = ast.parse(expression.strip(), mode="eval")
            result = _safe_eval(tree)
            return ToolResult(success=True, output=result)
        except (ValueError, TypeError, SyntaxError, ZeroDivisionError) as e:
            return ToolResult(success=False, output=None, error=str(e))

    def get_input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate, e.g. '(25 * 4) + 100'",
                }
            },
            "required": ["expression"],
        }
