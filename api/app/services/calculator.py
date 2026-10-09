import ast

from fastapi import HTTPException


def calculate_expression(equation: str) -> float:
    expression = equation.replace(r"\times", "*").replace(r"\div", "/").replace(r"\cdot", "*").replace(" ", "")

    if expression.endswith("="):
        expression = expression[:-1]

    try:
        tree = ast.parse(
            expression,
            mode="eval",
        )

        result = evaluate_node(tree.body)

        return float(result)

    except ZeroDivisionError:
        raise HTTPException(
            status_code=400,
            detail="Division by zero.",
        )

    except (SyntaxError, ValueError, TypeError):
        raise HTTPException(
            status_code=400,
            detail=f"Unable to calculate {equation} due to invalid syntax or unsupported operations.",
        )


def evaluate_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value

        raise ValueError("Unsupported value.")

    if isinstance(node, ast.BinOp):
        left = evaluate_node(node.left)
        right = evaluate_node(node.right)

        if isinstance(node.op, ast.Add):
            return left + right

        if isinstance(node.op, ast.Sub):
            return left - right

        if isinstance(node.op, ast.Mult):
            return left * right

        if isinstance(node.op, ast.Div):
            return left / right

        raise ValueError("Unsupported operator.")

    if isinstance(node, ast.UnaryOp):
        value = evaluate_node(node.operand)

        if isinstance(node.op, ast.UAdd):
            return value

        if isinstance(node.op, ast.USub):
            return -value

        raise ValueError("Unsupported unary operator.")

    raise ValueError("Unsupported expression.")
