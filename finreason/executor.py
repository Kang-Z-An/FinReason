"""Restricted FinQA DSL interpreter. Never executes generated Python.

Independent implementation; numerical conventions follow the FinQA evaluator:
https://github.com/czyssrs/FinQA/blob/main/code/evaluate/evaluate.py
This is a local execution metric, not the official symbolic program metric.
"""
import math
import re
from dataclasses import dataclass
from typing import Any


class ProgramError(ValueError):
    pass


@dataclass
class Execution:
    valid: bool
    value: Any = None
    error: str = ""
    steps: int = 0


def number(text):
    text = str(text).strip().replace(",", "")
    if text.startswith("const_"):
        text = text[6:]
        if text == "m1":
            text = "-1"
    value = float(text[:-1]) / 100 if text.endswith("%") else float(text)
    if not math.isfinite(value):
        raise ProgramError("non_finite_number")
    return value


def parse(program):
    if not isinstance(program, str) or not program.strip():
        raise ProgramError("empty_program")
    if len(program) > 8192:
        raise ProgramError("program_too_long")
    # Split at operation boundaries, allowing parentheses in a table row label.
    parts = re.split(r"\)\s*,\s*(?=[a-z_]+\s*\()", program.strip())
    if len(parts) > 32:
        raise ProgramError("too_many_steps")
    result = []
    for index, part in enumerate(parts):
        if index < len(parts) - 1:
            part += ")"
        match = re.fullmatch(r"([a-z_]+)\s*\((.*)\)", part.strip())
        if not match:
            raise ProgramError("invalid_syntax")
        op, args = match.groups()
        # Table labels may contain commas; the second argument is a placeholder.
        pair = args.rsplit(",", 1) if op.startswith("table_") else args.split(",")
        if len(pair) != 2:
            raise ProgramError("invalid_arity")
        result.append((op, pair[0].strip(), pair[1].strip()))
    return result


def execute(program, table):
    values = []
    try:
        operations = parse(program)

        def resolve(arg):
            if arg.startswith("#"):
                if not re.fullmatch(r"#\d+", arg) or int(arg[1:]) >= len(values):
                    raise ProgramError("invalid_reference")
                value = values[int(arg[1:])]
                if isinstance(value, str):
                    raise ProgramError("non_numeric_reference")
                return value
            return number(arg)

        for op, left, right in operations:
            if op in {"table_sum", "table_average", "table_min", "table_max"}:
                rows = {row[0]: row[1:] for row in table if row}
                if left not in rows:
                    raise ProgramError("unknown_table_row")
                # Official FinQA ignores the second table argument. Two training
                # annotations contain a number instead of the usual "none".
                nums = [number(str(cell).replace("$", "").split("(")[0].strip())
                        for cell in rows[left]]
                if not nums:
                    raise ProgramError("empty_table_row")
                value = {"table_sum": sum, "table_average": lambda x: sum(x) / len(x),
                         "table_min": min, "table_max": max}[op](nums)
            else:
                if op not in {"add", "subtract", "multiply", "divide", "exp", "greater"}:
                    raise ProgramError("unknown_operator")
                a, b = resolve(left), resolve(right)
                if op == "add":
                    value = a + b
                elif op == "subtract":
                    value = a - b
                elif op == "multiply":
                    value = a * b
                elif op == "divide":
                    value = a / b
                elif op == "greater":
                    value = "yes" if a > b else "no"
                else:
                    if abs(b) > 1000:
                        raise ProgramError("exponent_limit")
                    value = a ** b
            if not isinstance(value, str) and (isinstance(value, complex) or not math.isfinite(value)):
                raise ProgramError("non_finite_result")
            values.append(value)
        value = values[-1]
        return Execution(True, value if isinstance(value, str) else round(value, 5), steps=len(values))
    except (ValueError, TypeError, OverflowError, ZeroDivisionError, IndexError) as exc:
        return Execution(False, error=str(exc) or type(exc).__name__, steps=len(values))


def answer_matches(result, gold):
    """Strict equality after execution rounding, matching official exe_ans comparison."""
    return result.valid and result.value == gold


def canonical_program(program):
    """Normalize the unused table argument without changing execution semantics."""
    return ", ".join("{}({}, {})".format(op, left, "none" if op.startswith("table_") else right)
                     for op, left, right in parse(program))
