import json
from pathlib import Path

SYSTEM = """Read the financial context and answer the question by writing a FinQA program.
Output only the program, with operations separated by a comma and a space.
Allowed operations: add, subtract, multiply, divide, exp, greater,
table_sum, table_average, table_min, table_max.
Every operation has two arguments. Use #0, #1, ... for previous results.
For table operations use the exact row label and none as the second argument.
Use numbers from the context or question and const_ constants when required.
Example syntax: subtract(125, 100), divide(#0, 100)
Do not output explanations, Markdown, or the final answer separately."""


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def messages(example):
    """Whitelist only report text, table, and question; never serialize the whole qa."""
    context = {"pre_text": example["pre_text"], "table": example["table"],
               "post_text": example["post_text"], "question": example["qa"]["question"]}
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]


def load_split(raw_dir, split):
    rows = read_json(Path(raw_dir) / (split + ".json"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty list: " + split)
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError("Duplicate id in split: " + row["id"])
        seen.add(row["id"])
        for field in ("pre_text", "post_text", "table"):
            if not isinstance(row[field], list):
                raise ValueError("Invalid field " + field)
        for field in ("question", "program", "exe_ans"):
            if field not in row["qa"]:
                raise ValueError("Missing label field " + field)
    return rows

