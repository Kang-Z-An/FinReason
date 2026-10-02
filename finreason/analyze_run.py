"""Replay a saved run against local gold; report smoke metrics separately from full dev."""
import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path
import statistics

from .data import load_split, read_json, write_json
from .executor import execute, answer_matches, parse, ProgramError


def format_issues(program):
    """Diagnostic contract checks, NOT changes to the execution metric or RL reward."""
    issues = []
    try:
        operations = parse(program)
    except (ValueError, TypeError) as exc:
        return [str(exc)]
    for index, (op, left, right) in enumerate(operations):
        if op.startswith("table_"):
            if right != "none":
                issues.append("table_second_argument_not_none")
            if left.startswith("#"):
                issues.append("table_row_label_is_reference")
        else:
            for arg in (left, right):
                if arg.startswith("#"):
                    try:
                        reference = int(arg[1:])
                        if reference < 0 or reference >= index:
                            issues.append("forward_or_missing_reference")
                    except ValueError:
                        issues.append("malformed_reference")
    return sorted(set(issues))


def official_functions(path):
    # Same reviewed upstream function extraction as check_official.py. No model code execution.
    tree = ast.parse(Path(path).read_bytes())
    names = {"str_to_num", "process_row", "eval_program", "program_tokenization"}
    selected = [node for node in tree.body if
                (isinstance(node, ast.FunctionDef) and node.name in names) or
                (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "all_ops" for t in node.targets))]
    if len(selected) != 5:
        raise ValueError("Upstream source changed; review first")
    scope = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    return scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", default="dev", choices=("train", "dev", "test"))
    parser.add_argument("--raw-dir", default="data/raw/finqa")
    parser.add_argument("--official-source", default="data/raw/finqa/official_evaluate.py")
    parser.add_argument("--output", default="reports/smoke_v1_analysis.json")
    args = parser.parse_args()
    run = Path(args.run)
    predictions = [json.loads(line) for line in (run/"predictions.jsonl").read_text().splitlines()]
    if not predictions or len({r["id"] for r in predictions}) != len(predictions):
        raise ValueError("Empty or duplicate predictions")
    raw = load_split(args.raw_dir, args.split)
    gold = {r["id"]: r for r in raw}
    official = official_functions(args.official_source)
    details = []
    for prediction in predictions:
        example = gold[prediction["id"]]
        local = execute(prediction["program"], example["table"])
        tokens = official["program_tokenization"](prediction["program"])
        invalid, value = official["eval_program"](tokens, example["table"])
        details.append({"id": example["id"], "question": example["qa"]["question"],
            "predicted_program": prediction["program"], "gold_program": example["qa"]["program"],
            "gold_answer": example["qa"]["exe_ans"], "local_valid": local.valid,
            "local_answer": local.value, "local_correct": answer_matches(local, example["qa"]["exe_ans"]),
            "local_error": local.error, "official_valid": invalid == 0, "official_answer": value,
            "official_correct": invalid == 0 and value == example["qa"]["exe_ans"],
            "format_issues": format_issues(prediction["program"]),
            "finish_reason": prediction["finish_reason"], "completion_tokens": prediction["completion_tokens"],
            "generation_seconds": prediction["generation_seconds"]})
    n = len(details)
    summary = {"scope": "generated_subset_only", "generated": n, "dataset_size": len(raw),
        "local_valid": sum(r["local_valid"] for r in details), "local_correct": sum(r["local_correct"] for r in details),
        "official_valid": sum(r["official_valid"] for r in details),
        "official_correct": sum(r["official_correct"] for r in details),
        "local_errors": dict(collections.Counter(r["local_error"] for r in details if r["local_error"])),
        "format_issue_counts": dict(collections.Counter(i for r in details for i in r["format_issues"])),
        "length_stopped": sum(r["finish_reason"] == "length" for r in details),
        "generation_seconds_total": sum(r["generation_seconds"] for r in details),
        "generation_seconds_median": statistics.median(r["generation_seconds"] for r in details),
        "run_manifest": read_json(run/"run.json"),
        "source_hashes": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in {
            "predictions": run/"predictions.jsonl", "official_evaluator": Path(args.official_source),
            "raw_split": Path(args.raw_dir)/(args.split+".json")}.items()}, "details": details}
    if (run/"smoke_report.json").exists():
        original = read_json(run/"smoke_report.json")
        assert (original["examples"], original["valid"], original["correct"]) == (n, summary["local_valid"], summary["local_correct"])
        summary["matches_colab_smoke_report"] = True
    write_json(args.output, summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("details", "run_manifest", "source_hashes")}, indent=2))


if __name__ == "__main__":
    main()
