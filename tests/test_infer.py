import contextlib
import fcntl
import io
import json
from pathlib import Path
import tempfile
import unittest

from finreason.infer import (atomic_json, encode_prompts, item_seed, load_prompts,
                             native_bf16_supported, recover_predictions, run_predictions)


def prompts():
    return [{"id": name, "messages": [{"role": "system", "content": "Write a program"},
                                      {"role": "user", "content": "Add one and one"}]} for name in ("a", "b", "c")]


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = prompts()
        self.spec = {"config": {"seed": 42}, "prompts_sha256": "fixture"}

    def run_fixture(self, callback, limit=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_predictions(self.rows, self.spec, self.root, callback, limit)

    def test_interrupt_then_resume_preserves_completed_rows_and_seeds(self):
        first_calls, second_calls = [], []

        def interrupted(row, seed):
            first_calls.append((row["id"], seed))
            if row["id"] == "b":
                raise KeyboardInterrupt()
            return {"program": "add(1, 1)"}

        with self.assertRaises(KeyboardInterrupt):
            self.run_fixture(interrupted)
        self.assertEqual(json.loads((self.root/"progress.json").read_text())["state"], "interrupted")
        first_line = (self.root/"predictions.jsonl").read_bytes()

        def resumed(row, seed):
            second_calls.append((row["id"], seed))
            return {"program": "add(1, 1)"}

        result = self.run_fixture(resumed)
        self.assertTrue(result["complete_dataset"])
        self.assertEqual([x[0] for x in second_calls], ["b", "c"])
        self.assertEqual(first_calls[1], second_calls[0])
        self.assertTrue((self.root/"predictions.jsonl").read_bytes().startswith(first_line))
        self.assertEqual(len((self.root/"predictions.jsonl").read_text().splitlines()), 3)
        # A completed rerun never invokes the backend (or loads model weights).
        self.run_fixture(lambda *_: self.fail("Completed sample regenerated"))

    def test_error_is_saved_and_retryable(self):
        def fail(*_):
            raise RuntimeError("simulated OOM")
        with self.assertRaises(RuntimeError):
            self.run_fixture(fail)
        self.assertFalse((self.root/"predictions.jsonl").exists())
        self.assertEqual(json.loads((self.root/"errors.jsonl").read_text())["id"], "a")
        self.assertEqual(self.run_fixture(lambda *_: {"program": "add(1, 1)"})["completed_total"], 3)

    def test_limit_can_expand_without_repeating_predictions(self):
        result = self.run_fixture(lambda *_: {"program": "add(1, 1)"}, limit=1)
        self.assertEqual(result["state"], "subset_complete")
        calls = []
        self.run_fixture(lambda row, _: calls.append(row["id"]) or {"program": "add(1, 1)"})
        self.assertEqual(calls, ["b", "c"])

    def test_changed_config_refuses_resume(self):
        self.run_fixture(lambda *_: {"program": "add(1, 1)"}, limit=1)
        self.spec["config"]["seed"] = 99
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_fixture(lambda *_: self.fail("Must not generate"))

    def test_unterminated_tail_is_archived_and_retried(self):
        path = self.root/"predictions.jsonl"
        valid = b'{"id":"a","program":"add(1, 1)"}\n'
        fragment = b'{"id":"b","program":"add('
        path.write_bytes(valid + fragment)
        self.assertEqual(set(recover_predictions(path, {"a", "b"})), {"a"})
        self.assertEqual(path.read_bytes(), valid)
        self.assertEqual(next(self.root.glob("*.interrupted-*")).read_bytes(), fragment)

    def test_interior_corruption_duplicate_or_unknown_ids_fail(self):
        path = self.root/"predictions.jsonl"
        for text in ('{"id":\n', '{"id":"z","program":"x"}\n',
                     '{"id":"a","program":"x"}\n{"id":"a","program":"x"}\n'):
            path.write_text(text)
            with self.assertRaises(ValueError):
                recover_predictions(path, {"a", "b"})

    def test_single_writer_lock(self):
        with (self.root/".lock").open("a") as other:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "Another process"):
                self.run_fixture(lambda *_: self.fail("Must not generate"))

    def test_sft_or_label_input_is_rejected(self):
        path = self.root/"input.jsonl"
        row = prompts()[0]
        path.write_text(json.dumps(row) + "\n")
        self.assertEqual(load_prompts(path), [row])
        row["exe_ans"] = 2
        path.write_text(json.dumps(row) + "\n")
        with self.assertRaises(ValueError):
            load_prompts(path)
        del row["exe_ans"]
        row["messages"].append({"role": "assistant", "content": "add(1, 1)"})
        path.write_text(json.dumps(row) + "\n")
        with self.assertRaises(ValueError):
            load_prompts(path)

    def test_context_guard_preserves_full_input(self):
        class Tokenizer:
            def apply_chat_template(inner, messages, **kwargs):
                self.assertEqual(kwargs, {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False})
                return list(range(100))
        with self.assertRaisesRegex(ValueError, "Context budget"):
            encode_prompts(self.rows, Tokenizer(), {"context_length": 120, "max_new_tokens": 21})
        self.assertEqual(len(encode_prompts(self.rows, Tokenizer(), {"context_length": 120, "max_new_tokens": 20})["a"]), 100)

    def test_per_item_seed_is_stable(self):
        self.assertEqual(item_seed(42, "a"), item_seed(42, "a"))
        self.assertNotEqual(item_seed(42, "a"), item_seed(42, "b"))

    def test_bf16_check_excludes_emulation(self):
        class CUDA:
            def is_bf16_supported(inner, including_emulation=True):
                return including_emulation
        class Torch:
            cuda = CUDA()
        self.assertFalse(native_bf16_supported(Torch()))

    def test_bf16_check_older_torch_uses_architecture(self):
        class CUDA:
            def is_bf16_supported(inner):
                return True
            def get_device_capability(inner):
                return (7, 5)
        class Torch:
            cuda = CUDA()
        self.assertFalse(native_bf16_supported(Torch()))


if __name__ == "__main__":
    unittest.main()
