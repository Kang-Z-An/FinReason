import unittest
from finreason.executor import execute, answer_matches, canonical_program

class ExecutorTests(unittest.TestCase):
    def test_multistep_growth(self):
        result = execute("subtract(125, 100), divide(#0, 100)", [])
        self.assertTrue(answer_matches(result, 0.25))
        self.assertEqual(result.steps, 2)

    def test_numeric_conventions(self):
        self.assertEqual(execute("multiply(25%, const_100)", []).value, 25)
        self.assertEqual(execute("multiply(-3, const_m1)", []).value, 3)
        self.assertEqual(execute("divide(1, 3)", []).value, 0.33333)
        self.assertEqual(execute("greater(3, 2)", []).value, "yes")
        self.assertFalse(answer_matches(execute("divide(1, 4)", []), 25))

    def test_table(self):
        table = [["revenue (net), total", "$ 20", "-10 (10)"]]
        self.assertEqual(execute("table_average(revenue (net), total, none)", table).value, 5)

    def test_legacy_table_placeholder_normalization(self):
        table = [["17695228", "963202", "155213", "0"]]
        raw = "table_sum(17695228, 963202), add(#0, 155213)"
        canonical = canonical_program(raw)
        self.assertEqual(canonical, "table_sum(17695228, none), add(#0, 155213)")
        self.assertEqual(execute(raw, table).value, 1273628)
        self.assertEqual(execute(raw, table).value, execute(canonical, table).value)

    def test_invalid_and_unsafe_programs(self):
        for program in ("", "divide(1, 0)", "add(#0, 1)", "add(#-1, 1)",
                        "add(nan, 1)", "exp(10, 9999999)", "exp(-1, 0.5)",
                        "__import__('os').system('echo BAD')", "add(1,2) trailing",
                        "```add(1, 2)```", "add(1, 2, 3)", "table_sum(missing, none)"):
            with self.subTest(program=program):
                self.assertFalse(execute(program, []).valid)
