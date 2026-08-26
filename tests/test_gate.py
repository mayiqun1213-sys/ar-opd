import unittest

from ar_opd.core import OptionCandidate, OptionKind
from ar_opd.rollout import ValueGate


def candidate(kind: OptionKind, value: float, query: float = 0.0, execution: float = 0.0):
    return OptionCandidate(
        kind=kind,
        actions=(0,),
        preview_steps=1,
        estimated_task_value=value,
        query_cost=query,
        execution_cost=execution,
        terminated=False,
        truncated=False,
    )


class ValueGateTest(unittest.TestCase):
    def test_selects_highest_value_after_teacher_cost(self) -> None:
        selected = ValueGate().choose(
            (
                candidate(OptionKind.STUDENT, 0.60),
                candidate(OptionKind.TEACHER_CORRECTION, 1.00, 0.10, 0.20),
                candidate(OptionKind.TEACHER_RECOVERY, 1.20, 0.10, 0.60),
            )
        )
        self.assertIs(selected.kind, OptionKind.TEACHER_CORRECTION)
        self.assertAlmostEqual(selected.net_score, 0.70)

    def test_exact_tie_conservatively_prefers_student(self) -> None:
        selected = ValueGate().choose(
            (
                candidate(OptionKind.TEACHER_RECOVERY, 0.5),
                candidate(OptionKind.TEACHER_CORRECTION, 0.5),
                candidate(OptionKind.STUDENT, 0.5),
            )
        )
        self.assertIs(selected.kind, OptionKind.STUDENT)


if __name__ == "__main__":
    unittest.main()
