import json
import subprocess
import sys

from aix_page.experimentation.assignment import jackknife_bucket, policy_partition, treatment_for


def test_assignment_stable_in_process() -> None:
    assert treatment_for("e", "u0") == treatment_for("e", "u0")
    assert 0 <= jackknife_bucket("e", "u0") < 20
    assert policy_partition("e", "u0") in {
        "policy_discovery",
        "policy_selection",
        "policy_locked_test",
    }


def test_assignment_stable_across_processes() -> None:
    code = "from aix_page.experimentation.assignment import treatment_for; import json; print(json.dumps(treatment_for('e','u0')))"
    outputs = [
        subprocess.check_output([sys.executable, "-c", code], text=True).strip() for _ in range(2)
    ]
    assert json.loads(outputs[0]) == json.loads(outputs[1])


def test_anonymous_raw_id_not_assignment_identity() -> None:
    assignments = {treatment_for("e", f"sim_{i}") for i in range(100)}
    assert len(assignments) == 12
