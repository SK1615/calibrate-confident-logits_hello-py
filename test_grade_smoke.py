from grade import evaluate_once

def test_grade_smoke():
    ok, rep = evaluate_once(seed=1234, mod_path="solve.py")
    assert "base" in rep and "cal" in rep and "deltas" in rep
    # Pass/Fail is allowed either way here; the test ensures the grader runs
