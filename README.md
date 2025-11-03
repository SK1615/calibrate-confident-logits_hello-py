# Calibrate Those Confident Logits – Reinforcement Learning Task

##  Objective
This RL task teaches a practical ML engineering skill — **probability calibration**.  
Given over-confident classification logits, the goal is to learn a calibrator on **validation data** and apply it to **test data**, returning valid probabilities.  
The task models a realistic ML workflow where models must remain reliable under mild distribution shift.

---

##  Task Description
You are provided with:
- `logits_val [Nv, K]` – validation logits  
- `y_val [Nv]` – validation labels  
- `logits_test [Nt, K]` – test logits  

Implement:
```python
def solve(logits_val: np.ndarray, y_val: np.ndarray, logits_test: np.ndarray) -> np.ndarray:
    """
    Fit a calibrator ONLY on (logits_val, y_val) and apply it to logits_test.
    Return calibrated probabilities [Nt, K], non-negative, with each row summing to 1.
    Any correct calibration method is allowed (temperature scaling, isotonic, Platt,
    histogram binning, Dirichlet, vector/bias scaling, etc.).
    Use only numpy; keep it deterministic; no file or network access.
    """

Grading (on TEST) :
Metric	Requirement	Description
NLL improvement	≥ REQ_NLL_IMP	Negative log-likelihood must improve
ECE improvement	≥ REQ_ECE_IMP	Expected calibration error must improve
Mean max-confidence reduction	≥ REQ_CONF_RED	Reduce over-confidence
Accuracy drop	≤ MAX_ACC_DROP	Maintain classification accuracy

Thresholds are tuned for a 10–40% pass rate across random seeds.


Setup Instructions
1️. Clone this repository
git clone https://github.com/SK1615/calibrate-confident-logits_hello-py.git
cd calibrate-confident-logits_hello-py

2️. Create and activate a virtual environment
python -m venv .venv

3️. Install dependencies
pip install numpy pytest anthropic


4. Run Locally:
python grade.py              # single-seed smoke run (may PASS/FAIL)
python run_many.py --n 20    # shows pass-rate; aim for 10–40%

Tune Thresholds (optional):
python tune_thresholds.py --n 20


5. Run with LLM (Anthropic Claude) :

Set your Anthropic API key first:

windows/ vs code:
$env:ANTHROPIC_API_KEY ="sk-ant-******"

Then run:
python llm_attempt.py --model claude-3-5-haiku-latest --n 10

This writes the LLM’s generated solution to solve_backup.py and evaluates it via the stable wrapper solve.py.

6. unit Tests :

Run all tests to confirm environment integrity:
pytest -v

Expected output :

test_grade_smoke.py::test_grade_smoke :PASSED
test_metrics_and_data.py::test_make_synthetic_shapes :PASSED
test_metrics_and_data.py::test_softmax_valid :PASSED
test_metrics_and_data.py::test_metrics_monotonicity :PASSED
test_wrapper_contract.py::test_solve_contract :PASSED
