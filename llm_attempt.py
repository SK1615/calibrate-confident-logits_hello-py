import os, re, subprocess, argparse, textwrap, json
from pathlib import Path
from anthropic import Anthropic

ROOT = Path(__file__).parent

SYSTEM = """You are an expert ML engineer. You must return ONLY a Python file that implements:
def solve(logits_val, y_val, logits_test) -> np.ndarray:
    - Fit a calibrator using ONLY (logits_val, y_val).
    - Apply it to logits_test and return probabilities [Nt,K].
    - Non-negative; each row sums to 1.
Important:
- Use only numpy; no sklearn. Keep it deterministic.
- Do not read files or the internet.
- Return ONLY a single Python code block that is a complete file (with imports)."""

def build_user_prompt():
    task_prompt = (ROOT / "task_prompt.txt").read_text()
    return f"{task_prompt}\n\nReturn a single ```python code block containing the full file."

def extract_code(text: str) -> str:
    m = re.search(r"```python(.*)```", text, flags=re.S|re.I)
    return (m.group(1).strip() if m else text.strip())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-3-5-haiku-latest")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY not set")

    client = Anthropic(api_key=key)

    # Use Anthropic canonical aliases; reject unknown model names early
    allowed = {"claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"}
    if args.model not in allowed:
        print(f"[WARN] Model '{args.model}' not recognized. Falling back to 'claude-3-5-haiku-latest'.")
        args.model = "claude-3-5-haiku-latest"

    # Ask Claude ONCE; reuse for all N attempts (deterministic prompt)
    print(f"[INFO] Using Anthropic model: {args.model}")
    msg = client.messages.create(
        model=args.model,
        max_tokens=2500,
        temperature=args.temperature,
        system=SYSTEM,
        messages=[{"role": "user", "content": build_user_prompt()}],
    )
    text = "".join([blk.text for blk in msg.content if getattr(blk, "type", "") == "text"])
    code = extract_code(text)

    candidate_path = ROOT / "solve_backup.py"
    candidate_path.write_text(code, encoding="utf-8")
    print(f"[SAVED] {candidate_path} – wrapper solve.py remains untouched.")

    # Quick single check goes through wrapper (solve.py)
    def run_once():
        proc = subprocess.run(["python", "grade.py"], cwd=ROOT, text=True, capture_output=True)
        print(proc.stdout or proc.stderr)
        return proc.returncode == 0

    ok = run_once()
    print(f"[SINGLE PASS]: {ok}")

    # Multi-seed pass-rate through wrapper
    proc = subprocess.run(["python", "run_many.py", "--n", str(args.n), "--mod_path", "solve.py"],
                          cwd=ROOT, text=True, capture_output=True)
    print(proc.stdout or proc.stderr)

if __name__ == "__main__":
    main()
