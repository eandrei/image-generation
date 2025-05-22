# 🤖 AGENTS.md — Runtime & Repository Instructions
*Scope: entire directory tree rooted here*

---

## 1 📌 Mission

Implement and maintain an **Agentic Prompt → Image → Evaluator loop** that:

1. Accepts `prompt` (+ optional `reference_images`) via HTTP POST.  
2. Iteratively runs **Prompter → Image Generator → Evaluator**, max **10** rounds.  
3. Stops early if evaluator **score ≥ 90**.  
4. Returns JSON containing best image URL and full iteration logs.  

All modules must be **Python 3.11**, **async-first**, type-hinted, and unit-tested.

---

## 2 📂 Expected Directory Layout

```
.
├── agentic_image_gen/
│   ├── __init__.py
│   ├── main.py # FastAPI entrypoint
│   ├── loop_controller.py # orchestrates the loop
│   ├── prompter.py # GPT-4 prompt refiner
│   ├── image_gen.py # calls DALL·E / SDXL
│   ├── evaluator.py # GPT-4-Vision + CLIP scorer
│   └── storage.py # optional cloud uploads
├── tests/
│   └── ... # pytest suites (see §5)
├── requirements.txt
├── pre-commit-config.yaml
└── AGENTS.md # ← this file
```

*If you add new modules, keep them inside `agentic_image_gen/` and mirror tests in `tests/`.*

---

## 3 🧠 Agent Roles & APIs

| Agent            | Function                                   | Key I/O                                                     |
|------------------|--------------------------------------------|-------------------------------------------------------------|
| **Prompter**     | `generate_prompt(previous_prompt, feedback) -> str` | Improves prompt, preserves intent. |
| **ImageGen**     | `generate_image(prompt) -> str`            | Returns local path / URL of image. |
| **Evaluator**    | `evaluate_image(image_path, prompt) -> dict` | Returns `{score:int, feedback:str}` (*0–100*). |
| **LoopController** | `run_image_generation_loop(prompt, ref_imgs) -> dict` | Orchestrates iterations & termination logic. |

All public functions **must be fully type-hinted** and include Google-style docstrings.

---

## 4 🛠 Coding Conventions

* **Format:** `black --line-length 100`  
* **Imports:** `isort`  
* **Lint:** `ruff` (error level = E + F)  
* **Style:** PEP 8 + descriptive variable names  
* **Async:** prefer `async def` + `await` for I/O (HTTP, file).  
* **Env vars:** read `OPENAI_API_KEY`, `STABILITY_API_KEY`, etc. via `os.getenv`.

---

## 5 ✅ Tests & Programmatic Checks

| Check                 | Command                       | Goal                           |
|-----------------------|-------------------------------|--------------------------------|
| Unit tests           | `pytest -q`                   | All must pass (≥ 90 % coverage*) |
| Linting              | `ruff check .`                | 0 errors                       |
| Formatting           | `black --check . && isort --check-only .` | No diff                        |
| Type-checking†       | `pyright` or `mypy -p agentic_image_gen` | 0 errors                       |

\* Use `coverage run -m pytest && coverage xml` (threshold enforced in CI).  
† Type-checking is **optional** for small patches but **mandatory** for new modules.

> **Codex MUST run *all* checks above after every code change**.  
> If any fail, fix, re-commit, and re-run until the worktree is clean.

---

## 6 🔁 Iteration Logic (enforced by tests)

1. **MAX_ITERATIONS = 10**  
2. **SCORE_THRESHOLD = 90**  
3. On each loop:  
   - Prompter refines prompt.  
   - ImageGen produces image file.  
   - Evaluator returns `score` & `feedback`.  
4. If `score ≥ SCORE_THRESHOLD` → break.  
5. Return the image with the **highest score** plus histories (`prompt_history`, `image_history`, `feedback_history`).

Unit tests in `tests/test_loop_controller.py` verify this contract.

---

## 7 🗃 Git Rules (mirror system message)

1. **No new branches.** Work directly on current branch.  
2. Commit often with **conventional commits** (e.g. `feat: add evaluator`).  
3. If `pre-commit` fails, fix and retry.  
4. Leave repo **clean (`git status` → nothing to commit)** before finishing.  
5. Do **not** amend existing commits.

---

## 8 ✉️ Pull-Request Message Template

**Summary**
Short imperative description (≤ 50 chars).

**Details**
What was added / changed

Why it was necessary

Any API or config updates

**Tests**
Include ruff/black output & pytest results as terminal citations per system guidelines.

---

## 9 🔐 Secrets & Safety

- Never hard-code API keys; use env vars.  
- Strip metadata from images before upload.  
- Limit prompt temperature to ≤ 0.5 for determinism.

---

## 10 📣 Contact / Ownership

**Owner:** Andrei Gigirtu (<gigirtu.andrei@gmail.com>)  
For urgent issues mention `@gigirtu.andrei` in PR.

---

> **Read this file on every run.**  
> All rules here are **mandatory** unless overridden by a direct, explicit instruction in the prompt.

---

If new node packages are needed please provide the npm install command to install those instead of changing package.json.
For nhost apps please provide any migrations or changes in the table structures as sql queries to run manually instead of applying them automatically.
