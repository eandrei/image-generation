# 🤖 AGENTS.md — Runtime & Repository Instructions
*Scope: entire directory tree rooted here*

---

## 1 📌 Mission

Implement and maintain an **Agentic Prompt → Image → Evaluator loop** that:

1. Accepts `prompt` (+ optional `reference_images`) via the **Command Line Interface (CLI)**.
2. Iteratively runs **Prompter → Image Generator → Evaluator**, for a configurable number of rounds (e.g., `MAX_ITERATIONS`).
3. Stops early if evaluator **score ≥ SCORE_THRESHOLD** (e.g., 90).
4. Outputs JSON to standard output, containing the best image path/URL and full iteration logs.

All modules must be **Python 3.9+** (as per current environment), **async-first**, type-hinted, and unit-tested.

---

## 2 📂 Expected Directory Layout

```
.
├── agentic_image_gen/
│   ├── __init__.py
│   ├── __main__.py         # Main entry point for CLI
│   ├── cli.py              # argparse CLI handler
│   ├── loop_controller.py  # Orchestrates the loop
│   ├── prompter.py         # GPT-4 prompt refiner
│   ├── image_gen.py        # Calls OpenAI Responses API (gpt-4o) for image generation
│   ├── evaluator.py        # GPT-4o Vision scorer
│   ├── assistant_manager.py # Manages OpenAI Assistant for prompter
│   ├── thread_manager.py   # Manages OpenAI Threads for prompter
│   ├── run_orchestrator.py # Orchestrates runs with OpenAI Assistant
│   ├── message_sender.py   # Sends messages to OpenAI Threads
│   └── storage.py          # (Optional) Cloud uploads - currently not implemented
├── tests/
│   └── ... # pytest suites (see §5)
├── requirements.txt
├── .gitignore
├── pre-commit-config.yaml
└── AGENTS.md # ← this file
```

*If you add new modules, keep them inside `agentic_image_gen/` and mirror tests in `tests/`.*

---

## 3 🧠 Agent Roles & APIs

| Agent            | Module(s)                                 | Key Function(s) & I/O                                                                                                |
|------------------|-------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| **CLI Handler**  | `cli.py`                                  | `main()` → parses args, calls `run_image_generation_loop`.                                                           |
| **LoopController** | `loop_controller.py`                      | `run_image_generation_loop(prompt, ref_imgs) -> dict` → Orchestrates iterations & termination.                     |
| **Prompter**     | `prompter.py`, `assistant_manager.py`, `thread_manager.py`, `run_orchestrator.py`, `message_sender.py` | `generate_prompt(previous_prompt, feedback) -> str` → Improves prompt using an OpenAI Assistant.                         |
| **ImageGen**     | `image_gen.py`                            | `generate_image(prompt, ref_imgs?, prev_resp_id?) -> {image_path, response_id}` → Uses OpenAI Responses API (gpt-4o). Accepts optional reference images (local/URL) for initial generation, and `previous_response_id` for multi-turn edits. Returns local path to image & API response ID. |
| **Evaluator**    | `evaluator.py`                            | `evaluate_image(image_path, prompt) -> dict` → Returns `{score:int, feedback:str}` (0–100) using GPT-4o with Vision. |

All public functions **must be fully type-hinted** and include Google-style docstrings.

---

## 4 🛠 Coding Conventions

* **Format:** `black --line-length 100`
* **Imports:** `isort`
* **Lint:** `ruff` (error level = E + F)
* **Style:** PEP 8 + descriptive variable names
* **Async:** prefer `async def` + `await` for I/O (HTTP, file).
* **Env vars:** read `OPENAI_API_KEY`, etc. via `os.getenv`.

---

## 5 ✅ Tests & Programmatic Checks

| Check                 | Command                       | Goal                           |
|-----------------------|-------------------------------|--------------------------------|
| Unit tests           | `pytest -q`                   | All must pass (≥ 90 % coverage*) |
| Linting              | `ruff check .`                | 0 errors                       |
| Formatting           | `black --check . && isort --check-only .` | No diff                        |
| Type-checking†       | `pyright` or `mypy -p agentic_image_gen` | 0 errors                       |

\\* Use `coverage run -m pytest && coverage xml` (threshold enforced in CI).
† Type-checking is **optional** for small patches but **mandatory** for new modules.

> **You MUST run *all* checks above after every code change**.
> If any fail, fix, re-commit, and re-run until the worktree is clean.

---

## 6 🔁 Iteration Logic (enforced by tests)

1. **MAX_ITERATIONS** (e.g., 1 or 2, configurable in `loop_controller.py`)
2. **SCORE_THRESHOLD** (e.g., 90, configurable in `loop_controller.py`)
3. On each loop:
   - ImageGen produces an image. If it's not the first iteration, it uses the `previous_response_id` from the last generation for context.
   - Evaluator returns `score` & `feedback`.
   - If `score ≥ SCORE_THRESHOLD` → break.
   - If not breaking, Prompter refines prompt based on feedback.
4. Return the image with the **highest score** plus histories (`prompt_history`, `image_history`, `feedback_history`).

Unit tests in `tests/test_loop_controller.py` should verify this contract.

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

- Never hard-code API keys; use env vars. `assistant_config.json` (if containing assistant ID) should be in `.gitignore`.
- Strip metadata from images before upload (if applicable).
- Limit prompt temperature to ≤ 0.5 for determinism (where applicable, e.g., prompter or evaluator LLM calls). GPT-4o for evaluation is set to temperature 0.

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
