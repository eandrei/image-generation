# 🧠📸 Agentic Image Generation System

## 📌 Objective

Design and implement an autonomous system that:
- Takes a **natural language prompt** and optional **reference image(s)**
- Generates up to **10 iterations of images**
- Refines prompts and outputs through a **Prompter–Evaluator loop**
- Outputs the **best image**, selected based on evaluation score and prompt fidelity

---

## 🧠 System Overview

The system operates as a loop of:
1. **Prompter Agent** — improves prompts based on feedback
2. **Image Generator** — creates images using prompts
3. **Evaluator Agent** — scores and critiques generated images
4. **Loop Controller** — manages iteration flow and stopping conditions

### 🧬 Core Loop

```plaintext
User Input: [prompt + optional images]
    ↓
Prompter → Image Generator → Evaluator → Feedback
    ↓
Repeat up to 10 times or until score ≥ 90
    ↓
Final Output (best image + logs)
```

## 📦 Component Breakdown

### main.py — API Server
**Framework:** FastAPI

**Routes:**

`POST /generate`: Accepts request body with:

```json
{
  "prompt": "text description",
  "reference_images": ["base64_or_url", ...]
}
```

Calls `loop_controller.run_image_generation_loop()` and returns:

```json
{
  "best_image_url": "https://...",
  "final_score": 94,
  "prompt_history": [...],
  "image_history": [...],
  "feedback_history": [...]
}
```

### loop_controller.py — Core Logic
`run_image_generation_loop(prompt, reference_images) -> dict`

**Responsibilities:**

- Manage up to 10 iterations
- Stop early if evaluation score ≥ 90
- Track prompt_history, image_history, feedback_history
- Select final image based on highest score

### prompter.py — Prompt Refinement Agent
`generate_prompt(previous_prompt: str, feedback: list[str]) -> str`

Powered by OpenAI GPT-4 API

**Capabilities:**

- Refine prompt for clarity and precision
- Apply feedback from previous evaluation
- Avoid drift from original intent
Prompt Example:

```python
"Refine the following image generation prompt by applying the feedback. Prompt: 'A vintage ring on silk.' Feedback: 'Lighting is too harsh. Ring needs to be centered.'"
```

### assistant_manager.py — Assistant Lifecycle Helper
`create_assistant() -> str`

Handles creation and persistence of the OpenAI Assistant used by the prompter.
The assistant ID is stored in `assistant_config.json` so it can be reused in
future runs.

### image_gen.py — Image Generator
`generate_image(prompt: str) -> str`

Uses:

- OpenAI DALL·E (v3 or above)
- Optionally: SDXL via Stability API

Returns:

- Local path or public URL of generated image

**Requirements:**

- Save each image with iteration index
- Optionally store in S3, Firebase, or Nhost

### evaluator.py — Image Evaluator Agent
`evaluate_image(image_path: str, prompt: str) -> dict`

Powered by:

- GPT-4o with vision capabilities
- Optional: CLIP-based similarity scoring

Returns structured JSON:

```json
{
  "score": 82,
  "feedback": "The silk is not visible, and the lighting is too yellow. Consider soft white light and repositioning the ring to the center."
}
```

Must evaluate based on:

- Semantic alignment with the prompt
- Object presence and positioning
- Lighting, color, and visual tone

### storage.py — Cloud Uploads (Optional)
`upload_image_to_cloud(image_path: str) -> str`

**Providers:**

- Amazon S3
- Firebase
- Nhost

**Outputs:**

- Publicly accessible image URL

## 🔄 Iteration Workflow
1. User sends prompt and optional `reference_images` to `POST /generate`
2. Loop begins:
   - Prompter creates initial image prompt
   - Image generator creates image
   - Evaluator critiques and scores image
   - Prompter refines prompt using feedback
3. Loop continues until:
   - Score ≥ 90, OR
   - 10 iterations are completed
4. System returns:
   - Best image URL
   - Score
   - Full prompt / image / feedback history

## 🧪 Example Input / Output

**Input:**
```json
{
  "prompt": "A vintage ring on pink silk under soft daylight",
  "reference_images": []
}
```

**Output:**
```json
{
  "best_image_url": "https://cdn.indira.com/final_ring_9.png",
  "final_score": 93,
  "prompt_history": ["...", "..."],
  "image_history": ["image_1.png", "image_2.png", "..."],
  "feedback_history": ["Too much shadow.", "Better light but silk not visible.", "..."]
}
```

## 🛠 Libraries & Tools
- openai — GPT-4 + DALL·E API
- fastapi — web server
- Pillow, opencv-python — image preprocessing
- boto3, firebase_admin, nhost — cloud upload SDKs
- transformers — CLIP for similarity scoring (optional)
- uuid, os, datetime — for file versioning and storage

## 🔧 Environment Variables

Set the following variables before running the server:

- `OPENAI_API_KEY` – required for GPT-4 and DALL·E requests
- `STABILITY_API_KEY` – optional, enables SDXL image generation
- `NHOST_URL` and `NHOST_ANON_KEY` – optional, for Nhost storage
- `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` – optional, for S3 uploads

Example on Linux/macOS:

```bash
export OPENAI_API_KEY="sk-..."
export STABILITY_API_KEY="sdxl-..."  # optional
```

## 🚀 Running the API Server

Use [Uvicorn](https://www.uvicorn.org/) to start the FastAPI app:

```bash
uvicorn agentic_image_gen.main:app --reload
```

The server will listen on `http://127.0.0.1:8000`.  Send `POST /generate`
requests as described above to trigger the image-generation loop.

## 🧠 Prompter Agent Instructions
- Maintain original subject (do not change core topic)
- Improve specificity (e.g., lighting, background, positioning)
- Use feedback directly to revise wording
- Avoid hallucinations or shifts in meaning

## 👁 Evaluator Agent Instructions
- Score from 0–100
- Critique based on:
  - Accuracy to subject
  - Layout, composition
  - Color, lighting, background
- Provide feedback as a brief paragraph, actionable, and descriptive
- Ensure scoring is consistent across iterations

## ✅ Acceptance Criteria
- Accepts input prompt and optional image list via API
- Generates 1–10 image iterations
- Refines prompts based on prior evaluation
- Evaluator scores and critiques each image
- Terminates on high score or max loops
- Returns structured JSON with all history
- Fully modular architecture with testable components

## ➕ Optional Enhancements
- Add `user_id` field to track sessions
- Integrate front-end viewer to compare iterations
- Add “style guide” prompt anchoring (e.g., moodboard input)
- Provide full image metadata and tags

## 📎 Final Notes
- Use low temperature in GPT calls for deterministic behavior
- Log all prompt-feedback-image triplets to file or DB
- Consider retry mechanism for API failures
- Design for cloud deployment with async support
