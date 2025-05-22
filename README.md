# 🧠📸 Agentic Image Generation System

## 📌 Objective

Design and implement an autonomous system that:
- Takes a **natural language prompt** and optional **reference image(s)** via a Command Line Interface (CLI).
- Generates images through an iterative **Prompter–Image Generator–Evaluator loop**.
- Refines prompts and image outputs based on feedback, potentially over multiple turns using context from previous generations.
- Outputs the **best image** (as a local file path) and detailed logs, selected based on evaluation score and prompt fidelity.

---

## 🧠 System Overview

The system operates as a loop controlled by `loop_controller.py`:
1. **Input**: User provides an initial prompt and optional reference image paths/URLs via the CLI.
2. **Image Generator (`image_gen.py`)**: Creates an image using the OpenAI Responses API (with `gpt-4o`).
   - For the first iteration, it uses the initial prompt and any reference images.
   - For subsequent iterations, it uses the refined prompt and the `previous_response_id` from the last generation to maintain context for multi-turn editing/refinement.
3. **Evaluator Agent (`evaluator.py`)**: Scores (0-100) and critiques the generated image using GPT-4o with Vision, comparing it against the current prompt.
4. **Prompter Agent (`prompter.py`)**: If the score is below `SCORE_THRESHOLD` and `MAX_ITERATIONS` is not reached, this agent refines the prompt based on the evaluator's feedback. This uses an OpenAI Assistant.
5. **Loop**: Steps 2-4 repeat until `MAX_ITERATIONS` is reached or `SCORE_THRESHOLD` is met.
6. **Output**: The system prints a JSON object containing the path to the best image, its score, and histories of prompts, image paths, and feedback.

### 🧬 Core Loop Diagram

```plaintext
User CLI Input: [prompt + optional reference images (local/URL)]
    ↓
[Iteration 1]
Image Generator (prompt, ref_images?) → Evaluator → Feedback
    ↓ (if score < threshold & iterations < max)
Prompter (uses feedback to create new_prompt)
    ↓
[Iteration 2+]
Image Generator (new_prompt, previous_response_id) → Evaluator → Feedback
    ↓ ...
Final Output (best image path + logs)
```

## 📦 Component Breakdown

### `__main__.py` & `cli.py` — Command Line Interface
- **`__main__.py`**: Entry point when running `python -m agentic_image_gen`.
- **`cli.py`**: Uses `argparse` to handle command-line arguments:
    - `prompt`: The initial text prompt (required).
    - `--refs`: Optional list of reference image paths (local or HTTP/HTTPS URLs).
- Invokes `loop_controller.run_image_generation_loop`.

### `loop_controller.py` — Core Logic
- `run_image_generation_loop(prompt: str, reference_images: list[str] | None) -> dict`
- Manages the iterative generation loop (up to `MAX_ITERATIONS`).
- Stops early if `SCORE_THRESHOLD` is met.
- Tracks history of prompts, image paths, feedback, and OpenAI API response IDs for multi-turn generation.
- Selects the final best image based on the highest score.

### `prompter.py` (+ `assistant_manager.py`, `thread_manager.py`, `run_orchestrator.py`, `message_sender.py`)
- `prompter.generate_prompt(previous_prompt: str, feedback: list[str]) -> str`
- Uses an OpenAI Assistant (GPT-4 based) to refine prompts based on evaluation feedback.
- The associated manager modules handle Assistant creation, thread management, and run orchestration.

### `image_gen.py` — Image Generator
- `generate_image(prompt: str, reference_images: list[str] | None = None, previous_response_id: str | None = None) -> dict`
- Uses the **OpenAI Responses API** with the `gpt-4o` model and the `image_generation` tool.
- **Input Handling**:
    - For initial generation: Takes a text `prompt` and an optional list of `reference_images` (local paths or URLs, which are fetched and base64 encoded).
    - For iterative refinement: Takes the new `prompt` and `previous_response_id` to continue the generation context.
- **Output**: Returns a dictionary `{"image_path": "/path/to/temp_image.png", "response_id": "openai_response_id"}`. The image is saved to a temporary local file.

### `evaluator.py` — Image Evaluator Agent
- `evaluate_image(image_path: str, prompt: str) -> dict`
- Uses **GPT-4o with Vision capabilities** via the Chat Completions API (JSON mode enabled).
- Evaluates the generated image (from `image_path`) against the `prompt` it was generated for.
- Returns a structured JSON: `{"score": int, "feedback": "textual critique"}`.

### `storage.py` — Cloud Uploads (Optional)
- `upload_image_to_cloud(image_path: str) -> str`
- This module exists but is not currently integrated into the main loop. It could be used to upload generated images to services like S3, Firebase, or Nhost.

## 🔄 Iteration Workflow
1. User executes the script via CLI: `python -m agentic_image_gen "User prompt here" --refs "img1.jpg" "url/to/img2.png"`.
2. The `loop_controller` starts:
   a. **Iteration 1**: `image_gen` is called with the initial prompt and reference images. It returns the path to the generated image and an API `response_id`.
   b. `evaluator` scores the image and provides feedback.
   c. If score is low and iterations continue, `prompter` refines the prompt.
   d. **Iteration 2+**: `image_gen` is called with the new prompt and the `previous_response_id`. Reference images are not re-sent.
   e. Steps b-d repeat.
3. Loop terminates when `SCORE_THRESHOLD` is met or `MAX_ITERATIONS` are completed.
4. The system prints a JSON blob to standard output with the path to the best image, its score, and full histories.

## 🧪 Example Usage

**Command:**
```bash
python -m agentic_image_gen "A futuristic car in a neon-lit city, inspired by cyberpunk art" --refs "./local_cyberpunk_style.jpg" "https://example.com/another_style_ref.png"
```

**Expected Output (to stdout):**
```json
{
  "best_image_url": "/tmp/generated_image_xxxxxxxx.png", // Path to the best image
  "final_score": 92,
  "prompt_history": ["Initial prompt...", "Refined prompt..."],
  "image_history": ["/tmp/generated_image_aaaaaaa.png", "/tmp/generated_image_bbbbbbb.png"],
  "feedback_history": ["Critique for image 1...", "Critique for image 2..."],
  "thread_id": "thread_xxxxxxxxxxxx"
}
```

## 🛠 Libraries & Tools
- `openai` (version 1.0.0+): For interacting with OpenAI APIs (GPT-4o, Assistants).
- `aiohttp`: For asynchronously fetching images from URLs (used in `image_gen.py`).
- `Pillow`, `opencv-python` (Potentially used by underlying image processing, or could be added for explicit preprocessing if needed. Not directly in `requirements.txt` currently but good to be aware of for image tasks).
- `pytest`: For running tests.

## 🔧 Environment Variables

Set the following environment variables before running:

- `OPENAI_API_KEY`: **Required** for all OpenAI API requests.

Example on Linux/macOS:
```bash
export OPENAI_API_KEY="sk-YourSecretKey"
```

## 🚀 Running the Application

1.  **Ensure Python 3.9+ is installed.**
2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set the `OPENAI_API_KEY` environment variable.**
5.  **Run the CLI:**
    ```bash
    python -m agentic_image_gen "Your text prompt" --refs "optional/path/to/image.jpg" --quality <quality> --size <size> --background <background> --format <format>
    ```
    - The prompt is a required positional argument.
    - `--refs` can take one or more local file paths or HTTP/HTTPS URLs.
    - `--quality`: Image quality (auto, low, medium, high). Default: `high`.
    - `--size`: Image dimensions (auto, 1024x1024, 1024x1536, 1536x1024). Default: `1024x1024`.
    - `--background`: Background style (auto, opaque, transparent). Default: `transparent` (for PNG/WEBP).
    - `--format`: Output image format (png, jpeg, webp). Default: `png`.

## ✨ Tailoring for Jewelry Product Photography

This system is well-suited for generating jewelry product images. Here are tips for optimal results:

*   **Use Specific Prompts:** Describe the jewelry type, materials (e.g., "18k yellow gold," "platinum"), gemstones ("VS1 diamond," "Ceylon sapphire"), setting style, desired lighting ("soft studio lighting," "dramatic single light source"), background ("clean white seamless," "dark reflective surface"), and composition ("macro shot of the center stone," "ring on a textured silk fabric").
*   **Leverage Reference Images (`--refs`):**
    *   Provide images of the *exact product* if you have them to guide the AI on specific details.
    *   Use images that showcase the *desired style* of lighting, composition, or background.
    *   Include images of similar jewelry pieces as inspiration.
*   **Utilize Image Parameters:**
    *   `--quality high` (default) is recommended for detail.
    *   `--background transparent --format png` (defaults) is excellent for e-commerce, allowing easy placement on various website backgrounds.
    *   Choose `--size` appropriate for your website or social media (e.g., `1024x1024` for square, `1024x1536` for portrait like Instagram stories).

## 🧠 Prompter Agent Instructions (Jewelry Focus)
When configuring the OpenAI Assistant used by the `prompter.py` module, tailor its instructions for jewelry:

-   "Your goal is to refine image generation prompts specifically for high-quality jewelry product photography."
-   "Focus on accurately describing the jewelry type, materials (gold, silver, platinum, specific gemstones), and any unique features."
-   "Incorporate details about desired lighting (e.g., 'soft studio light,' 'rim lighting to highlight facets'), background (e.g., 'clean white,' 'neutral gradient,' 'reflective surface'), and composition (e.g., 'macro shot,' 'flat lay,' 'angled view')."
-   "If the user's feedback mentions issues like poor sparkle, unclear details, or bad lighting, suggest specific prompt modifications to address these, such as adding terms like 'enhance gemstone brilliance,' 'sharp focus on details,' or 'even, diffused lighting.'"
-   "Prioritize commercial appeal and accuracy for e-commerce and marketing use."
-   "Avoid overly artistic or abstract interpretations unless specifically requested."

## 👁 Evaluator Agent Instructions
- Score from 0–100 based on adherence to the prompt, image quality, and desired aesthetics.
- Critique based on:
  - Accuracy to subject and specific details in the prompt.
  - Layout, composition, color, lighting, and overall visual appeal.
- Provide feedback as a brief, actionable paragraph.
- Ensure scoring is consistent across iterations.

## ✅ Acceptance Criteria
- Accepts input prompt and optional reference image list (local/URL) via CLI.
- Generates image iterations using OpenAI Responses API (`gpt-4o`).
- Supports multi-turn image generation by passing context (`previous_response_id`).
- Refines prompts based on prior evaluation using an OpenAI Assistant.
- Evaluator (GPT-4o Vision) scores and critiques each image in JSON format.
- Terminates on high score (`SCORE_THRESHOLD`) or max loops (`MAX_ITERATIONS`).
- Returns structured JSON to stdout with all history and the path to the best generated image.
- Fully modular architecture with asynchronous operations.

## ➕ Optional Enhancements
- Integrate `storage.py` to upload the final best image to a cloud service and return a public URL.
- Add more sophisticated error handling and retry mechanisms for API calls.
- Implement a front-end viewer to display iterations and allow easier interaction.
- Expand `evaluator.py` to use more complex evaluation metrics or even human-in-the-loop feedback.

## 📎 Final Notes
- The system relies on temporary local files for storing generated images between steps. Ensure the temporary directory is writable.
- `assistant_config.json` is used by `assistant_manager.py` to store the OpenAI Assistant ID and should be in `.gitignore`.
