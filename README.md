# Lab 4 — Text-to-Image Generation with Stable Diffusion

Compares two visual styles using Stable Diffusion v1.5: watercolor and pencil sketch.

## Experiment settings
- Model: `stable-diffusion-v1-5/stable-diffusion-v1-5`
- Image size: 512×512
- Inference steps: 25
- Guidance scale: 7.5
- Seeds: 42 and 43 for repeatable prompt comparisons

## Setup
Python 3.10 or 3.11 is recommended.
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
# Set a Hugging Face token if the model requires gated access:
# Windows PowerShell: $env:HF_TOKEN="your_token"
# macOS/Linux: export HF_TOKEN="your_token"
python stable_diffusion.py
```

Outputs are saved in `lab_outputs/stable_diffusion/`. GPU with sufficient VRAM is strongly recommended. The safety checker remains enabled; flagged images are not saved.

## Human evaluation
Review the generated images for prompt adherence, style clarity, visual quality, and consistency. Record human scores separately; this script does not fabricate evaluation scores.
