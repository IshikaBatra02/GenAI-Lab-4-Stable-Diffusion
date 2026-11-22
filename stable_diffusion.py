"""Generate watercolor and pencil-sketch comparisons with Stable Diffusion v1.5."""
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
PROMPT_BASE = "A small peaceful study desk beside a window, books and a cup of tea"
STYLES = {
    "watercolor": "watercolor painting, soft washes, delicate paper texture",
    "pencil_sketch": "detailed pencil sketch, graphite shading, hand-drawn lines",
}


def main():
    output_dir = Path("lab_outputs/stable_diffusion")
    output_dir.mkdir(parents=True, exist_ok=True)
    token = os.getenv("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
        token=token,
    )
    pipe = pipe.to(device)
    # Keep the model's safety checker enabled.
    pipe.enable_attention_slicing()

    metadata = []
    for style_name, style_prompt in STYLES.items():
        prompt = f"{PROMPT_BASE}, {style_prompt}"
        for seed in (42, 43):
            generator = torch.Generator(device=device).manual_seed(seed)
            result = pipe(
                prompt=prompt,
                height=512,
                width=512,
                num_inference_steps=25,
                guidance_scale=7.5,
                generator=generator,
            )
            image = result.images[0]
            flagged = bool(result.nsfw_content_detected and result.nsfw_content_detected[0])
            if flagged:
                print(f"Safety checker flagged {style_name}, seed {seed}; image not saved.")
                continue
            filename = f"{style_name}_seed_{seed}.png"
            image.save(output_dir / filename)
            metadata.append({
                "file": filename,
                "style": style_name,
                "seed": seed,
                "prompt": prompt,
                "steps": 25,
                "guidance_scale": 7.5,
                "size": "512x512",
            })
            print(f"Saved {filename}")

    import json
    (output_dir / "generation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Finished. Outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
