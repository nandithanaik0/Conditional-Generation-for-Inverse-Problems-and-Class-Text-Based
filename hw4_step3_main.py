

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from pathlib import Path

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import csv


# ---------------
def load_pipeline(model_id: str = "sd-legacy/stable-diffusion-v1-5"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe, device


def run_part_a(pipe, device):

    output_dir = Path("results/step3_part_a")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = 123
    generator = torch.Generator(device=device).manual_seed(seed)
    torch.manual_seed(seed)

    prompt = ""
    negative_prompt = None

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=0.0,   
        eta=0.0,
        generator=generator,
    )

    image = result.images[0]
    out_path = output_dir / "part_a_uncond_cfg0.png"
    image.save(out_path)

def get_part_b_prompts():
  
    topics_and_prompts = [
        (
            "futuristic_city",
            "simple",
            "A futuristic city street at night",
        ),
        (
            "futuristic_city",
            "medium",
            "A futuristic city street at night glowing with neon signs, people walking under bright holograms",
        ),
        (
            "futuristic_city",
            "long",
            "A bustling futuristic city street at night, drenched in neon lights and holographic billboards, with crowds of people in cyberpunk clothing walking past reflective puddles and flying cars streaking through a hazy sky",
        ),

        (
            "dragon_castle",
            "simple",
            "A dragon flying over a medieval castle",
        ),
        (
            "dragon_castle",
            "medium",
            "A large dragon flying over a medieval stone castle at sunset",
        ),
        (
            "dragon_castle",
            "long",
            "A majestic dragon with shimmering scales circling a towering medieval stone castle at sunset, warm orange light illuminating the turrets while tiny figures watch from the battlements below",
        ),

        (
            "cozy_reading",
            "simple",
            "A cozy reading corner with a chair and a lamp",
        ),
        (
            "cozy_reading",
            "medium",
            "A cozy reading corner with a soft armchair, a warm lamp, and a small bookshelf",
        ),
        (
            "cozy_reading",
            "long",
            "A cozy reading nook with a plush armchair, a warm golden floor lamp, overflowing bookshelves, a steaming mug of tea on a wooden side table, and soft daylight filtering through a nearby window",
        ),

        (
            "cat_chef",
            "simple",
            "A cute cat wearing a chef hat",
        ),
        (
            "cat_chef",
            "medium",
            "A cute orange cat wearing a chef hat, standing in a kitchen",
        ),
        (
            "cat_chef",
            "long",
            "A cute fluffy orange cat wearing a big white chef hat and apron, happily cooking in a bright modern kitchen, surrounded by vegetables, pots, and a tiny cat-sized cutting board",
        ),

        (
            "lake_mountains",
            "simple",
            "A calm lake with mountains in the background",
        ),
        (
            "lake_mountains",
            "medium",
            "A calm lake with tall snow-capped mountains in the background under a blue sky",
        ),
        (
            "lake_mountains",
            "long",
            "A tranquil crystal-clear lake reflecting tall snow-capped mountains under a bright blue sky, with pine trees along the shoreline and a small wooden dock extending into the water",
        ),
    ]
    return topics_and_prompts



def run_part_b(pipe, device):

    output_dir = Path("results/step3_part_b")
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = get_part_b_prompts()

    CFG_SCALE = 10.0
    ETA = 0.0
    NUM_STEPS = 50
    NEGATIVE_PROMPT = None  

    base_seed = 1000

    meta_info = []  

    for idx, (topic_name, version_name, prompt) in enumerate(prompts):
        print(f"\n[Prompt {idx+1}/15] {topic_name} - {version_name}")
        print("Prompt:", prompt)

        seed = base_seed + idx
        generator = torch.Generator(device=device).manual_seed(seed)

        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=NUM_STEPS,
            guidance_scale=CFG_SCALE,
            eta=ETA,
            generator=generator,
        )

        image = result.images[0]
        filename = f"{idx+1:02d}_{topic_name}_{version_name}.png"
        out_path = output_dir / filename
        image.save(out_path)
        print(f"[PART (b)] Saved image to: {out_path}")

        meta_info.append(
            {
                "index": idx + 1,
                "topic": topic_name,
                "version": version_name,
                "prompt": prompt,
                "seed": seed,
                "filename": filename,
            }
        )

    print("\n Generated 15 images.")

    return meta_info




def load_clip_model(device, model_name: str = "openai/clip-vit-base-patch32"):
    
    print(f"\n[PART (c/d)] Loading CLIP model: {model_name}")
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    clip_model.eval()
    return clip_model, clip_processor


def compute_clip_similarity(clip_model, clip_processor, image_path: Path, text: str, device: str):

    image = Image.open(image_path).convert("RGB")

    inputs = clip_processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_embeds = outputs.image_embeds  # [1, D]
        text_embeds = outputs.text_embeds    # [1, D]

    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    # Cosine similarity
    cos_sim = (image_embeds * text_embeds).sum(dim=-1).item()

    scaled_score = (cos_sim + 1.0) / 2.0 * 10.0

    return cos_sim, scaled_score


def run_part_c(meta_info, device, clip_model, clip_processor):
   

    images_dir = Path("results/step3_part_b")
    output_csv = images_dir / "clip_similarity_scores.csv"

  
    fieldnames = [
        "index",
        "topic",
        "version",
        "prompt",
        "seed",
        "filename",
        "clip_cosine_similarity",
        "clip_score_0_to_10",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for entry in meta_info:
            image_path = images_dir / entry["filename"]
            cos_sim, scaled_score = compute_clip_similarity(
                clip_model,
                clip_processor,
                image_path,
                entry["prompt"],
                device,
            )

            entry["clip_cosine_similarity"] = cos_sim
            entry["clip_score_0_to_10"] = scaled_score

            writer.writerow(
                {
                    "index": entry["index"],
                    "topic": entry["topic"],
                    "version": entry["version"],
                    "prompt": entry["prompt"],
                    "seed": entry["seed"],
                    "filename": entry["filename"],
                    "clip_cosine_similarity": cos_sim,
                    "clip_score_0_to_10": scaled_score,
                }
            )

    print(f"[PART (c)] Saved CLIP similarity scores to: {output_csv}")

 
    print("\n[PART (c)] Example CLIP scores (0–10 scale):")
    for entry in meta_info:
        print(
            f"  #{entry['index']:02d} | {entry['topic']:15s} | {entry['version']:6s} | "
            f"CLIP ~ {entry['clip_score_0_to_10']:.2f}/10"
        )



def run_part_d_simple(pipe, device, clip_model, clip_processor):
  

    prompt = "A futuristic city street at night glowing with neon lights."
    negative_prompt = "no people, no neon lights, no holograms, no flying cars"

    guidance_scales = [0.0, 2.0, 5.0, 8.0, 12.0, 15.0]
    NUM_STEPS = 50

    output_dir = Path("results/step3_part_d_simple")
    output_dir.mkdir(parents=True, exist_ok=True)

    # For reproducibility
    base_seed = 3000

    for i, scale in enumerate(guidance_scales):
        print(f"\n[CFG={scale}] Generating...")

        generator = torch.Generator(device=device).manual_seed(base_seed + i)

 
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=NUM_STEPS,
            eta=0.0,
            guidance_scale=scale,
            generator=generator
        ).images[0]

  
        filename = f"guidance_{str(scale).replace('.', '_')}.png"
        out_path = output_dir / filename
        image.save(out_path)
        print("Saved:", out_path)

        inputs = clip_processor(
            text=[prompt],
            images=[image],
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = clip_model(**inputs)
            img_embed = outputs.image_embeds
            txt_embed = outputs.text_embeds

        # Normalize vectors
        img_embed = img_embed / img_embed.norm(p=2, dim=-1, keepdim=True)
        txt_embed = txt_embed / txt_embed.norm(p=2, dim=-1, keepdim=True)

        # Cosine similarity
        similarity = (img_embed * txt_embed).sum(dim=-1).item()

        print(f"[CFG={scale}] CLIP similarity: {similarity:.4f}")


def main():
    pipe, device = load_pipeline()
    run_part_a(pipe, device)
    meta_info = run_part_b(pipe, device)


    clip_model, clip_processor = load_clip_model(device)

    run_part_c(meta_info, device, clip_model, clip_processor)
    run_part_d_simple(pipe, device, clip_model, clip_processor)


if __name__ == "__main__":
    main()
