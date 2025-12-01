"""
A minimal, self-contained script to run the entire image → video pipeline
with Stable Video Diffusion on RunPod.

Why this script exists:
- You asked for the simplest possible entrypoint.
- No command-line arguments.
- No complex structure.
- Just edit the variables below, run "python run_pipeline.py", get your video.
"""

import torch
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import imageio
import numpy as np
import os

def choose_device(preferred: str = "cuda") -> str:
    """
    Decide which device to use.

    Why:
    - Hardcoding 'cuda' makes the script brittle and leads to confusing
      errors when CUDA isn't actually usable.
    - Here we detect availability and fall back to CPU if needed, with
      a clear message so you know what's going on.
    """
    if preferred == "cuda":
        if torch.cuda.is_available():
            # We can refine this later (e.g. select specific GPU index),
            # but 'cuda' (device 0) is fine for a single-GPU RunPod.
            print("[Device] Using CUDA GPU")
            return "cuda"
        else:
            print("[Device] WARNING: CUDA requested but not available. Falling back to CPU.")
            return "cpu"
    else:
        # If you ever call choose_device('cpu'), just return cpu directly.
        print("[Device] Using CPU (explicitly requested)")
        return "cpu"

# -----------------------------------------------------
# CONFIGURATION (edit these for quick experiments)
# -----------------------------------------------------

INPUT_IMAGE = "example_input.jpg"                     # upload this to RunPod
OUTPUT_VIDEO = "outputs/output.mp4"                  # video will be saved here
MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"

TARGET_SIZE = (1024, 576)          # use the canonical SVD size
NUM_FRAMES = 12                    # slightly shorter clip
FPS = 7
# subtle, slow movement
MOTION_BUCKET_ID = 20              # very subtle motion
NOISE_AUG_STRENGTH = 0.005         # almost no noise
SEED = 42

DEVICE = choose_device()  # will be "cuda" or "cpu" depending on environment# RunPod GPU



# -----------------------------------------------------
# STEP 1 — Load Model
# -----------------------------------------------------

def load_model():
    """
    Loads the Stable Video Diffusion model in float16 on GPU.
    Why float16:
    - Saves VRAM
    - Faster inference
    """
    print("Loading model...")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16
    ).to(DEVICE)

    # Optional: offload some layers to CPU if needed
    # pipe.enable_model_cpu_offload()

    return pipe



# -----------------------------------------------------
# STEP 2 — Load + Preprocess Image
# -----------------------------------------------------

def load_image(path):
    """
    Loads and resizes the input image.

    Why simple resize:
    - For MVP simplicity
    - Later we can add face/body-aware cropping or padding
    """
    print("Loading and preprocessing image...")
    image = Image.open(path).convert("RGB")
    image = image.resize(TARGET_SIZE, Image.BICUBIC)
    return image



# -----------------------------------------------------
# STEP 3 — Generate Frames
# -----------------------------------------------------

def generate_frames(pipe, image):
    """
    Runs diffusion model to create frames.

    Why these parameters:
    - NUM_FRAMES: keeps output short and GPU cost low
    - MOTION_BUCKET_ID: controls movement intensity
    - NOISE_AUG_STRENGTH: keeps identity stable
    """
    print("Generating frames...")

    generator = torch.manual_seed(SEED)

    result = pipe(
        image,
        decode_chunk_size=4,  # smaller chunk => slightly more stable
        num_frames=NUM_FRAMES,
        fps=FPS,
        motion_bucket_id=MOTION_BUCKET_ID,
        noise_aug_strength=NOISE_AUG_STRENGTH,
        num_inference_steps=12,  # not too high, avoids over-baking details
        min_guidance_scale=1.0,  # keep guidance low so model doesn't overcook
        max_guidance_scale=1.5,
        generator=generator,
    )

    frames = result.frames

    # Handle different diffusers versions
    if isinstance(frames, list):
        pil_frames = frames[0]
        frames_np = np.stack([np.array(f) for f in pil_frames])
    else:
        if isinstance(frames, torch.Tensor):
            frames_np = frames.cpu().numpy()[0]
        else:
            frames_np = np.array(frames)[0]

    return frames_np

def blend_with_original(frames: np.ndarray, original_np: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    """
    Blend every generated frame with the original image to reduce deformation.

    Why:
    - The diffusion model tends to "repaint" details each frame, which creates
      obvious AI artifacts, especially on faces and hands.
    - By linearly mixing each frame with the original photo, we:
        * keep identity and clothing very close to the original
        * allow only subtle motion to remain
        * make the result feel more like a living photograph instead of a hallucination.
    """
    # frames: (T, H, W, C)
    # original_np: (H, W, C)
    print("Blending generated frames with original image...")

    # Ensure float32 for safe blending
    frames_f = frames.astype(np.float32)
    # Broadcast original over time dimension
    original_broadcast = np.broadcast_to(original_np, frames_f.shape)

    # Linear blend
    blended = alpha * frames_f + (1.0 - alpha) * original_broadcast

    # Clip and cast back to uint8
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended

# -----------------------------------------------------
# STEP 4 — Save as Video
# -----------------------------------------------------

def save_video(frames):
    """
    Saves frames to MP4 using imageio.
    """
    print("Saving video...")

    output_dir = os.path.dirname(OUTPUT_VIDEO)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    frames_uint8 = frames.astype(np.uint8)

    imageio.mimsave(
        OUTPUT_VIDEO,
        frames_uint8,
        fps=FPS,
        quality=8
    )

    print(f"Video saved at: {OUTPUT_VIDEO}")




# -----------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------

def main():
    pipe = load_model()
    image = load_image(INPUT_IMAGE)

    # Keep original as numpy for later blending
    original_np = np.array(image).astype(np.float32)

    frames = generate_frames(pipe, image)

    # Blend generated frames with the original
    frames = blend_with_original(frames, original_np, alpha=0.65)

    save_video(frames)



if __name__ == "__main__":
    main()
