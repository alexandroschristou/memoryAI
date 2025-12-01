import torch
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import imageio
import numpy as np
import os
from typing import Optional


# ---- MODEL LOADING ----

def load_svd_pipeline(
    model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt",
    device: str = "cuda",
) -> StableVideoDiffusionPipeline:
    """
    Load the Stable Video Diffusion image-to-video pipeline.

    Why:
    - We want a pre-trained, production-grade image-to-video model rather than
      training our own from scratch. SVD is actively maintained and integrated
      in diffusers, so it's a pragmatic baseline.
    """
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # half precision to reduce VRAM and increase throughput
    )

    # Move the pipeline to GPU for performance.
    pipe = pipe.to(device)

    # Enable memory optimizations where available.
    # This is important because diffusion models are VRAM heavy, and we want
    # to avoid OOM errors early.
    pipe.enable_model_cpu_offload()  # or pipe.enable_xformers_memory_efficient_attention()

    return pipe


# ---- IMAGE PREPROCESSING ----

def load_and_preprocess_image(
    image_path: str,
    target_size: tuple[int, int] = (576, 1024),
) -> Image.Image:
    """
    Load an image from disk and resize it to the SVD-friendly resolution.

    Why:
    - SVD expects a fixed resolution. We want to standardize inputs so that
      inference is stable and we don't run into shape-related errors.
    - We use a simple resize here as an MVP. Later, we can replace this with
      smart cropping/padding to preserve aspect ratio and important content.
    """
    image = Image.open(image_path).convert("RGB")

    # For now, we use a straightforward resize. This might distort aspect ratio
    # slightly, but it keeps the implementation simple for the MVP.
    # Once we validate the pipeline, we can refine this into:
    # - center crop + pad
    # - face/body-aware cropping using detectors
    image = image.resize(target_size, resample=Image.BICUBIC)

    return image


# ---- VIDEO GENERATION ----

def generate_video_frames(
    pipe: StableVideoDiffusionPipeline,
    image: Image.Image,
    num_frames: int = 14,
    fps: int = 7,
    motion_bucket_id: int = 80,
    noise_aug_strength: float = 0.02,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Given a preprocessed image and the SVD pipeline, generate video frames.

    Why these parameters:
    - num_frames: SVD is designed for short clips; 14 frames at 7 fps gives us
      a ~2 second clip, enough to show subtle motion but still cheap to compute.
    - fps: lower fps reduces file size and is more forgiving when motion is subtle.
    - motion_bucket_id: lower values => smaller motion; higher => more movement.
      For "memory revival", we favor gentle movement to preserve emotional realism.
    - noise_aug_strength: small noise helps avoid static artifacts while still
      keeping identity consistent.
    - seed: deterministic output for the same input, useful for debugging and
      for a consistent user experience.
    """
    generator = None
    if seed is not None:
        generator = torch.manual_seed(seed)

    # The pipe returns frames as a tensor or list of PIL images depending on version.
    # We rely on the documented API from diffusers for SVD.
    result = pipe(
        image,
        decode_chunk_size=8,        # trade-off between speed and memory
        num_frames=num_frames,
        fps=fps,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        generator=generator,
    )

    frames = result.frames  # shape: (batch, num_frames, H, W, C) or list[list[PIL]]
    # Handle both common formats in a robust way.

    if isinstance(frames, list):
        # Assume frames is a list of lists of PIL images (batch_size x num_frames).
        pil_frames = frames[0]  # batch_size 1
        frames_np = np.stack([np.array(f) for f in pil_frames], axis=0)
    else:
        # Assume frames is a numpy/torch array.
        if isinstance(frames, torch.Tensor):
            frames_np = frames.cpu().numpy()
        else:
            frames_np = np.array(frames)

        # frames_np shape: (batch, num_frames, H, W, C)
        frames_np = frames_np[0]  # take first in batch

    return frames_np  # (num_frames, H, W, C)


# ---- VIDEO ENCODING ----

def save_frames_as_mp4(
    frames: np.ndarray,
    output_path: str,
    fps: int = 7,
):
    """
    Save an array of frames (T, H, W, C) as an MP4 video using imageio.

    Why imageio:
    - Simple dependency for quick MVP.
    - Later we can switch to ffmpeg directly for more control over codec, bitrate, etc.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Ensure uint8 encoding for standard video formats
    frames_uint8 = frames.astype(np.uint8)

    # imageio will call ffmpeg under the hood if imageio-ffmpeg is installed.
    # We keep this simple for now. Later, we can expose bitrate, codec, etc.
    imageio.mimsave(
        output_path,
        frames_uint8,
        fps=fps,
        quality=8,  # moderate quality to balance size and visual fidelity
    )


# ---- PUBLIC PIPELINE ENTRYPOINT ----

def generate_video_from_image(
    input_image_path: str,
    output_video_path: str,
    model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt",
    device: str = "cuda",
    target_size: tuple[int, int] = (576, 1024),
    num_frames: int = 14,
    fps: int = 7,
    motion_bucket_id: int = 80,
    noise_aug_strength: float = 0.02,
    seed: Optional[int] = 42,
):
    """
    Full end-to-end pipeline: image path -> video file path.

    Why this abstraction:
    - This is the function your future web backend will call.
    - It hides all model / preprocessing details so you can swap the underlying
      model or parameters later without touching the HTTP layer.
    """
    # 1. Load model
    pipe = load_svd_pipeline(model_id=model_id, device=device)

    # 2. Preprocess input image
    image = load_and_preprocess_image(input_image_path, target_size=target_size)

    # 3. Generate video frames
    frames = generate_video_frames(
        pipe=pipe,
        image=image,
        num_frames=num_frames,
        fps=fps,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        seed=seed,
    )

    # 4. Save frames to disk as mp4
    save_frames_as_mp4(frames, output_video_path, fps=fps)


if __name__ == "__main__":
    # Example usage:
    # For a first test, put an image at "input.jpg" and run:
    #   python image_to_video_svd.py
    #
    # This simple main block exists purely for manual experimentation.
    input_path = "input.jpg"
    output_path = "outputs/output.mp4"

    generate_video_from_image(
        input_image_path=input_path,
        output_video_path=output_path,
        # You can tweak the parameters below to change the "feel" of the motion.
        num_frames=14,
        fps=7,
        motion_bucket_id=80,     # try lower (~40) for ultra-subtle movement, higher (~120) for more motion
        noise_aug_strength=0.02,
        seed=42,
    )
