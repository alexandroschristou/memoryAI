# src/pipeline/image_to_video_svd.py

import os
from typing import Optional, Tuple

import torch
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import imageio
import numpy as np


def load_svd_pipeline(
    model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt",
    device: str = "cuda",
    enable_cpu_offload: bool = False,
) -> StableVideoDiffusionPipeline:
    """
    Load the Stable Video Diffusion pipeline.

    Why this exists:
    - We want a single place in the codebase responsible for creating the
      model pipeline. This makes it easier to:
        * change model versions,
        * change device placement policy,
        * add logging / monitoring later.
    """
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # saves VRAM and speeds up inference on modern GPUs
    )

    # Move to the requested device. On RunPod this will typically be 'cuda'.
    pipe = pipe.to(device)

    # Optional: enable CPU offload if VRAM is tight.
    # On a big RunPod GPU like an A100 you might not need this, but the flag
    # gives us flexibility if you ever run on smaller hardware.
    if enable_cpu_offload:
        pipe.enable_model_cpu_offload()

    return pipe


def load_and_preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (576, 1024),
) -> Image.Image:
    """
    Load an image and resize it to a resolution friendly to SVD.

    Why simple resize:
    - For the first MVP we want to minimize implementation complexity.
    - A pure resize is easy and stable; once the pipeline is validated, we
      can evolve this into:
        * aspect ratio preserving resize + padding,
        * content-aware cropping using detectors (face / body).
    """
    image = Image.open(image_path).convert("RGB")

    # Resize to the target resolution. This may distort the aspect ratio slightly,
    # but it's acceptable for a first iteration. We'll refine this if it hurts
    # realism too much on your real data.
    image = image.resize(target_size, resample=Image.BICUBIC)

    return image


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
    Run SVD on a single image and return frames as a numpy array.

    Design decisions:
    - num_frames: keeps clip short and compute manageable.
    - fps: relatively low fps suits subtle "breathing / swaying" motion and
      keeps file sizes reasonable.
    - motion_bucket_id: controls motion intensity. For "memory revival", we
      bias towards subtle motion so the clip feels plausible and not uncanny.
    - noise_aug_strength: small noise helps avoid static artifacts while still
      keeping the original identity clear.
    - seed: reproducible results are crucial when debugging or when offering
      "regenerate" options that should be consistent.
    """
    generator = None
    if seed is not None:
        generator = torch.manual_seed(seed)

    result = pipe(
        image,
        decode_chunk_size=8,     # balances VRAM vs speed
        num_frames=num_frames,
        fps=fps,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        generator=generator,
    )

    frames = result.frames

    # Handle possible return types robustly in case diffusers versions differ.
    if isinstance(frames, list):
        # Most common: list of list of PIL images => [batch][frame]
        pil_frames = frames[0]  # batch=1, we only care about first element
        frames_np = np.stack([np.array(f) for f in pil_frames], axis=0)
    else:
        # Frames might be a tensor or numpy array already.
        if isinstance(frames, torch.Tensor):
            frames_np = frames.cpu().numpy()
        else:
            frames_np = np.array(frames)

        # Expected shape: (batch, num_frames, H, W, C)
        frames_np = frames_np[0]

    return frames_np  # shape: (T, H, W, C)


def save_frames_as_mp4(
    frames: np.ndarray,
    output_path: str,
    fps: int = 7,
):
    """
    Save frames (T, H, W, C) to an MP4 video using imageio.

    Why imageio:
    - Provides a simple API on top of ffmpeg.
    - Good enough for an MVP; later we can expose more control over encoding
      (bitrate, codec, etc.) if needed for production.
    """
    # Ensure output directory exists so the function is robust against missing
    # directories when called from various working directories.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert to uint8 because most video encoders expect this format.
    frames_uint8 = frames.astype(np.uint8)

    # imageio runs ffmpeg under the hood (via imageio-ffmpeg).
    # We start with a simple call; if quality or size is an issue,
    # we can refine this.
    imageio.mimsave(
        output_path,
        frames_uint8,
        fps=fps,
        quality=8,  # mid-range quality; tweak later if needed
    )


def generate_video_from_image(
    input_image_path: str,
    output_video_path: str,
    model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt",
    device: str = "cuda",
    enable_cpu_offload: bool = False,
    target_size: Tuple[int, int] = (576, 1024),
    num_frames: int = 14,
    fps: int = 7,
    motion_bucket_id: int = 80,
    noise_aug_strength: float = 0.02,
    seed: Optional[int] = 42,
):
    """
    High-level convenience function: image path -> video file.

    Why this wrapper:
    - Your future REST API or job worker should not need to know *how* the
      model runs. They only care about "give me a video for this image".
    - Encapsulating the steps here makes swapping out SVD or changing
      resolution/motion policies much easier later.
    """
    # Load model
    pipe = load_svd_pipeline(
        model_id=model_id,
        device=device,
        enable_cpu_offload=enable_cpu_offload,
    )

    # Preprocess the input
    image = load_and_preprocess_image(
        image_path=input_image_path,
        target_size=target_size,
    )

    # Generate frames
    frames = generate_video_frames(
        pipe=pipe,
        image=image,
        num_frames=num_frames,
        fps=fps,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        seed=seed,
    )

    # Save to disk
    save_frames_as_mp4(
        frames=frames,
        output_path=output_video_path,
        fps=fps,
    )
