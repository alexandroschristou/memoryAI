"""
Simple Image-to-Video Animation Pipeline using Wan2.2-I2V-A14B

This script:
1. Loads a single image.
2. Defines a text prompt for motion/scene description.
3. Uses Wan2.2-I2V-A14B to generate a short animated video.
4. Saves the video as MP4 using FFmpeg.

Requirements:
- Python 3.10+
- torch, torchvision
- diffusers
- PIL
- numpy
- ffmpeg installed and in PATH
"""

import torch
from PIL import Image
import numpy as np
import subprocess
from diffusers import StableDiffusionPipeline  # Replace with I2V-specific wrapper if available

# -------------------------------
# 1. Config / Inputs
# -------------------------------
IMAGE_PATH = "../../example_input.jpg"       # Your source image
PROMPT = "A realistic cinematic portrait of a person waving in slow motion, sunny day, natural lighting"
OUTPUT_VIDEO = "output.mp4"
VIDEO_FPS = 12                        # Frame rate
VIDEO_FRAMES = 30                      # Number of frames
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# 2. Load the input image
# -------------------------------
image = Image.open(IMAGE_PATH).convert("RGB")
# Resize to the model's expected resolution if needed
MODEL_RESOLUTION = (512, 512)
image = image.resize(MODEL_RESOLUTION)

# -------------------------------
# 3. Load Wan2.2-I2V-A14B model
# -------------------------------
# NOTE: replace 'Wan2.2-I2V-A14B' with actual model repo on HuggingFace
# This example assumes a standard diffusers-like API
pipe = StableDiffusionPipeline.from_pretrained(
    "Wan2.2-I2V-A14B",
    torch_dtype=torch.float16  # use half precision for speed/memory
)
pipe = pipe.to(DEVICE)

# -------------------------------
# 4. Generate video frames
# -------------------------------
frames = []
for frame_idx in range(VIDEO_FRAMES):
    # Optional: add slight variation per frame using seed or latent noise
    generator = torch.Generator(device=DEVICE).manual_seed(frame_idx)

    # Generate one frame from image + prompt
    output = pipe(prompt=PROMPT, init_image=image, strength=0.7, guidance_scale=7.5, generator=generator)
    frame = np.array(output.images[0])
    frames.append(frame)

# -------------------------------
# 5. Save frames as video using FFmpeg
# -------------------------------
# Save frames temporarily as PNG
tmp_frame_files = []
for idx, frame in enumerate(frames):
    tmp_file = f"frame_{idx:04d}.png"
    Image.fromarray(frame).save(tmp_file)
    tmp_frame_files.append(tmp_file)

# Build FFmpeg command
ffmpeg_cmd = [
    "ffmpeg",
    "-y",  # overwrite output if exists
    "-framerate", str(VIDEO_FPS),
    "-i", "frame_%04d.png",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    OUTPUT_VIDEO
]

subprocess.run(ffmpeg_cmd, check=True)

# Cleanup temporary frames (optional)
import os
for tmp_file in tmp_frame_files:
    os.remove(tmp_file)

print(f"Video saved to {OUTPUT_VIDEO}")
