"""
Simple Image-to-Video Animation Pipeline using Wan2.2-I2V-A14B
With verbose logging for better traceability.
"""

import torch
from PIL import Image
import numpy as np
import subprocess
import os
import logging
import sys
from diffusers import StableDiffusionPipeline  # Replace with I2V-specific wrapper if available
from wan2 import WanI2VPipeline

# -------------------------------
# 1. Logging setup
# -------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True  # ensures existing loggers are reconfigured
)
logger = logging.getLogger(__name__)

# -------------------------------
# 2. Config / Inputs
# -------------------------------
IMAGE_PATH = "../../example_input.jpg"       # Your source image
PROMPT = "A realistic cinematic portrait of a person waving in slow motion, sunny day, natural lighting"
OUTPUT_VIDEO = "output.mp4"
VIDEO_FPS = 12
VIDEO_FRAMES = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_RESOLUTION = (512, 512)
# Path to your local model folder
LOCAL_MODEL_PATH = "../../models/Wan2.2-I2V-A14B"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clear previous handlers
logger.handlers = []

# Add stdout handler with flushing
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.propagate = False  # avoid duplicate logs

# -------------------------------
# 4. Load Wan2.2-I2V-A14B model
# -------------------------------
logger.info("Loading Wan2.2-I2V-A14B model...")
pipe = WanI2VPipeline.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype=torch.float16,
    local_files_only=True
)
pipe = pipe.to(DEVICE)
logger.info("Model loaded successfully")

# -------------------------------
# 5. Generate video frames
# -------------------------------
frames = []
logger.info(f"Generating {VIDEO_FRAMES} frames for the video...")
for frame_idx in range(VIDEO_FRAMES):
    logger.info(f"Generating frame {frame_idx + 1}/{VIDEO_FRAMES}")
    generator = torch.Generator(device=DEVICE).manual_seed(frame_idx)

    # Generate one frame from image + prompt
    output = pipe(prompt=PROMPT, init_image=image, strength=0.7, guidance_scale=7.5, generator=generator)
    frame = np.array(output.images[0])
    frames.append(frame)

logger.info("All frames generated successfully")

# -------------------------------
# 6. Save frames as video using FFmpeg
# -------------------------------
logger.info("Saving frames as video...")

# Save frames temporarily as PNG
tmp_frame_files = []
for idx, frame in enumerate(frames):
    tmp_file = f"frame_{idx:04d}.png"
    Image.fromarray(frame).save(tmp_file)
    tmp_frame_files.append(tmp_file)

ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-framerate", str(VIDEO_FPS),
    "-i", "frame_%04d.png",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    OUTPUT_VIDEO
]

logger.info("Running FFmpeg to encode video...")
subprocess.run(ffmpeg_cmd, check=True)
logger.info(f"Video saved to {OUTPUT_VIDEO}")

# Cleanup temporary frames
logger.info("Cleaning up temporary frame files...")
for tmp_file in tmp_frame_files:
    os.remove(tmp_file)

logger.info("Pipeline completed successfully!")
