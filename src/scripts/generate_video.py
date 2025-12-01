# src/scripts/generate_video.py

import argparse
import os

from src.pipeline.image_to_video_svd import generate_video_from_image


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the generate_video script.

    Why we use argparse:
    - It gives you a clear, self-documenting interface for experimentation.
    - Later, when you wrap this in a job queue or API, you can still reuse the
      underlying functions without being tied to CLI details.
    """
    parser = argparse.ArgumentParser(
        description="Generate a short video from a single image using Stable Video Diffusion."
    )

    parser.add_argument(
        "--input_image",
        type=str,
        required=True,
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        required=True,
        help="Path where the output video (mp4) will be saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device to use, e.g. 'cuda' or 'cpu'. On RunPod this will almost always be 'cuda'.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=14,
        help="Number of frames in the generated video.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=7,
        help="Frames per second for the generated video.",
    )
    parser.add_argument(
        "--motion_bucket_id",
        type=int,
        default=80,
        help="Controls motion magnitude. Lower = subtle, higher = more dramatic.",
    )
    parser.add_argument(
        "--noise_aug_strength",
        type=float,
        default=0.02,
        help="Amount of noise augmentation during generation. Small values preserve identity better.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible results.",
    )
    parser.add_argument(
        "--enable_cpu_offload",
        action="store_true",
        help="Enable model CPU offload to save VRAM at the cost of speed.",
    )

    return parser.parse_args()


def main():
    """
    CLI entrypoint for generating a video from an image.

    Why we separate main():
    - Makes the script importable and testable.
    - Keeps side effects (like actually running the pipeline) nicely
      contained for easier debugging and later unit testing.
    """
    args = parse_args()

    # Basic sanity checks on paths to fail fast with clear messages.
    if not os.path.isfile(args.input_image):
        raise FileNotFoundError(f"Input image not found: {args.input_image}")

    output_dir = os.path.dirname(args.output_video)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    generate_video_from_image(
        input_image_path=args.input_image,
        output_video_path=args.output_video,
        device=args.device,
        enable_cpu_offload=args.enable_cpu_offload,
        num_frames=args.num_frames,
        fps=args.fps,
        motion_bucket_id=args.motion_bucket_id,
        noise_aug_strength=args.noise_aug_strength,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
