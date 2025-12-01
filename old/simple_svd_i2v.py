import argparse  # We keep argument parsing so it's easy to tweak settings from the CLI.
from pathlib import Path  # Path objects make file operations less error-prone and more readable.

import torch  # Needed for dtype selection and RNG.
from diffusers import StableVideoDiffusionPipeline  # This is the SVD image-to-video pipeline.
from diffusers.utils import load_image, export_to_video  # Helpers to load images and save a list of frames as a video.


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for this simple SVD test.

    Even though this is just a 'playground' script, exposing a few key parameters
    makes it much easier to iterate on prompt, frames, and output location
    without editing the file each time.
    """
    parser = argparse.ArgumentParser(description="Test Stable Video Diffusion (image-to-video) on low VRAM.")

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image (JPEG/PNG). This image will be animated by the model.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="svd_output.mp4",
        help="Path to the output video file.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=25,
        help=(
            "Number of frames to generate. SVD docs use 25 frames as a common example. "
            "More frames => longer video but higher VRAM/compute."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=7,
        help=(
            "Frames per second of the output video. "
            "Lower FPS means the clip feels slower/longer for the same frame count."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help=(
            "Number of denoising steps. 20–30 is a good range for first tests: "
            "lower values are faster but degrade quality."
        ),
    )
    parser.add_argument(
        "--decode-chunk-size",
        type=int,
        default=2,
        help=(
            "How many frames to decode at once in the VAE. "
            "Smaller values use less VRAM but are slower. "
            "Hugging Face's docs show that small values help SVD fit into <8GB VRAM."
        ),
    )
    parser.add_argument(
        "--motion-bucket-id",
        type=int,
        default=100,
        help=(
            "Controls how strong the motion is. Higher IDs = more motion. "
            "We start with 180 as suggested in the docs for noticeable but not insane motion."
        ),
    )
    parser.add_argument(
        "--noise-aug-strength",
        type=float,
        default=0.05,
        help=(
            "How much noise to add to the conditioning image. "
            "Higher values add more motion but also deviate more from the original image."
        ),
    )
    return parser.parse_args()


def choose_dtype() -> torch.dtype:
    """
    Select a floating-point precision compatible with SVD and low VRAM.

    On GPUs, float16 is essential to keep memory usage manageable.
    On CPU, we stick with float32 to avoid numerical issues.
    """
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_svd_pipeline() -> StableVideoDiffusionPipeline:
    """
    Load the Stable Video Diffusion img2vid-xt pipeline with low-VRAM settings in mind.

    Key design choices:
    - We use float16 on GPU to reduce VRAM.
    - We call `enable_model_cpu_offload()` so diffusers automatically moves
      components between CPU and GPU to fit into 8GB.
    - We don't try to manually micro-manage device maps; we defer to the built-in,
      well-tested optimization path described in the SVD docs.
    """
    dtype = choose_dtype()

    # This is the official SVD img2vid-xt checkpoint for image-to-video in diffusers.:contentReference[oaicite:1]{index=1}
    model_id = "stabilityai/stable-video-diffusion-img2vid-xt"

    # variant="fp16" is recommended in the docs when using float16.:contentReference[oaicite:2]{index=2}
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16",
    )

    # This is the critical call for low VRAM setups: it offloads components to CPU
    # when they're not in use, which dramatically reduces peak GPU memory usage.:contentReference[oaicite:3]{index=3}
    pipe.enable_model_cpu_offload()

    # Optional: this can further reduce memory spikes in the UNet by chunking feed-forward layers.
    # It's described in diffusers' memory optimization docs for video.:contentReference[oaicite:4]{index=4}
    if hasattr(pipe.unet, "enable_forward_chunking"):
        pipe.unet.enable_forward_chunking()

    return pipe


def generate_video_with_svd(
    pipe: StableVideoDiffusionPipeline,
    image_path: Path,
    output_path: Path,
    num_frames: int,
    fps: int,
    num_inference_steps: int,
    decode_chunk_size: int,
    motion_bucket_id: int,
    noise_aug_strength: float,
) -> None:
    """
    Run SVD on a single image to produce a video and write it to disk.

    We keep this logic separate so that later you can:
    - call it from a web API,
    - wrap it in an experiment loop,
    - or swap out SVD for CogVideoX while keeping the 'image in → frames out' contract.
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Load and resize the image to the resolution recommended in the SVD docs (1024x576).:contentReference[oaicite:5]{index=5}
    image = load_image(str(image_path))
    image = image.resize((576, 324))


    # Tie the RNG to the appropriate device. This lets you set seeds for reproducibility later.
    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device)
    # For reproducibility, uncomment:
    # generator.manual_seed(42)

    # Call the pipeline. The SVD docs show using motion_bucket_id and noise_aug_strength
    # to control motion intensity. decode_chunk_size keeps VRAM under control.:contentReference[oaicite:6]{index=6}
    result = pipe(
        image=image,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        decode_chunk_size=decode_chunk_size,
        generator=generator,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
    )

    frames = result.frames[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output_path), fps=fps)


def main() -> None:
    """
    Entry point: load pipeline once, run one image-to-video generation.

    This mirrors the structure you'd want for a backend service:
    - one-time model load,
    - per-request generation function.
    """
    args = parse_args()

    image_path = Path(args.image).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    print("Loading Stable Video Diffusion pipeline (first time can be slow)...")
    pipe = load_svd_pipeline()
    print("Pipeline loaded.")

    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB VRAM)")
    else:
        print("No CUDA device detected; running on CPU only (will be slow).")

    print(f"Generating video from: {image_path}")
    print(f"Saving result to: {output_path}")

    generate_video_with_svd(
        pipe=pipe,
        image_path=image_path,
        output_path=output_path,
        num_frames=args.num_frames,
        fps=args.fps,
        num_inference_steps=args.steps,
        decode_chunk_size=args.decode_chunk_size,
        motion_bucket_id=args.motion_bucket_id,
        noise_aug_strength=args.noise_aug_strength,
    )

    print("Done. Check your output video.")


if __name__ == "__main__":
    main()
