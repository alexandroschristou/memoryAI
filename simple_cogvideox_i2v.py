import argparse  # Argument parsing so we can tweak behavior from the command line without editing code each time.
from pathlib import Path  # Path objects make file handling more robust and cross-platform.

import torch  # Used to configure data types and check GPU capabilities.
from diffusers import CogVideoXImageToVideoPipeline  # The image-to-video pipeline we want to experiment with.
from diffusers.utils import export_to_video, load_image  # Helpers for loading images and exporting frames as a video.
from huggingface_hub import login  # So we can programmatically ensure we're authenticated if needed.


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for this quick test script.

    Even for a 'simple experiment', adding arguments pays off immediately:
    you can run multiple tests with different images, prompts, or settings
    from the command line, without constantly editing the file.
    """
    parser = argparse.ArgumentParser(description="Local test for CogVideoX image-to-video with CPU/GPU offload.")

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image (JPEG/PNG). This is the base frame the model will animate.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help=(
            "Path where the generated video will be saved. "
            "Making this configurable is useful when you batch different experiments."
        ),
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "A realistic video of the people in this photo with subtle natural movement. "
            "No camera shake, no scene changes, stable composition."
        ),
        help=(
            "Text prompt that guides the style/motion. "
            "Even though the model is conditioned on the image, the prompt still influences motion and details."
        ),
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=48,
        help=(
            "Number of frames to generate. 48 is a reasonable starting point for short clips. "
            "More frames mean a longer video but also significantly higher memory and compute cost."
        ),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help=(
            "Number of denoising steps. 40 is a compromise between quality and speed. "
            "You can try 30 for faster tests or 50+ when you care about quality."
        ),
    )

    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.0,
        help=(
            "Classifier-free guidance scale. Higher values force the model to follow the prompt more strictly, "
            "which can help style consistency but sometimes introduces artifacts if pushed too far."
        ),
    )

    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help=(
            "Optional output height. If left as None, the pipeline uses its default. "
            "Explicitly controlling resolution is useful later for product constraints, "
            "but for early experiments we keep it optional to avoid surprises with VRAM."
        ),
    )

    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help=(
            "Optional output width. If left None, defaults are used. "
            "For now we rely on defaults to reduce the chance of OOM on an 8GB GPU."
        ),
    )

    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face token. If provided, the script will call login() explicitly. "
            "Otherwise it assumes you've already run `huggingface-cli login`."
        ),
    )

    return parser.parse_args()


def choose_dtype() -> torch.dtype:
    """
    Decide which floating point precision to use.

    On GPUs, lower precision is essential for fitting big models.
    CogVideoX is designed around bfloat16/float16 on GPU.
    On CPU we stick to float32 to avoid unnecessary numerical headaches.
    """
    if torch.cuda.is_available():
        # If the GPU supports bfloat16, we prefer it because the model is tuned for it
        # and it tends to be more numerically stable than float16 in many modern setups.
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        # Otherwise, float16 is still dramatically more memory-efficient than float32.
        return torch.float16

    # If there's no CUDA device, fall back to float32 on CPU. Performance will be slow,
    # but here correctness and simplicity matter more than speed for initial experiments.
    return torch.float32


def load_pipeline_with_offload(
    hf_token: str | None = None,
) -> CogVideoXImageToVideoPipeline:
    """
    Load the CogVideoX image-to-video pipeline with CPU/GPU offloading enabled.

    We explicitly use device_map='auto' so 🤗 Accelerate spreads weights across available devices.
    This is essential on an 8GB GPU because the full model cannot reside entirely in VRAM.

    Keeping pipeline loading in a dedicated function makes it trivial to:
    - switch to a different checkpoint later (e.g. a fine-tuned one),
    - replace the local model with a thin HTTP client that talks to a cloud inference server,
    without having to touch the rest of the generation logic.
    """
    if hf_token:
        # Logging in programmatically is useful when running in environments where
        # the standard `huggingface-cli login` is not practical (e.g., CI or containers).
        login(token=hf_token)

    dtype = choose_dtype()

    model_id = "THUDM/CogVideoX-5b-I2V"

    # device_map='auto' is the critical flag here: it instructs Accelerate to shard the model
    # across CPU and GPU automatically. On an 8GB GPU, this typically means:
    # - some layers on GPU for speed,
    # - the rest on CPU to avoid OOM.
    #
    # offload_folder provides a disk location for offloaded weights if needed.
    # This makes it possible to run very large models at the cost of speed and disk I/O.
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="cuda",  # crucial for 8GB setups: we don't manually call .to("cuda")
        offload_folder="offload",  # folder used for offloading weights to disk when RAM/VRAM is tight.
    )

    # Setting a "channels_last" memory format can help convolution-heavy models perform better on GPU.
    # This line is safe even if most of the model is on CPU, and it's a cheap tweak.
    pipe.transformer.to(memory_format=torch.channels_last)

    # We deliberately avoid more aggressive optimizations (quantization, compile, etc.)
    # in this first test to keep behavior easier to debug. Those can be layered on later.
    return pipe


def generate_video_from_image(
    pipe: CogVideoXImageToVideoPipeline,
    image_path: Path,
    output_path: Path,
    prompt: str,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    height: int | None = None,
    width: int | None = None,
) -> None:
    """
    Core 'image-to-video' function: loads the image, runs the pipeline, and writes an MP4.

    Keeping this logic separate from CLI and pipeline-loading code makes it easy to:
    - reuse in a FastAPI/Flask/Django endpoint,
    - plug it into a notebook for experiments,
    - or later swap the underlying model implementation (local vs remote) without changing call sites.
    """
    if not image_path.is_file():
        # Early, explicit failure is nicer than a cryptic PIL or OS error deeper down.
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Using diffusers' load_image ensures consistent preprocessing and supports URLs
    # if you decide to pass remote images later.
    image = load_image(str(image_path))

    # We use a generator tied to the device where the main computation happens.
    # This lets you set seeds for reproducibility when you care about it.
    # For now we keep it random to see a variety of behaviors.
    if torch.cuda.is_available():
        generator = torch.Generator(device="cuda")
    else:
        generator = torch.Generator(device="cpu")

    # Example: Uncomment for reproducible generations
    # generator.manual_seed(42)

    # The pipeline returns an object with .frames: List[List[PIL.Image]].
    # We ask for a single video, so we take frames[0].
    # We pass height/width only if explicitly provided; otherwise defaults are used
    # to reduce the risk of hitting VRAM limits on your 8GB GPU.
    video_output = pipe(
        image=image,
        prompt=prompt,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
        use_dynamic_cfg=True,  # dynamic CFG often improves stability vs static CFG.
    )

    frames = video_output.frames[0]

    # We choose a modest FPS. 16 fps is a good balance: smooth enough to feel like real motion,
    # but not so high that we explode the number of frames for a given clip length.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output_path), fps=16)


def main() -> None:
    """
    Script entry point: parse arguments, load pipeline, run a single generation.

    The structure mirrors what you'd want in a future microservice:
    - one-time model load,
    - per-request generation,
    - clean separation between configuration, model, and I/O.
    """
    args = parse_args()

    image_path = Path(args.image).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    print(f"Loading pipeline (this may take a while on first run)...")
    pipe = load_pipeline_with_offload(hf_token=args.hf_token)
    print("Pipeline loaded.")

    if torch.cuda.is_available():
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Detected CUDA device: {torch.cuda.get_device_name(0)} ({total_vram_gb:.1f} GB VRAM)")
    else:
        print("No CUDA device detected; running entirely on CPU. This will be very slow.")

    print(f"Generating video from: {image_path}")
    print(f"Output will be saved to: {output_path}")

    generate_video_from_image(
        pipe=pipe,
        image_path=image_path,
        output_path=output_path,
        prompt=args.prompt,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
    )

    print("Done. Check your output video.")


if __name__ == "__main__":
    main()
