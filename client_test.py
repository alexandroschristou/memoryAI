import requests  # We use requests because it's simple and widely understood.
from pathlib import Path  # For robust path handling on your local machine.


def generate_remote_video(server_url: str, image_path: str, output_path: str) -> None:
    """
    Helper to call the remote RunPod FastAPI endpoint and store the resulting MP4.

    We add explicit error logging so that, when something goes wrong server-side,
    we can see the actual error message returned by FastAPI rather than only the
    HTTP status code.
    """
    with open(image_path, "rb") as f:
        # We send the file under the field name "file" because that is what
        # the FastAPI endpoint is expecting for the UploadFile parameter.
        files = {"file": (image_path, f, "image/jpeg")}

        # Form fields; keeping them as strings keeps the form encoding simple.
        data = {
            "prompt": (
                "A realistic video of the person in this photo with subtle natural movement. "
                "No camera shake, no scene changes, stable composition."
            ),
            "num_frames": "16",       # Start SMALL to avoid OOM while debugging
            "steps": "30",            # Also keep steps modest initially
            "guidance_scale": "6",
            "fps": "8",
        }

        response = requests.post(
            f"{server_url}/generate",
            files=files,
            data=data,
            timeout=900,
        )

    if response.status_code != 200:
        # Print server-side error details so we can debug without guessing.
        print("Server returned error:")
        print(response.text)
        response.raise_for_status()

    with open(output_path, "wb") as out_f:
        out_f.write(response.content)

    print(f"Saved generated video to: {output_path}")


if __name__ == "__main__":
    # Replace this with your actual RunPod endpoint URL.
    SERVER_URL = "https://o0elfzc9ri9wp4-8000.proxy.runpod.net"
    IMAGE = "example_input.jpg"
    OUTPUT = "cogvideo_remote_output.mp4"

    generate_remote_video(SERVER_URL, IMAGE, OUTPUT)
