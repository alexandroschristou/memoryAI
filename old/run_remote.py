import os  # For environment lookups or path handling if needed later.
from pathlib import Path  # Safer path handling than bare strings, especially on Windows.
import time  # To timestamp logs and measure how long the remote job took.

import paramiko  # SSH + SFTP library that lets us automate upload / command / download.


# ----------------------------
# Configuration – FILL THESE IN
# ----------------------------

# These values come directly from your RunPod "Connect" tab.
SSH_HOST = "ssh.runpod.io"  # <-- replace with your actual SSH hostname
SSH_PORT = 22                             # <-- replace with your SSH port (int)
SSH_USERNAME = "o0elfzc9ri9wp4-64411131"                        # RunPod pods typically use root by default.

# Path to your private SSH key that can access the pod.
# This is the same key you use when you SSH manually from your terminal.
SSH_KEY_PATH = r"C:\Users\alexc\.ssh\runpod_ed25519"  # <-- replace with your key path (Windows-style raw string)


# Remote paths inside the pod; these must match what generate_local.py expects.
REMOTE_PROJECT_DIR = "/workspace/memoryAI"
REMOTE_INPUT_PATH = f"{REMOTE_PROJECT_DIR}/input.jpg"
REMOTE_OUTPUT_PATH = f"{REMOTE_PROJECT_DIR}/output.mp4"
REMOTE_SCRIPT_PATH = f"{REMOTE_PROJECT_DIR}/generate_local.py"


def _connect_ssh() -> paramiko.SSHClient:
    """
    Establish an SSH connection to the RunPod instance.

    We let Paramiko handle the key type automatically by passing `key_filename`.
    This avoids the earlier bug where we forced an RSA key on an ed25519 file.
    """
    client = paramiko.SSHClient()

    # This tells Paramiko to automatically accept the host key the first time.
    # In a production setup you’d be more strict, but for this workflow it’s fine.
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # We do NOT manually parse the key as RSA/Ed25519; instead we give Paramiko
    # the path and let it figure out the right type, mirroring the normal ssh CLI.
    client.connect(
        hostname=SSH_HOST,
        port=SSH_PORT,
        username=SSH_USERNAME,
        key_filename=SSH_KEY_PATH,
        look_for_keys=False,
        allow_agent=False,
    )

    return client


def upload_image(ssh_client: paramiko.SSHClient, local_image: Path) -> None:
    if not local_image.is_file():
        raise FileNotFoundError(f"Local image not found: {local_image}")

    data = local_image.read_bytes()
    remote_cmd = f"mkdir -p {REMOTE_PROJECT_DIR} && cat > {REMOTE_INPUT_PATH}"

    print(f"Uploading {local_image} -> {REMOTE_INPUT_PATH}")
    # 🔑 ask for PTY
    stdin, stdout, stderr = ssh_client.exec_command(remote_cmd, get_pty=True)

    stdin.write(data)
    stdin.channel.shutdown_write()  # EOF

    out = stdout.read().decode("utf-8", errors="ignore")
    err = stderr.read().decode("utf-8", errors="ignore")
    exit_status = stdout.channel.recv_exit_status()

    # If RunPod prints the PTY error, treat it as failure even if exit_status is 0.
    if "Your SSH client doesn't support PTY" in out or "Your SSH client doesn't support PTY" in err:
        raise RuntimeError(
            "Remote upload failed because no PTY was allocated. "
            "Make sure exec_command is called with get_pty=True."
        )

    if exit_status != 0:
        raise RuntimeError(
            f"Remote upload failed with exit status {exit_status}. Stderr:\n{err}"
        )

    if out:
        print("Upload stdout:")
        print(out)



def run_remote_generation(ssh_client: paramiko.SSHClient) -> None:
    remote_command = (
        f"cd {REMOTE_PROJECT_DIR} && "
        f"source memoryAI/bin/activate && "
        f"python {REMOTE_SCRIPT_PATH}"
    )

    print(f"Running remote command:\n{remote_command}\n")

    start = time.time()
    # 🔑 ask for PTY
    stdin, stdout, stderr = ssh_client.exec_command(remote_command, get_pty=True)
    stdin.close()

    out = stdout.read().decode("utf-8", errors="ignore")
    err = stderr.read().decode("utf-8", errors="ignore")
    end = time.time()

    print(f"Remote generation finished in {end - start:.1f} seconds")

    if out:
        print("---- remote stdout ----")
        print(out)
    if err:
        print("---- remote stderr ----")
        print(err)

    if "Your SSH client doesn't support PTY" in out or "Your SSH client doesn't support PTY" in err:
        raise RuntimeError(
            "Remote generation failed because no PTY was allocated. "
            "Make sure exec_command is called with get_pty=True."
        )

    exit_status = stdout.channel.recv_exit_status()
    if exit_status != 0:
        raise RuntimeError(f"Remote generation failed with exit status {exit_status}")


def download_video(ssh_client: paramiko.SSHClient, local_output: Path) -> None:
    print(f"Downloading {REMOTE_OUTPUT_PATH} -> {local_output}")

    remote_cmd = f"cat {REMOTE_OUTPUT_PATH}"
    # 🔑 ask for PTY
    stdin, stdout, stderr = ssh_client.exec_command(remote_cmd, get_pty=True)
    stdin.close()

    data = stdout.read()
    err = stderr.read().decode("utf-8", errors="ignore")
    exit_status = stdout.channel.recv_exit_status()

    if "Your SSH client doesn't support PTY" in err.decode("utf-8", errors="ignore") if isinstance(err, bytes) else err:
        raise RuntimeError(
            "Remote download failed because no PTY was allocated. "
            "Make sure exec_command is called with get_pty=True."
        )

    if exit_status != 0:
        raise RuntimeError(
            f"Remote download failed with exit status {exit_status}. Stderr:\n{err}"
        )

    local_output.parent.mkdir(parents=True, exist_ok=True)
    local_output.write_bytes(data)


def main() -> None:
    """
    Full workflow:
    - Connect to the pod over SSH,
    - Upload a local image to the pod,
    - Run CogVideoX generation entirely on the pod,
    - Download the resulting MP4 back to the local machine.
    """
    # Paths on your Windows machine. Adjust them to your liking.
    local_image = Path(r"/example_input.jpg")
    local_output = Path(r"C:\Users\alexc\Desktop\memoryAI\output.mp4")

    print("Connecting to RunPod via SSH...")
    ssh_client = _connect_ssh()
    print("SSH connection established.")

    try:
        upload_image(ssh_client, local_image)
        run_remote_generation(ssh_client)
        download_video(ssh_client, local_output)
    finally:
        ssh_client.close()
        print("SSH connection closed.")

    print(f"All done. Generated video saved to: {local_output}")


if __name__ == "__main__":
    main()