from huggingface_hub import whoami

token = "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"  # paste the same token here (between quotes)
print(whoami(token=token))