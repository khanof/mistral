import os
from huggingface_hub import snapshot_download

model_path = "/app/models/mistralai/Mistral-7B-Instruct-v0.1-AWQ"

# Ensure model is downloaded only if not already present
if not os.path.exists(model_path) or len(os.listdir(model_path)) == 0:
    print("Downloading Mistral-7B AWQ model...")
    snapshot_download(repo_id="TheBloke/Mistral-7B-Instruct-v0.1-AWQ", local_dir=model_path, token=None)
else:
    print("Model already exists. Skipping download.")
