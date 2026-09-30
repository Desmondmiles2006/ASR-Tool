"""Publish the repository to a Hugging Face Space (run by GitHub Actions).

Creates the Space on first run, then mirrors the repo into it. Hugging Face
reads the Space settings from YAML at the top of README.md, so that header is
added here instead of cluttering the README on GitHub.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

SPACE_NAME = "ASR-Tool"

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = f"{api.whoami()['name']}/{SPACE_NAME}"
api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

with tempfile.TemporaryDirectory() as tmp:
    stage = Path(tmp)
    # Only files tracked by git, so nothing local or secret slips in.
    for name in subprocess.check_output(["git", "ls-files"], text=True).splitlines():
        if name.startswith((".github/", "tests/")) or name == "README.md":
            continue
        (stage / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, stage / name)

    header = Path(".github/space_header.md").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    (stage / "README.md").write_text(header + readme, encoding="utf-8")

    api.upload_folder(
        folder_path=str(stage),
        repo_id=repo_id,
        repo_type="space",
        commit_message=f"Deploy {os.environ.get('GITHUB_SHA', 'local')[:7]} from GitHub",
        delete_patterns="*",  # remove files that were deleted from the repo
    )

print(f"Deployed: https://huggingface.co/spaces/{repo_id}")
