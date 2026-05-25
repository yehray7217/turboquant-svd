#!/usr/bin/env bash
set -euo pipefail
repo_root="${1:-.}"
cd "$repo_root"
mkdir -p utils modules
cp "$(dirname "$0")/utils/svdllama_loader.py" utils/svdllama_loader.py
cp "$(dirname "$0")/modules/svd_hf_registry.py" modules/svd_hf_registry.py
# Try source patch for script call sites; if it was already applied, keep going.
if patch --dry-run -p1 < "$(dirname "$0")/patches/turboquant_svd_svdllama_loader_v2.patch" >/tmp/svdllama_patch_dryrun.log 2>&1; then
  patch -p1 < "$(dirname "$0")/patches/turboquant_svd_svdllama_loader_v2.patch"
else
  echo "Patch call-site diff was not applied, probably already applied or source drifted."
  echo "Copied compatibility files: utils/svdllama_loader.py and modules/svd_hf_registry.py"
fi
