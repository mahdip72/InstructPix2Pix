#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

# Run in a fresh Python 3.10-3.12 environment. Select an official wheel channel;
# CUDA builds require a compatible NVIDIA driver. CPU is for offline checks.
channel="${PYTORCH_CHANNEL:-cu126}"
case "$channel" in
    cpu|cu126|cu130) ;;
    *) echo "Unsupported PYTORCH_CHANNEL: $channel" >&2; exit 1 ;;
esac
python -m pip install torch==2.13.0 torchvision==0.28.0 \
    --index-url "https://download.pytorch.org/whl/$channel"
if [[ "$channel" != "cpu" ]]; then
    # The PyPI xFormers wheel targets CUDA 12.8. Match our selected CUDA channel.
    python -m pip install --force-reinstall --no-deps xformers==0.0.35 \
        --index-url "https://download.pytorch.org/whl/$channel"
fi
python -m pip install -r requirements.txt
