#!/bin/bash

set -eo pipefail

# Verify we're running from project root (check for tiptop-specific files)
if [ ! -f "tiptop/__init__.py" ]; then
    echo "ERROR: This script must be run from the tiptop project root directory"
    echo "Please run: pixi run install-orbbec"
    exit 1
fi

echo "==> Installing Orbbec Python SDK (Gemini 336L)..."

# --no-deps: the wheel lists example-only dependencies (av, pygame, pynput) that TiPToP does not use
python -m pip install --no-deps "pyorbbecsdk2==2.1.1"

echo "✓ pyorbbecsdk2 installed successfully"
echo "If the camera is not detected, set up the udev rule: python -m pyorbbecsdk.shared.setup_env"
