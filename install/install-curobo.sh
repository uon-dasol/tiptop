#!/bin/bash

set -eo pipefail

# Verify we're running from project root (check for tiptop-specific files)
if [ ! -f "tiptop/__init__.py" ]; then
    echo "ERROR: This script must be run from the tiptop project root directory"
    echo "Please run: pixi run install-curobo"
    exit 1
fi

REPO_URL="https://github.com/williamshen-nz/curobo.git"
INSTALL_DIR="curobo"
CUROBO_REVISION="b5fad1df2a3ac4d3e33e369918b7d62d0e59ebd1"

echo "==> Installing curobo..."

# Check if directory exist
should_clone=true
if [ -d "$INSTALL_DIR" ]; then
    echo "curobo already exists at $INSTALL_DIR"

    # Check git integrity
    cd "$INSTALL_DIR"
    if git fsck --full &> /dev/null; then
        echo "✓ curobo repository is healthy"
        should_clone=false
        cd ..
    else
        echo "✗ curobo repository is corrupted, removing..."
        cd ..
        rm -rf $INSTALL_DIR
    fi
fi

# Clone and install
if [ "$should_clone" = true ]; then
    echo "Cloning curobo"
    git clone "$REPO_URL" "$INSTALL_DIR"
fi

# Use the revision verified with our collision patch, including on a fresh clone.
cd curobo
echo "Fetching verified curobo revision..."
git fetch origin "$CUROBO_REVISION"
git checkout --detach "$CUROBO_REVISION"
echo "✓ curobo at verified revision: $(git rev-parse --short HEAD)"

# Keep the large-model self-collision fixes reproducible after a fresh clone.
collision_patch="../patches/curobo-self-collision.patch"
if git apply --reverse --check "$collision_patch" 2>/dev/null; then
    echo "✓ self-collision fixes already applied"
elif git apply --check "$collision_patch"; then
    git apply "$collision_patch"
    echo "✓ self-collision fixes applied"
else
    echo "ERROR: curobo changed; review patches/curobo-self-collision.patch before building"
    exit 1
fi

# pip install (builds CUDA kernels too)
echo "Installing curobo (might take 5-20 minutes)..."
pip install -e . --no-build-isolation --no-deps
cd ..

echo "✓ curobo installed successfully"
