#!/bin/bash
#
# Build DeepThinker Sandbox Docker Images
#
# This script builds all secure Docker sandbox images for tiered execution.
# Images: sandbox (base), sandbox-gpu, sandbox-browser, sandbox-node, sandbox-trusted
#

set -e

# Array of (dockerfile, imagename) pairs
IMAGES=(
    "Dockerfile.sandbox:deepthinker-sandbox:latest"
    "Dockerfile.sandbox-gpu:deepthinker-sandbox-gpu:latest"
    "Dockerfile.sandbox-browser:deepthinker-sandbox-browser:latest"
    "Dockerfile.sandbox-node:deepthinker-sandbox-node:latest"
    "Dockerfile.sandbox-trusted:deepthinker-sandbox-trusted:latest"
)

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         DeepThinker Sandbox Image Builder                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed"
    echo ""
    echo "Please install Docker first:"
    echo "  - Linux: sudo apt-get install docker.io"
    echo "  - macOS/Windows: Install Docker Desktop"
    echo ""
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "❌ Error: Docker daemon is not running"
    echo ""
    echo "Please start Docker:"
    echo "  - Linux: sudo systemctl start docker"
    echo "  - macOS/Windows: Start Docker Desktop"
    echo ""
    exit 1
fi

echo "✅ Docker is available"
echo ""
echo "Building all sandbox images..."
echo "This may take several minutes on first build..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

BUILD_FAILED=0
BUILT_IMAGES=()

# Build each image
for IMAGE_SPEC in "${IMAGES[@]}"; do
    IFS=':' read -r DOCKERFILE IMAGE_NAME <<< "$IMAGE_SPEC"
    
    # Check if Dockerfile exists
    if [ ! -f "$DOCKERFILE" ]; then
        echo "⚠️  Warning: $DOCKERFILE not found, skipping $IMAGE_NAME"
        continue
    fi
    
    echo "Building: $IMAGE_NAME"
    echo "  Using: $DOCKERFILE"
    
    if docker build -f "$DOCKERFILE" -t "$IMAGE_NAME" . > /dev/null 2>&1; then
        echo "  ✅ Success"
        BUILT_IMAGES+=("$IMAGE_NAME")
    else
        echo "  ❌ Failed"
        BUILD_FAILED=1
    fi
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $BUILD_FAILED -eq 0 ]; then
    echo "✅ All images built successfully!"
    echo ""
    echo "Built images:"
    for IMG in "${BUILT_IMAGES[@]}"; do
        IMAGE_SIZE=$(docker images "$IMG" --format "{{.Size}}" 2>/dev/null || echo "unknown")
        echo "  - $IMG ($IMAGE_SIZE)"
    done
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                    Build Complete!                         ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
else
    echo "⚠️  Some images failed to build"
    echo ""
    echo "Built images:"
    for IMG in "${BUILT_IMAGES[@]}"; do
        echo "  ✅ $IMG"
    done
    echo ""
    echo "Try rebuilding failed images manually:"
    echo "  docker build -f <dockerfile> -t <imagename> ."
    echo ""
fi

