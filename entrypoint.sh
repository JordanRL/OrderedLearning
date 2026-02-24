#!/bin/sh
# entrypoint.sh - Runs when the RunPod container starts.
#
# Clones or pulls the repo using a GitHub personal access token (for private repos).
#
# Environment variables:
#   GITHUB_TOKEN   - GitHub PAT (required for private repos, optional for public)
#   REPO_URL       - Repository URL without protocol (default: github.com/your-user/OrderedLearning.git)
#   REPO_BRANCH    - Branch to clone/pull (default: master)
#   WORKSPACE_DIR  - Where to place the repo (default: /workspace/OrderedLearning)

set -e

TARGET_DIR="${WORKSPACE_DIR:-/workspace/OrderedLearning}"
REPO="${REPO_URL:-github.com/JordanRL/OrderedLearning.git}"
BRANCH="${REPO_BRANCH:-master}"

echo "--- Container Startup Script ---"

# Check for GitHub token (required for private repo)
if [ -z "$GITHUB_TOKEN" ]; then
    echo "WARNING: GITHUB_TOKEN is not set." >&2
    echo "The repository is private. Set GITHUB_TOKEN in your RunPod pod environment variables." >&2
    echo "Skipping code sync. If code already exists at $TARGET_DIR, it will be used as-is." >&2
else
    REPO_URL="https://${GITHUB_TOKEN}@${REPO}"

    if [ -d "$TARGET_DIR/.git" ]; then
        echo "Repository found in $TARGET_DIR. Pulling latest changes..."
        cd "$TARGET_DIR"

        # Update the remote URL in case the token changed
        git remote set-url origin "$REPO_URL"

        if ! git pull origin "$BRANCH"; then
            echo "Git pull failed." >&2
            exit 1
        fi
    else
        echo "Repository not found. Cloning into $TARGET_DIR..."
        if ! git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$TARGET_DIR" 2>&1; then
            echo "Git clone failed." >&2
            exit 1
        fi
        cd "$TARGET_DIR"
    fi

    echo "Code is up-to-date in $TARGET_DIR."
fi

# Ensure we're in the target directory if it exists
if [ -d "$TARGET_DIR" ]; then
    cd "$TARGET_DIR"
fi

echo "Executing command: $@"

# Replace the current process with the CMD
exec "$@"
