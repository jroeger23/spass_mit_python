#!/bin/sh

set -aeu

BASE_DIR="$(dirname $(readlink -f $0))"
VENV_PATH="$BASE_DIR/.venv"
REQ_PATH="$BASE_DIR/requirements.txt"

echo "Creating venv at $VENV_PATH"
python3 -m venv "$VENV_PATH"

echo "Installing requirements inside venv..."
source "$VENV_PATH/bin/activate"
pip install -r "$REQ_PATH"
