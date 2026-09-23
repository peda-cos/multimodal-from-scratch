#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

export PYTHONDONTWRITEBYTECODE=1

if [ ! -x ".venv/bin/python" ]; then
    uv venv --python 3.14 .venv
fi

uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python -m unittest code/test_main.py -v
.venv/bin/python -m mypy code
.venv/bin/python code/main.py
