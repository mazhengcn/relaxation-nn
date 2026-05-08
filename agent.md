# Agent Notes

- Default GPU for this repository is `GPU 1`.
- When launching training or evaluation jobs, prefer `CUDA_VISIBLE_DEVICES=1`.
- Never use `GPU 2` for this repository because that device is unreliable.
- If `GPU 1` is unavailable or the user asks for another device, use `GPU 3` instead.
- Prefer `uv run --extra train python ...` for training and evaluation launches instead of calling the system `python` directly.
