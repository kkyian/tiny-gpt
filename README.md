# Tiny GPT from scratch

An educational character-level GPT in plain PyTorch, with hand-written causal attention and no pretrained model.

## Quick start

```bash
uv sync
uv run python prepare.py
uv run python check_tokenizer.py
uv run python check_model.py
uv run python train.py --overfit-check
uv run python train.py --steps 2000
uv run python generate.py "Alice was " --tokens 300 --temperature 0.8 --top-k 20
```

The device selector chooses Apple MPS, CUDA, then CPU. This is an educational text generator, not a ChatGPT-level assistant.
