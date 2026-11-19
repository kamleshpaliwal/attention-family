# Additive attention (Bahdanau-style)

A 12-step, hands-on walkthrough of the attention mechanism in an encoder-decoder translator. Plain NumPy for steps 1-9, PyTorch from step 10 (training).

Running example: translating "The cat sat today" into French, at the moment the decoder has written "Le" and must produce "chat". Example scores `[0.3, 3.0, 1.4, 0.3]` give weights `[0.05, 0.75, 0.15, 0.05]`.

## Setup

Run everything from inside this folder, with this folder's own virtual environment:

```
cd "additive attention"
python3 -m venv .venv
./.venv/bin/pip install numpy        # torch at step 10, matplotlib at step 11
./.venv/bin/python step01_softmax.py
```

## Steps

| # | File | What it shows | Status |
|---|------|---------------|--------|
| 1 | `step01_softmax.py` | scores to weights | done |
| 2 | `step02_embeddings.py` | words to vectors | done |
| 3 | `step03_encoder.py` | encoder states h1..h4 from a simple RNN | done |
| 4 | `step04_decoder.py` | decoder state s0, s1, s2 | done |
| 5 | `step05_scoring.py` | dot-product and Bahdanau scoring | done |
| 6 | `step06_context.py` | context vector (weighted sum) | done |
| 7 | prediction | stack [context, s], linear, softmax over vocabulary | to do |
| 8 | one full decoding step | assemble steps 1-7 | to do |
| 9 | decoding loop | whole sentence, word by word | to do |
| 10 | training | loss, backpropagation, PyTorch loop | to do |
| 11 | attention heatmap | check "chat" attends to "cat" | to do |
| 12 | recap | difference from Transformer self-attention | to do |

Each step imports from the earlier ones, so the pieces fit together by step 8.

## Revision

Open `revision_sheet.html` in a browser for a one-page summary: a flow diagram of how the steps connect, the shape of every array, and for each step the formula, the idea, example numbers and the worked arithmetic. Update it as new steps are added.
