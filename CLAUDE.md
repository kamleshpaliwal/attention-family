# Attention family: learning project

- Each attention variant lives in its own subfolder of this folder (side by side).
- First variant: `additive attention/` (Bahdanau-style, 12-step tutor course: step01_softmax.py, step02_..., each importing from earlier steps).
- Future variants get new sibling subfolders. Never mix files across variants.
- Run scripts from inside the relevant subfolder.
- Each variant folder has its own `.venv` (create with `python3 -m venv .venv`), so packages never conflict between variants. Run with `./.venv/bin/python stepXX_name.py`. Install packages only into that variant's `.venv` (numpy now; torch at step 10, matplotlib at step 11).
