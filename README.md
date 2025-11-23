# Informative RNA Embeddings

# Setup

## Setup Poetry

1. Install Poetry

- From latest instructions on [Poetry](https://python-poetry.org/).
- Or use your system package manager / pip and ignore the following steps.

2. Recommended - Use a project-local virtualenv (creates `.venv` in the repo directly instead of in poetry files.)

```bash
poetry config virtualenvs.in-project true
```

3. Install

```bash
poetry install  --no-root
```

- Run commands inside the environment:

```bash
poetry shell         # spawn a shell with .venv activated
# or
poetry run python my_script.py
```

## Setup Submodules

1. Initialize/update submodules (for clones)

```bash
git submodule update --init --recursive
# or when cloning
git clone --recurse-submodules <repo-url>
```

- After pulling changes that update submodules, run `git submodule update --init --recursive` and `poetry install` if dependencies changed.

2. Run

```bash
sh setup.sh
```

to add a symlink into the renata DQN folder

## Download Pretrained Model

Download from https://drive.google.com/drive/folders/10Yz-sdezhmazzVtrtdBGdzzqK1Z6f-Xv and place in `Renata-DQN/pretrained`
