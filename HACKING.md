# Hacking on whodat

## Required toolchain

- [mise](https://mise.jdx.dev/installing-mise.html), [activated in your shell](https://mise.jdx.dev/getting-started.html#activate-mise). It installs the pinned `uv`, `ffmpeg`, Node and the diagram renderers
- a [Hugging Face account](https://huggingface.co/join)
- an NVIDIA GPU is optional: whodat uses CUDA when present, the CPU otherwise

## Setup

```bash
mise install
uv run hf auth login   # once per machine
mise run setup         # safe to rerun
```

`setup` stops at the first missing piece and says how to fix it,
including the pyannote model terms you must accept once.

If `mise install` fails with `401 Unauthorized`,
the GitHub token mise borrows from `gh` is stale:

```bash
gh auth login
```

## Tasks

```bash
mise tasks                  # list everything
mise run fmt                # fix lint and format
mise run ci                 # lint and unit tests
mise run test:integration   # loads models and touches files
```

Inside the repo, mise puts `whd` on your PATH
and sets `WHODAT_CORPUS=corpora/oneyplays`,
so `whd ...` works without `-c`.

## Tests

- `tests/unit`: pure logic. Fast, no models, no network.
- `tests/integration`: real files, packaged assets and models.
- `tests/e2e`: Hypothesis types random key sequences into the questionnaires
  on a synthetic corpus, predicts each screen and checks the files after every key.
  When it fails, it prints the shortest sequence of actions that breaks.

```bash
mise run test:e2e                          # about 20 sessions per questionnaire
WHODAT_E2E_EXAMPLES=200 mise run test:e2e  # longer hunt
```

Write tests through public interfaces with real objects;
pass collaborators in rather than patching them.

## Where files go

whodat follows the [XDG Base Directory spec](https://specifications.freedesktop.org/basedir-spec/latest/):

| Directory | Holds |
|---|---|
| `$XDG_CACHE_HOME/whodat`, default `~/.cache/whodat` | Audio, models, diarization, embeddings, voiceprints, predictions. Safe to delete; it all regenerates. |
| `$XDG_STATE_HOME/whodat`, default `~/.local/state/whodat` | Pending review questions, enrollment proposals, answered Voices and answers files. |
| `runs/` | Evaluation runs and timelines. Gitignored. |

## WSL

Questionnaires open YouTube links in the default browser.
To use the Windows one, add this to your shell profile:

```bash
export BROWSER=explorer.exe
```

## Conventions

- Use the [glossary](README.md#glossary) terms in code and docs.
- Record hard-to-reverse decisions as an ADR in [docs/adr](docs/adr).
- No issue tracker: issue docs live in `issues/`, which is gitignored.
- After editing [docs/pipeline.mmd](docs/pipeline.mmd), run `mise run docs:diagram`.
