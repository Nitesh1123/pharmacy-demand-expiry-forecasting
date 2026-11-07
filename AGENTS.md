# Project working instructions

- Preserve the configured candidate Git identity. Never append AI co-author
  trailers or generated-by attribution to commits, PRs, issues, or comments.
- Read `docs/PRD.md` and `data/README.md` before changing targets or schemas.
- Deadline: November 15, 2026; prioritize a complete, explainable MVP.
- Use the existing Ponytail, test-driven-development, debugging,
  incremental-implementation, git-workflow-and-versioning, and code-review skills
  when applicable. Use pharmacy-ml-validation for data/model/split work and
  jupyter-notebook for EDA notebooks; these are installed locally.
- Read the installed skill instructions when applying them. Skills do not change
  task authorization. Do not install plugin/skill source trees into this repository.
- Python 3.13, dependencies in `requirements.txt`. Run `python -m pytest -q`,
  `python src/data_generator.py`, and `python -m src.data_quality` for data changes.
- A passing quality report is not a model-performance claim. Unknown expiry
  outcomes remain unknown; final inventory and simulator parameters are not features.
- Keep the README explicit about implemented versus planned capabilities.
