# Contributor agent instructions

This public repository welcomes focused improvements to the MERIT method, source code, tests, and method documentation. Contributions do not include thesis or manuscript writing.

- Read `CONTRIBUTING.md` before making changes. For method changes, consult `docs/method-theory.md`, `docs/figures/merit-flow.md`, `docs/research-protocol.md`, and `experiment/protocol.json` as applicable.
- Open all issues and pull requests at `mizulun/RCAEval`. Pull requests must target `merit`. Never target `phamquiluan/RCAEval` or `main`.
- Do not read, create, edit, quote, or commit thesis/manuscript content. Keep `paper/` and private thesis materials local and untracked. Do not add thesis outlines, writing plans, submission metadata, or paper drafts to this repository.
- Do not use ground-truth labels, fault names, or case-directory tokens in a method's ranking logic. Do not add dataset-, service-, or fault-specific ranking branches. Discuss protocol or default changes in an issue before implementation.
- Keep changes focused. Add or update tests for behavior changes, and report the exact commands and outcomes. Use the locked environment and result-reporting rules in `CONTRIBUTING.md` for MERIT work.
- Never commit downloaded datasets, private data, raw traces, credentials, or generated run artifacts. Keep generated results traceable to their source, protocol, and manifest.
- If a protocol source is inconsistent or underspecified, stop and raise an issue rather than guessing.

`AGENTS.md` is the shared project instruction source. Tool-specific instruction files should point here instead of copying these rules.
