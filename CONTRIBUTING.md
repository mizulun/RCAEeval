# Contributing to this repository

Thank you for helping improve MERIT. This guide covers contributions to the method, source code, tests, and public method documentation. Thesis and manuscript writing remain outside the scope of public contributions.

## Where to file issues and pull requests

**All issues and pull requests for this repository belong in [`mizulun/RCAEeval`](https://github.com/mizulun/RCAEeval), and every pull request must target its `merit` branch. Never open a pull request against the original author's `phamquiluan/RCAEval` repository. Do not target `main`.**

For a method proposal, protocol change, new metric, or experiment that could consume substantial compute, open an issue in `mizulun/RCAEeval` first and agree on the scope before implementation. For a reproducible bug, include the failing command, environment, and smallest available input or fixture. Do not upload private data, credentials, or large datasets to an issue.

## Issue and pull-request format


All contributors, including maintainers and contributors using coding agents, follow this standard. The structure is adapted from Apache DataFusion Comet's [issue forms](https://github.com/apache/datafusion-comet/tree/main/.github/ISSUE_TEMPLATE) and [PR template](https://github.com/apache/datafusion-comet/blob/main/.github/pull_request_template.md), with additional fields for reproducible RCA research. Use descriptive titles and enough detail for someone unfamiliar with the work to assess it. English is preferred for public contributions; keep the supplied section headings.

### Opening an issue

Choose the form that matches the work:

| Form | When to use it | What to include |
|---|---|---|
| [Bug report](.github/ISSUE_TEMPLATE/bug_report.yml) | Incorrect behavior or a reproducible failure | Actual behavior, reproduction steps, expected behavior, branch/commit, Python/lock file, operating system, and relevant errors |
| [Feature request](.github/ISSUE_TEMPLATE/feature_request.yml) | A feature, usability improvement, or documentation improvement | The problem, desired behavior, possible solution, and alternatives or examples |
| [Method or experiment proposal](.github/ISSUE_TEMPLATE/method_proposal.yml) | A method, protocol, metric, or substantial experiment | Motivation, choice and alternatives, assumptions, checked evidence and gaps, evaluation criteria, cost, risks, and scope |

Issue titles describe the problem or desired behavior directly, for example `MERIT fails when a trace column is missing`. An issue title does not need the PR title prefix. Complete required fields; if a reproducer or cost estimate is unavailable, explain the limitation. You can report a problem without volunteering to fix it.

GitHub's issue chooser uses these forms and disables the blank-issue option for ordinary contributors. Maintainers with write access can still open blank issues. If you use the CLI or API, include the same information and section headings in the issue body; the web form's required-field validation does not enforce CLI/API submissions.

### Pull-request titles

Use `type: description` or `type(scope): description`. Types are lowercase:

| Type | Use |
|---|---|
| `feat` | Add a feature |
| `fix` | Correct a defect |
| `docs` | Change documentation |
| `perf` | Improve performance |
| `refactor` | Restructure code while preserving behavior |
| `test` | Add or improve tests |
| `build` | Change dependencies or packaging |
| `ci` | Change continuous integration |
| `chore` | Other maintenance |
| `revert` | Revert a previous change |

The optional scope names the affected area, such as `merit`, `trace`, or `parser`. It starts with an ASCII letter or digit and may contain ASCII letters, digits, `.`, `_`, `/`, and `-`, without spaces. An optional `!` immediately before the colon marks a breaking change; explain that change in the PR body. Follow the colon with one space and a nonempty description starting with a non-whitespace character. Titles must fit on one line.

Examples:

```text
fix(merit): handle missing trace columns
feat(parser): support an additional trace field
docs: clarify the locked development environment
ci: check pull request titles
refactor(api)!: remove a deprecated argument
```

The [`Check PR Title` workflow](.github/workflows/pr_title_check.yml) checks titles on PRs targeting `merit`, including PRs from forks. It runs when a PR is opened, edited, reopened, or updated with commits, so correcting the title triggers a fresh check. It reads the title through an environment variable and does not check out or execute contribution code. The check validates title syntax; reviewers assess whether the title accurately describes the change.

GitHub may require a maintainer to approve workflow runs for fork contributions, and `pull_request` workflows do not run while a PR has merge conflicts. In those cases, resolve the approval or conflict before expecting the check to run. Workflow definitions are also part of a PR's changes, so reviewers must inspect changes to `.github/workflows/` rather than treating a passing check as approval of its own definition. See [GitHub's PR workflow behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#pull_request).

### Pull-request descriptions

Use the four main sections in [the PR template](.github/PULL_REQUEST_TEMPLATE.md):

1. **Related issue:** Link the issue. Use `Closes #123` only when the PR fully resolves it; use `Related to #123` for partial or follow-up work. Small documentation or maintenance changes may explain why no issue was needed. GitHub's automatic issue closing applies when the PR is merged into the default branch.
2. **Rationale:** Explain the problem and intended behavior, or refer to the linked issue without duplicating its full description.
3. **Changes:** Summarize the concrete modifications and compatibility implications.
4. **Testing:** Record the environment, commands, actual results, and CI checks. Explain when existing tests suffice or when a documentation-only change needs no new tests. Do not report a check as passed unless it was run.

Complete the target/checklist section. For research changes, also complete the expandable research-results section and follow [Reporting results consistently](#reporting-results-consistently). Keep evidence status and limitations explicit.

When creating a PR through the CLI or API, use the same description structure. For example, copy `.github/PULL_REQUEST_TEMPLATE.md` to a local Markdown file, fill it in, and pass that file using `gh pr create --body-file <path-to-filled-template.md>` together with the explicit repository, base, head, and title options below. The Markdown template supplies guidance; it does not itself make PR body fields mandatory.

## Branch and pull-request workflow

Fork `mizulun/RCAEeval` on GitHub, then clone your fork. Add the project repository as `upstream`, fetch its `merit` branch, and create a topic branch from that branch:

```bash
git clone https://github.com/<your-github-name>/RCAEeval.git
cd RCAEeval
git remote add upstream https://github.com/mizulun/RCAEeval.git
git fetch upstream
git switch -c feat/<issue-number>-short-description upstream/merit
```

Keep `origin` pointed at your fork and `upstream` pointed at `mizulun/RCAEeval`. Make focused commits, run relevant tests, and push your topic branch to your fork:

```bash
git push -u origin feat/<issue-number>-short-description
```

Open the pull request from your fork to **`mizulun/RCAEeval`**, with base branch **`merit`**. GitHub may preselect the original parent repository for a fork; explicitly select `mizulun/RCAEeval` as the base repository, then verify both the repository and `merit` base branch before submitting. You can force the intended destination with GitHub CLI:

```bash
gh pr create --repo mizulun/RCAEeval --base merit \
  --head <your-github-name>:feat/<issue-number>-short-description \
  --title "feat(merit): describe the resulting behavior" \
  --body-file <path-to-filled-template.md>
```

Link the related issue, explain the behavior change, list the commands and tests you ran, and attach or link the relevant manifest and generated summary for research results. A maintainer reviews and merges accepted changes into `merit`.

### Maintainer activation checklist

Repository files make the standard available to every contributor after they are merged and pushed. Repository settings must also be configured on GitHub:

1. Enable **Issues** and set `merit` as the repository's default branch. Setting the default branch does not change the contents of `main`.
2. Merge this guide, the Issue Forms and `config.yml`, the PR template, and the title-check workflow into `merit`. GitHub reads Issue and PR templates from the default branch.
3. Open **Issues → New issue** and confirm the three forms appear. Start a PR to `mizulun/RCAEeval:merit` and confirm the description template appears. Test an invalid title and then edit it to a valid title to confirm the check reruns.
4. Configure a ruleset or branch protection for `merit` that requires pull requests, at least one approval, and the two existing build-and-test status checks as named in the PR checks (matrix names may include the Python version), plus `check-pr-title`. Select `check-pr-title` after its first workflow run makes it available. A failed check blocks merging only when repository rules require it.

This checklist describes activation, not a claim that remote settings have already been changed. See GitHub's [Issue template configuration](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/configuring-issue-templates-for-your-repository), [PR template configuration](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/creating-a-pull-request-template-for-your-repository), and [linking a PR to an issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue).

## Development environment

The frozen MERIT research protocol uses **Python 3.10.18** and [`requirements_py310.lock`](requirements_py310.lock). The repository's general package CI also tests a Python 3.12 environment; that is a separate compatibility check and does not replace the locked MERIT environment.

On Ubuntu, install the system libraries used by the project. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) using its official instructions, then use it to download the exact Python patch release and create the protocol environment at `./py310`:

```bash
sudo apt-get update
sudo apt-get install -y build-essential libxml2 libxml2-dev zlib1g-dev \
  python3-tk graphviz libjpeg-dev

uv python install 3.10.18
PYTHON310="$(uv python find 3.10.18)"
"$PYTHON310" -m venv py310
./py310/bin/python -m pip install -r requirements_py310.lock
```

If your system does not provide Python 3.10.18, install that exact patch release with your preferred Python version manager. Do not update the lock file to make a local installation work; propose dependency changes separately and document their effect on reproducibility.

The lock fixes Python package versions; it does not make timing identical across machines. When reporting runtime or memory, record the operating system, CPU, memory, and numerical-library/thread settings required by the protocol.

Check the environment before testing:

```bash
./py310/bin/python --version
./py310/bin/python -m pip check
```

The general package installation instructions in [README.md](README.md) and [docs/SETUP.md](docs/SETUP.md) serve the upstream RCAEval package modes. For MERIT changes and protocol-aligned runs, use the Python 3.10.18 lock above.

## Editing and testing

- Before changing MERIT, read [`docs/figures/merit-flow.md`](docs/figures/merit-flow.md) and [`docs/method-theory.md`](docs/method-theory.md). Before changing an experiment or making a research claim, read [`docs/research-protocol.md`](docs/research-protocol.md) and [`experiment/protocol.json`](experiment/protocol.json). If these protocol sources appear to differ or leave a required field unclear, stop and ask in an issue; do not guess or edit the frozen contract.
- The current MERIT implementation is split across `RCAEval/e2e/merit.py`, `merit_config.py`, `merit_ablation.py`, `merit_evidence.py`, `merit_trace.py`, `merit_game.py`, and `merit_diagnostics.py`. Follow these module boundaries, keep changes focused, and add or update a focused test under `tests/` for behavior changes.
- For MERIT or result-generation code changes, run the relevant tests in the locked environment. A broad local check is:

  ```bash
  ./py310/bin/python -m pytest -q
  ```

- Dataset download tests are skipped unless `RCAEVAL_RUN_NETWORK_TESTS=1` is explicitly set. Leave it unset for ordinary unit-test runs.
- The `merit` branch triggers two existing GitHub Actions jobs: the package checks on Python 3.12 and a legacy RCD check on Python 3.8. Neither job installs `requirements_py310.lock`; the Python 3.8 RCD job is not the frozen research environment. Run the locked-environment tests locally and review both CI jobs on your pull request.
- Do not edit the frozen research protocol, experiment defaults, or baseline implementations as incidental cleanup. Discuss any such change in an issue first. Exploratory runs must be labeled and reported as exploratory; a dirty-worktree run is not a clean reproduction.
- Do not commit downloaded datasets, raw traces, generated reports, run directories, or other large outputs. Keep them in the ignored `data/`, `artifacts/`, or `output/` directories. Commit source, tests, maintained documentation, and small fixtures only.

## Reporting results consistently

Use [`docs/research-protocol.md`](docs/research-protocol.md) and [`experiment/protocol.json`](experiment/protocol.json) as the authority for frozen research comparisons. They must agree before a protocol-aligned run. Do not adjust defaults or select the headline configuration after seeing the results.

- Report service-level AC@1 as the primary metric. For the headline, show the equal-weight macro mean across Online Boutique, Sock Shop, and Train Ticket, with the pooled result beside it. Include the exact denominator (`n`) and per-system values. AC@3, AC@5, Avg@5, execution success rate, successful-cases-only accuracy (sensitivity analysis only), wall time, and peak memory are secondary measures.
- Keep failures in the denominator: a crash or timeout is a miss. Report execution success separately. Mark a pre-declared infeasible or modality-inapplicable method/system cell as `N/A`, not zero. Report a pooled 270-case value only when all three 90-case system cells are complete; otherwise state the included systems and exact `n`.
- For protocol-aligned runs, use RCAEval Zenodo dataset release v2 and verify the archive MD5 recorded in the protocol. For paired inference, use only the valid shared case set and report its exact `n`. Report effect sizes in percentage points, confidence intervals, raw p-values, Holm-adjusted p-values, and the comparison family. Do not describe descriptive or exploratory intervals as passing a frozen inferential gate.
- When reporting results, follow the established presentation: rates and execution success as percentages to one decimal place, runtime in seconds to two decimals, effect sizes and confidence bounds in percentage points, and p-values to three decimals (or `<.001`). Always include units and the exact sample size where applicable.
- Label the evidence status (for example, `INTERIM` or the status required by the protocol), dataset release, code/protocol version, seeds, denominator, operating system, CPU, memory, thread settings, and relevant manifest. Preserve per-case outcomes and errors so another contributor can reproduce the summary.
- Provide machine-readable summaries and manifests for experimental results. Do not hand-edit generated values. The maintainer handles any manuscript integration.

The formal frozen campaign is resource-intensive and requires a clean Git snapshot. Coordinate any full run with the maintainer before starting; do not launch it merely to validate a small code change.
