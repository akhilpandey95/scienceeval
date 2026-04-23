# AGENTS.md

## Purpose

This repo prefers Python scripts that are strongly planned, helper-driven, and source-grounded.  
When editing or creating scripts here, use `scripts/evaluate_frontierscience.py` as the style reference.

## Python Script Style

### Structure

- Group imports by purpose with short section comments such as `# stdlib`, `# data`, or `# init logging`.
- Put constants near the top of the file.
- Separate constants into meaningful blocks such as directory constants, endpoint defaults, benchmark constants, and prompt templates.
- Keep the main execution path short. Push real work into helper functions.
- Prefer a final orchestration function such as `run_<task>()` plus a thin `if __name__ == "__main__":` entrypoint.

### Function Design

- Prefer many small helpers over a few large mixed-responsibility functions.
- Give helpers direct names such as `load_dataset`, `build_attempts`, `generate_sync`, or `summarize_results`.
- Pass plain dicts, lists, and namespaces unless a stronger abstraction is clearly needed.
- Prefer strong defaults and explicit constants rather than optional modes that weaken the script’s purpose.
- Keep scripts grounded to the primary source or paper they are implementing. Do not add convenience behavior unless it is clearly requested.

### Comments And Docstrings

- Use brief section comments like `# helper function to ...` ahead of nontrivial helpers.
- Keep inline comments sparse and functional.
- For nontrivial helpers, use compact docstrings with:
  - a one-line purpose
  - `Parameters`
  - `Returns`
- Do not write decorative or chatty comments.

### Logging And Output

- Prefer `logging` for progress and operational messages.
- Keep logs concise and factual.
- Save resumable state incrementally when a script is expected to run for a long time.
- Use stable ids for resumability and deterministic filenames for outputs.

### Planning Preferences

- Make strong choices early and encode them as constants.
- Avoid sprawling configuration surfaces when the script is meant to reproduce a fixed protocol.
- When a script is derived from a paper, benchmark, or spec, match that source first and expose only the controls needed to run it.
