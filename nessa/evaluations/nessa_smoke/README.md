# NESSA offline coding smoke evaluation

This package reuses `agentharness batch`: it does not introduce another agent loop.
Only Python's standard library and git are required. It makes no model or network
calls, downloads, installs, or changes to the harness repository.

## What is measured

Four small Python repair tasks: interval merging, stable unhashable deduplication,
key/value parsing, and retry semantics. Fixtures are deterministic. Only prompts
and buggy source go into the batch snapshots. Independent test bodies and reference
fixes stay in this package, outside those snapshots. A solve requires applying the
returned patch to a fresh original snapshot and completing every independent check;
`status=verified` from the harness is never enough. Missing predictions fail and
duplicate IDs are rejected rather than silently choosing a successful retry.

This is a development smoke suite, not a statistically useful capability benchmark.
The known reference solutions are grader self-tests, NOT model results. No real-model
baseline or candidate result has been measured here. Tests are withheld from model
snapshots, but are not a secure private holdout: both run under one OS account unless
an external isolation boundary is added. Never tune on these four tasks then call
improvements held-out generalization. For a real evaluation, author/freeze a separate
larger suite before changing the harness, restrict grader/reference access at the OS
boundary, and record exact model weights and every protocol control.

## Offline checks

    cd evaluations/nessa_smoke
    python -m unittest discover -s . -p 'test_*.py' -v
    python evaluate.py prepare /tmp/nessa-smoke-fixtures
    python evaluate.py hash

Use a new output directory; preparation refuses to overwrite an existing suite.
The tests prove that original bugs fail, reference patches pass, forged status does
not count, missing/duplicate predictions are handled, out-of-scope patches are
rejected, premature exit zero fails, and grader timeouts remain failures.

## Paired real-model runs (only when local model is already available)

From each pinned harness checkout, use its existing batch CLI:

    python -m agentharness batch --comp /tmp/nessa-smoke-fixtures --out /tmp/baseline.jsonl --runs /tmp/baseline-runs --workers 1 --model EXACT_LOCAL_MODEL --max-steps 40 --max-tokens 2048 --no-shell

Run the identical command from the candidate checkout, changing only output/run
paths. Both must use the same frozen model weights, tokenizer, quantization,
inference version, hardware, server settings, seed, temperature, concurrency,
step/token ceilings, fixtures, and number of attempts. Record their harness commits
and full CLI/config. Keep first-attempt outputs immutable; use fresh paths, because
batch normally resumes/retries. This v1 package supports one predeclared attempt per
task. Counterbalance baseline/candidate order for larger repeated experiments.

`--no-shell` removes `run_command`, but managed development tools can still launch
processes. It is not a command-execution boundary or an OS sandbox. For trustworthy holdout runs, keep the grader/reference outside the
agent's accessible filesystem. Only grade after the run has ended. Do not feed
independent-test failures or reference patches back into the evaluated attempt.

Copy `manifest.template.json` once per variant, fill unknown values from actual
provenance and set `suite_sha256` to the `hash` output. Never invent missing hashes.
The current CLI has per-call output-token and step limits. The harness also has a
soft 1,800-second elapsed-time guard checked before execute-phase model turns;
planning, model requests, and checks can overrun it. It does not enforce a strict
whole-run wall-time or token cap. Record the complete harness configuration and
use the literal `unbounded` for strict cap fields unless a separately documented
external limiter truly enforces one. That records a limitation; it does not provide
a matched-compute guarantee.

    python evaluate.py grade /tmp/baseline.jsonl baseline-manifest.json baseline-report.json --runs /tmp/baseline-runs
    python evaluate.py grade /tmp/candidate.jsonl candidate-manifest.json candidate-report.json --runs /tmp/candidate-runs
    python evaluate.py compare baseline-report.json candidate-report.json

Comparison rejects missing or mismatched controls. It reports gains and regressions
per task rather than hiding them in an aggregate. Reports refuse to overwrite an
existing file. Metadata is operator-supplied provenance, not cryptographic proof.

## Metrics and limits

- Verified solves: independent checks on clean patched originals, denominator all four tasks
- Seconds and steps: existing batch result fields; grade time is recorded separately
- Prompt/completion tokens: sum of usage reported for successful model responses,
  including planning/repair responses. Unknown/incomplete usage stays null, never
  estimated or converted to zero. Rejected requests/server-side retries may consume
  unreported compute; these are not billed-token totals
- Tool invalid rate: invalid proposed calls / all proposed calls, not tools executed.
  A valid call withheld because another call in the same turn was invalid is not
  itself invalid. Protocol errors are distinct from runtime/tool/check failures
- Failure classes: missing/empty/scope-invalid/rejected patches, failed tests,
  grader timeouts/errors; raw harness status is also retained
- Coverage counts accompany every aggregate; a missing task metric makes that total
  unavailable rather than allowing partial totals to look complete

`--runs` reads `TASK_ID/evidence/events.jsonl` sidecars. `start.telemetry_schema=1`
identifies full protocol-error instrumentation and honest optional token fields.
Older logs can supply some completion usage if present, but invalid rate and prompt
usage are unavailable. Interrupted traces are not counted as complete telemetry.

Generated code executes with a five-second subprocess timeout and bounded reported
logs. Patch paths are restricted to regular `solution.py`, and user Python site
configuration is disabled. This is NOT a security sandbox: untrusted code can still
access OS resources, create subprocesses, or tamper with evaluation. Use an externally
managed offline container/user boundary and resource limits for untrusted models.

## Repository tests

From the repository root:

    python -m unittest discover -s evaluations/nessa_smoke -p 'test_*.py' -v

The evaluation package does not change the batch JSONL contract or introduce an
alternate execution loop.
