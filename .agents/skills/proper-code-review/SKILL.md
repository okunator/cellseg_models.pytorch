---
name: proper-code-review
description: Review a diff, pull request, or recent change in this PyTorch segmentation library for evidenced correctness, compatibility, simplicity, and meaningful performance problems.
---

# Code review

Read `AGENTS.md`, the diff, and only the surrounding code needed to verify the
changed contract. Do not edit code or submit an external review unless requested.

## Establish the contract

Identify the trigger, before/after behavior, affected callers, exports, tests,
fixtures, configuration, and user-visible compatibility. Check reuse before
proposing a helper and trace the execution path before claiming a regression.

## Review the changed behavior

Prioritize relevant evidence about:

- tensor/image axes, dtype/device, logits versus probabilities, class/instance
  labels, background conventions, and preprocessing;
- autograd, train/eval state, mixed precision, export behavior, restoration after
  failure, checkpoint loading, and supported PyTorch/Python versions;
- mask/image alignment, tile stitching, metric matching, data leakage, external
  input validation, and failures that could overwrite images or checkpoints;
- full-slide memory growth, unnecessary copies/transfers/synchronization, and
  performance measured with representative data and hardware;
- direct dependencies, optional import boundaries, wheel/source installs, and
  actual execution of integration tests rather than silent skips;
- small complete changes that fit existing interfaces. Reject speculative
  machinery, cosmetic churn, and optimizations with no meaningful measured gain.

Treat each concern as a hypothesis. Verify its condition and impact against
callers, tests, or a focused reproducer. Tests that pass are evidence, not proof;
check whether plausibly wrong code could satisfy their assertions. Do not demand
expensive experiments when a targeted test establishes the contract.

## Report findings

Return only actionable, high-confidence findings, ordered by practical impact.
Each finding identifies the smallest useful file/line location, the concrete
trigger, the consequence, and the smallest adequate correction. Separate
confirmed defects from risks requiring validation. Combine one root cause into
one finding; omit stylistic nits and generic advice. If no meaningful findings
remain, say so and identify any material validation gaps.
