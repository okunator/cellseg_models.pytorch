---
name: commit
description: Commit this repository's changes in small thematic chunks, preserving direct regression tests and contributor attribution. Use when committing work.
---

# Commit workflow

Read the repository's `AGENTS.md` first. This skill governs commit organization;
it does not authorize pushing, merging, publishing, or messaging contributors.

1. Inspect `git status --short` and `git diff --stat`; leave unrelated user edits
   unstaged. Read targeted diffs when needed to establish the change's scope.
2. Run the relevant checks from `AGENTS.md`. A passing skipped integration test
   is not validation; identify unverified optional packages, devices, or versions.
3. Group changes by independently understandable behavior. Keep implementation,
   its callers, exports, and direct regression tests together. Separate unrelated
   documentation, formatting, dependency upgrades, and tooling migrations.
4. Stage explicit paths and inspect `git diff --cached --stat` and
   `git diff --cached --check`. Never bypass failing hooks to finish a commit.
5. Use Conventional Commits, e.g. `fix(postproc): preserve labels at tile borders`.
   Describe the resulting behavior; preserve existing contributor attribution.
6. Update `CHANGELOG.md` for public, release-relevant changes, without inventing
   a release/version. Internal documentation and test-only work need no entry.
7. Inspect final status and report the commit and remaining changes.

Do not amend, force-push, or rewrite shared history unless the user requests it.
Do not split a fix from the tests needed to demonstrate its correctness.
