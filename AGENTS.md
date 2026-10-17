# CrispEmbed agent entry point

Read the [current state and executable lanes](https://github.com/CrispStrobe/CrispEmbed/blob/main/docs/current-state-and-next-steps.md) and
[CLAUDE.md](https://github.com/CrispStrobe/CrispEmbed/blob/main/CLAUDE.md), then
consult [PLAN.md](https://github.com/CrispStrobe/CrispEmbed/blob/main/PLAN.md)
and the public [contribution checklist](https://github.com/CrispStrobe/CrispEmbed/blob/main/docs/contributing.md).
Use the [executable next-agent handoff](docs/next-agent-handoff.md) for exact
pending sources, acceptance gates and artifact preservation. Choose one bounded
lane; keep unrelated vendor work intact. The app consumes an
immutable minimal release, not arbitrary vendor main.

Use an isolated worktree for code changes. Before costly shared-host work inspect
load, RAM, swap and free space; keep builds, suites, corpus/browser/model batches
on hosted CI. Use authorized remote GPU resources for large training. Retain
large artifacts remotely and download only needed reports/images. Clean only
known task-owned transient outputs, never other projects or credentials.

Trace the real inference blueprint, compare the earliest differing stage with
magnitudes/absolute errors and final decoded output, preserve gated alternatives,
and keep reference probes out of production artifacts. A speed win or component
cosine is not decoded-quality proof. Record exact sources, initial/final evidence,
limits and dependencies in the public plan. Update versions for actual release
changes according to the repository procedure; documentation-only handoffs do
not create a runtime release.

Public Markdown uses public source/workflow/service paths. Keep host locations,
private access references and operator identities in private environment notes
outside Git. Never commit credential values. The shared handoff is mirrored in
[CrispMath](https://github.com/CrispStrobe/CrispMath/blob/main/docs/current-state-and-next-steps.md).

Optional private continuity may be present in ignored
`.claude/local-handoff/AGENTS.md`. Root `CLAUDE.md` and this file are tracked
public documents; never place machine/access details in them.
