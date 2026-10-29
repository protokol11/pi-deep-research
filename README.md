# @protokol11/pi-deep-research

Multi-agent web research swarm for pi: a planner fans out researchers over a question, a skeptic attacks every finding against its source, and a merger returns one cited report. This package bundles the runtime wiring: the `deep-research` skill plus the `agent-directory` extension that resolves the swarm's agents from declarative markdown.

The skill itself is runtime-agnostic and lives in its own repository: [protokol11/deep-research](https://github.com/protokol11/deep-research). This package is the pi install path; the sources of truth live in the [agent-governance](https://github.com/protokol11/agent-governance) monorepo.

## Why

A single agent doing research grades its own work: it judges findings with the same context that produced them. The swarm fixes this with three controls — no agent scores its own coverage, every finding lands on disk before context compacts, and a skeptic pass writes a ternary verdict (`confirmed` / `needs-revisit` / `rejected`) per claim. Rejected claims never reach the report.

## Quick start

```sh
npm install @protokol11/pi-deep-research
```

Or add the package to `packages` in your pi settings. See pi.dev for the extension API.

Then, in pi:

```text
/skill:deep-research
```

Or say: "run deep research on X". The skill asks for five policy answers (question, output path, model split, researcher count, compaction directives), then runs phases P0–P5. The final `report.md` carries a `source:` line under every claim.

## What is inside

| Path | What it is |
|---|---|
| `extensions/deep-research/SKILL.md` | The skill procedure (P0–P5 phases, gates) |
| `extensions/deep-research/REFERENCE.md` | State contract: file schemas, brief templates, gate commands |
| `extensions/agent-directory/` | pi extension: typed catalog of subagents loaded from `.pi/agents/*.md` and `.agents/agents/*.md`; provides the `agent(name, task)` tool the swarm spawns through |
| `extensions/custom-compaction.reference.ts` | Reference adaptation of pi's `session_before_compact` for compaction directives — reference material, not loaded |

## Prerequisites

- pi with subagent and web tools.
- Model ids for researchers and skeptic/merger (defaults assume `deepseek/deepseek-flash:low` and `kimi-coding/kimi-for-coding:low`; resolve yours with `subagent({action:"models"})` before a run).

## Tests

```sh
npm test
```

Runs the agent-directory loader and spawn tests (`node --experimental-strip-types`).

## Contributing

File issues in the [agent-governance](https://github.com/protokol11/agent-governance) monorepo. Skill changes belong in [protokol11/deep-research](https://github.com/protokol11/deep-research) first.

## License

MIT — see [LICENSE](./LICENSE).
