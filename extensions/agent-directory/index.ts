/**
 * Agent directory: the pi-facing half.
 *
 * The loader (./loader.ts) turns agent markdown into a typed catalog, the
 * planner (./spawn-plan.ts) decides the child's arguments, and ./spawn-run.ts
 * runs the child and maps its outcome. This file only wires those parts to pi:
 * reload on change, register one `agent(name, task)` tool, spawn the named role.
 *
 * Only the catalog names reach the model (CP-22): the parameter enum is built
 * from the loaded catalog, never from free text. The role's tool list travels to
 * the child through `--tools`, so the child runtime enforces the filter (CS-30).
 */

import { readdirSync, rmSync, statSync, unlinkSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

import { loadAgentDirectory } from "./loader.ts";
import { buildSpawnPlan, buildTraceEntry, formatAgentsList, type SpawnPlan } from "./spawn-plan.ts";
import { emptyUsage, mapOutcome, runSpawnPlan } from "./spawn-run.ts";
import type { AgentCatalog } from "./types.ts";

// The loader owns scope resolution and publishes no accessor for the two
// directories. The snapshot repeats the paths because it only decides when to
// reload: a drift here costs a reload, never a wrong load.
const PROJECT_AGENT_DIR = join(".pi", "agents");
const USER_AGENT_DIR = join(".agents", "agents");

function snapshotDir(dir: string): string {
  try {
    const names = readdirSync(dir).sort();
    const parts: string[] = [];
    for (const name of names) {
      try {
        parts.push(`${name}:${statSync(join(dir, name)).mtimeMs}`);
      } catch {
        // A file that vanishes between readdir and stat still changes the
        // snapshot, which is all this comparison needs.
        parts.push(`${name}:?`);
      }
    }
    return parts.join(",");
  } catch (error) {
    const code = (error as { code?: string }).code;
    return code === "ENOENT" ? "missing" : `error:${code ?? String(error)}`;
  }
}

function snapshotAgentDirs(projectRoot: string): string {
  return [snapshotDir(join(projectRoot, PROJECT_AGENT_DIR)), snapshotDir(join(homedir(), USER_AGENT_DIR))].join("|");
}

export default function (pi: ExtensionAPI) {
  let catalog: AgentCatalog | null = null;
  let snapshot = "";
  let registeredNames = "";

  const reload = (cwd: string): AgentCatalog => {
    const next = loadAgentDirectory({ projectRoot: cwd });
    catalog = next;
    snapshot = snapshotAgentDirs(cwd);
    syncTool();
    return next;
  };

  /**
   * Reload when the agent files changed, and only then.
   *
   * mtime is checked at call time instead of on a timer: a second polling
   * mechanism would answer the same question with more moving parts. The check
   * is a string comparison over directory listings, so it can also run before
   * every agent start.
   */
  const currentCatalog = (cwd: string): AgentCatalog => {
    if (catalog === null || snapshotAgentDirs(cwd) !== snapshot) {
      return reload(cwd);
    }
    return catalog;
  };

  const registerAgentTool = (names: string[], exposure?: "hidden"): void => {
    // StringEnum demands a non-empty tuple; the withdrawn registration has no
    // name to offer, and it is never declared to the model.
    const nameSchema = names.length > 0 ? StringEnum(names as [string, ...string[]]) : Type.String();

    pi.registerTool({
      name: "agent",
      label: "Agent",
      description: [
        "Run a subagent defined in the agent directory.",
        "Names come from .pi/agents/*.md (project) and .agents/agents/*.md (user).",
        "The agent's tool list is passed to the child pi process with --tools,",
        "so the runtime refuses any tool the agent file does not declare.",
      ].join(" "),
      parameters: Type.Object({
        name: nameSchema,
        task: Type.String({ description: "Task handed to the agent as its prompt" }),
      }),
      exposure: exposure ?? "direct",

      async execute(_toolCallId, params, signal, _onUpdate, ctx) {
        const live = currentCatalog(ctx.cwd);
        const agent = live.agents.find((entry) => entry.name === params.name);
        if (agent === undefined) {
          const available = live.agents.map((entry) => entry.name).join(", ") || "none";
          return {
            content: [{ type: "text" as const, text: `Unknown agent "${params.name}". Available: ${available}.` }],
            isError: true,
            details: { agent: params.name, tools: [], model: "pi-default", usage: emptyUsage(), exitCode: 1 },
          };
        }

        const modelDefault = ctx.model ? `${ctx.model.provider}/${ctx.model.id}` : undefined;
        const startedAt = new Date().toISOString();

        // A typo in the tool list silently narrows the child's powers, so the
        // name is checked against the live registry and reported. The spawn still
        // runs: a missing tool fails inside the child, which is where the oracle
        // lives, and refusing here would hide the rest of the role's defects.
        const knownTools = new Set(pi.getAllTools().map((tool) => tool.name));
        const unknownTools = agent.tools.filter((name) => !name.includes("*") && !knownTools.has(name));
        const unknownToolsWarning =
          unknownTools.length > 0
            ? `WARNING: agent "${agent.name}" declares unknown tool(s): ${unknownTools.join(", ")} — the child cannot offer them.`
            : "";

        let plan: SpawnPlan | null = null;
        try {
          plan = buildSpawnPlan(agent, params.task, { cwd: ctx.cwd, modelDefault });
          // Trace before the spawn: a spawn that dies still leaves a record of
          // who was asked to do what, on which model and with which tools
          // (RTL-09).
          const trace = buildTraceEntry(agent, params.task, plan.resolvedModel, startedAt);
          pi.appendEntry(trace.customType, trace.content);

          const outcome = await runSpawnPlan(plan, signal);
          const mapped = mapOutcome(outcome, agent.name, plan);
          const text = [unknownToolsWarning, mapped.content].filter((part) => part !== "").join("\n\n");

          return {
            content: [{ type: "text" as const, text }],
            isError: mapped.isError,
            details: mapped.details,
          };
        } finally {
          if (plan?.tempPromptPath) {
            try {
              unlinkSync(plan.tempPromptPath);
              rmSync(dirname(plan.tempPromptPath), { recursive: true, force: true });
            } catch {
              // Cleanup runs after the result is decided; a leftover temp file
              // in the OS temp dir must not turn a success into a failure.
            }
          }
        }
      },
    });
  };

  /**
   * Register the tool for the current catalog.
   *
   * The name is fixed, so re-registration replaces the definition instead of
   * adding a second tool; the enum and the labels stay in step with the files.
   * An emptied catalog re-registers the same name as hidden, which withdraws a
   * schema that would otherwise keep advertising agents that no longer exist.
   */
  const syncTool = (): void => {
    const names = (catalog?.agents ?? []).map((entry) => entry.name).sort();
    const key = names.join(",");
    if (key === registeredNames) {
      return;
    }
    const wasRegistered = registeredNames !== "";
    registeredNames = key;

    if (names.length === 0) {
      if (wasRegistered) {
        registerAgentTool([], "hidden");
      }
      return;
    }

    registerAgentTool(names);
  };

  pi.on("session_start", async (_event, ctx) => {
    reload(ctx.cwd);
  });

  // pi asks for the system prompt before every agent start; comparing the cheap
  // directory snapshot there means a freshly added agent is callable without
  // waiting for a tool call to trigger the reload.
  pi.on("before_agent_start", async (_event, ctx) => {
    currentCatalog(ctx.cwd);
  });

  pi.registerCommand("agents", {
    description: "List the agents available to this session",
    handler: async (_args, ctx) => {
      const live = reload(ctx.cwd);
      ctx.ui.notify(formatAgentsList(live), "info");
    },
  });
}
