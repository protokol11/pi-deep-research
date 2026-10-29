/**
 * Pure spawn planning for the agent directory.
 *
 * This module decides every argument the child `pi` process receives, so the
 * argument assembly is testable without the pi runtime and without spawning
 * anything. It imports node builtins and ./types.ts only: the test suite runs it
 * under plain `node --experimental-strip-types`.
 *
 * The tool list reaches the child through `--tools`, so the pi runtime of the
 * child enforces the filter (CS-30): a refusal comes from the process, not from
 * the prompt.
 */

import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";

import type { AgentCatalog, AgentDef } from "./types.ts";

export interface PiInvocation {
  command: string;
  args: string[];
}

export interface SpawnPlanOptions {
  cwd: string;
  /** Session model, used only when the agent file names no model. */
  modelDefault?: string;
  /** Overrides process discovery, for tests and for hosts that own the binary. */
  piCommand?: PiInvocation;
}

export interface SpawnPlan {
  command: string;
  args: string[];
  /** Temp file holding the agent prompt, or null when the prompt was empty. */
  tempPromptPath: string | null;
  /** Model the child will use, or undefined when pi falls back to its own default. */
  resolvedModel: string | undefined;
  /** The tool filter the plan hands to the child; empty means pi's defaults apply. */
  tools: string[];
  cwd: string;
}

export interface TraceEntry {
  customType: "agent-directory";
  content: {
    agent: string;
    taskDigest: string;
    tools: string[];
    model: string | undefined;
    source: AgentDef["source"];
    startedAt: string;
  };
}

const TRACE_CUSTOM_TYPE = "agent-directory" as const;
const TASK_DIGEST_LENGTH = 12;

/**
 * Find the pi entry point the way the official subagent example does.
 *
 * When pi runs from a script (or a bun virtual script) the interpreter and the
 * script are reused. Otherwise a generic node/bun runtime cannot run pi by
 * itself, so the `pi` executable on PATH is used.
 */
export function resolvePiInvocation(args: string[]): PiInvocation {
  const currentScript = process.argv[1];
  const isBunVirtualScript = currentScript?.startsWith("/$bunfs/root/");
  if (currentScript && !isBunVirtualScript && existsSync(currentScript)) {
    return { command: process.execPath, args: [currentScript, ...args] };
  }

  const execName = basename(process.execPath).toLowerCase();
  const isGenericRuntime = /^(node|bun)(\.exe)?$/.test(execName);
  if (!isGenericRuntime) {
    return { command: process.execPath, args };
  }

  return { command: "pi", args };
}

function safeFileName(agentName: string): string {
  return agentName.replace(/[^\w.-]+/g, "_");
}

/**
 * Write the agent prompt to a private temp file.
 *
 * `--append-system-prompt` reads a path, and the prompt must not pass through a
 * shell, so a 0o600 file in a fresh directory is the narrowest channel. The
 * caller owns cleanup of both the file and its directory.
 */
function writePromptFile(agentName: string, prompt: string): string {
  const dir = mkdtempSync(join(tmpdir(), "pi-agent-directory-"));
  const filePath = join(dir, `prompt-${safeFileName(agentName)}.md`);
  writeFileSync(filePath, prompt, { encoding: "utf8", mode: 0o600 });
  return filePath;
}

export function buildSpawnPlan(agent: AgentDef, task: string, opts: SpawnPlanOptions): SpawnPlan {
  const args: string[] = ["--mode", "json", "-p", "--no-session"];

  // The agent file wins over the session model. Neither present means no
  // `--model` flag at all: pi's own default applies, and inventing a fallback
  // here would hide a missing capability (PS-FIXER-03).
  const resolvedModel = agent.model ?? opts.modelDefault;
  if (resolvedModel !== undefined && resolvedModel !== "") {
    args.push("--model", resolvedModel);
  }

  const tools = agent.tools;
  if (tools.length > 0) {
    args.push("--tools", tools.join(","));
    // `--tools` never removes MCP tools, so a read-only role could still reach
    // them. Denying the prefix is the only way to close that hole; an agent that
    // names an mcp__ tool has opted in and keeps them.
    if (!tools.some((name) => name.startsWith("mcp__"))) {
      args.push("--exclude-tools", "mcp__*");
    }
  }

  let tempPromptPath: string | null = null;
  if (agent.prompt.trim() !== "") {
    tempPromptPath = writePromptFile(agent.name, agent.prompt);
    args.push("--append-system-prompt", tempPromptPath);
  }

  args.push(`Task: ${task}`);

  const invocation = opts.piCommand ?? resolvePiInvocation(args);
  const finalArgs = opts.piCommand ? [...opts.piCommand.args, ...args] : invocation.args;

  return {
    command: invocation.command,
    args: finalArgs,
    tempPromptPath,
    resolvedModel: resolvedModel === "" ? undefined : resolvedModel,
    tools: [...tools],
    cwd: opts.cwd,
  };
}

/**
 * Build the trace record for one spawn (RTL-09).
 *
 * The task text can carry repository content, so only a digest is stored. The
 * session file then reconstructs who ran what, with which powers, on which model.
 */
export function buildTraceEntry(
  agent: AgentDef,
  task: string,
  resolvedModel: string | undefined,
  startedAt: string,
): TraceEntry {
  const taskDigest = createHash("sha256").update(task, "utf8").digest("hex").slice(0, TASK_DIGEST_LENGTH);
  return {
    customType: TRACE_CUSTOM_TYPE,
    content: {
      agent: agent.name,
      taskDigest,
      tools: [...agent.tools],
      model: resolvedModel,
      source: agent.source,
      startedAt,
    },
  };
}

export function formatAgentsList(catalog: AgentCatalog): string {
  const lines: string[] = [];

  if (catalog.agents.length === 0) {
    lines.push("No agents found.");
  }
  for (const agent of catalog.agents) {
    // An empty list is not "no restriction needed": the child gets every tool
    // pi enables by default, so the listing has to say so out loud.
    const tools = agent.tools.length > 0 ? `[${agent.tools.join(", ")}]` : "UNRESTRICTED (pi defaults)";
    const model = agent.model ?? "(pi default)";
    lines.push(`${agent.name} (${agent.source}): ${agent.description} — tools: ${tools} — model: ${model}`);
  }

  if (catalog.errors.length > 0) {
    lines.push("", "errors:");
    for (const error of catalog.errors) {
      lines.push(`  ${error.path}: ${error.error}`);
    }
  }

  if (catalog.shadowed.length > 0) {
    lines.push("", "shadowed:");
    for (const entry of catalog.shadowed) {
      lines.push(`  ${entry.name}: ${entry.path} (kept ${entry.shadowedBy})`);
    }
  }

  return lines.join("\n");
}
