/**
 * Spawn execution and outcome mapping for the agent directory.
 *
 * These two steps used to live inside the extension entry point, where no test
 * could reach them: the child spawns, the JSONL stream is parsed and the outcome
 * becomes the parent tool result. They sit here now, behind an injectable
 * spawner, so the mapping from a child run to a parent result is testable with a
 * fake child and no pi, network or API key.
 *
 * The parent must see a filter violation even though the child exits 0: the
 * refusal is a tool result inside the child, so the outcome mapping lifts it
 * into the parent content (CS-30).
 */

import { spawn } from "node:child_process";

import type { Message } from "@earendil-works/pi-ai";

import type { SpawnPlan } from "./spawn-plan.ts";

export interface UsageStats {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
  cost: number;
  contextTokens: number;
  turns: number;
}

export interface SpawnOutcome {
  exitCode: number;
  messages: Message[];
  stderr: string;
  usage: UsageStats;
  stopReason?: string;
  errorMessage?: string;
  aborted: boolean;
}

export interface MappedOutcome {
  content: string;
  isError: boolean;
  details: {
    agent: string;
    tools: string[];
    model: string;
    usage: UsageStats;
    exitCode: number;
  };
  /** Refusals the child runtime produced, so callers can assert on them directly. */
  refusals: string[];
}

export interface SpawnCommandOptions {
  cwd: string;
  shell: false;
  stdio: ["ignore", "pipe", "pipe"];
}

/** The slice of ChildProcess this module drives, so a fake child can stand in. */
export interface SpawnedChild {
  stdout: { on(event: "data", listener: (chunk: Buffer) => void): unknown };
  stderr: { on(event: "data", listener: (chunk: Buffer) => void): unknown };
  on(event: "close", listener: (code: number | null, signal: string | null) => void): unknown;
  on(event: "error", listener: (error: Error) => void): unknown;
  kill(signal?: NodeJS.Signals): unknown;
}

export type Spawner = (command: string, args: string[], options: SpawnCommandOptions) => SpawnedChild;

const defaultSpawner: Spawner = (command, args, options) =>
  spawn(command, args, options) as unknown as SpawnedChild;

/** The child loop refuses an inactive tool with `Tool <name> not found`. */
const TOOL_REFUSAL_PATTERN = /Tool .* not found/;

/**
 * pi resolves an unknown model id of a *known* provider into a warning plus a
 * fabricated model, so the child keeps running on a guess. The failure has to be
 * read from stderr because the exit code stays 0 (PS-FIXER-03).
 */
const MODEL_FALLBACK_PATTERN = /not found for provider/;

export function emptyUsage(): UsageStats {
  return { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0, contextTokens: 0, turns: 0 };
}

/** Last assistant text block, which is the subagent's answer. */
export function getFinalOutput(messages: Message[]): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "assistant") {
      for (const part of message.content) {
        if (part.type === "text") {
          return part.text;
        }
      }
    }
  }
  return "";
}

function collectRefusals(messages: Message[]): string[] {
  const found: string[] = [];
  for (const message of messages) {
    if (message.role !== "toolResult" || !message.isError) {
      continue;
    }
    const parts: string[] = [];
    for (const part of message.content) {
      if (part.type === "text") {
        parts.push(part.text);
      }
    }
    const text = parts.join("\n").trim();
    if (TOOL_REFUSAL_PATTERN.test(text) && !found.includes(text)) {
      found.push(text);
    }
  }
  return found;
}

function firstModelWarning(stderr: string): string | undefined {
  for (const line of stderr.split("\n")) {
    if (MODEL_FALLBACK_PATTERN.test(line)) {
      return line.trim().replace(/^warning:\s*/i, "");
    }
  }
  return undefined;
}

function failureReason(outcome: SpawnOutcome, modelWarning: string | undefined): string | undefined {
  if (outcome.aborted) {
    return "aborted";
  }
  if (outcome.stopReason === "error") {
    return "error";
  }
  if (outcome.stopReason === "aborted") {
    return "aborted by the child";
  }
  if (modelWarning !== undefined) {
    return "model not found";
  }
  if (outcome.exitCode !== 0) {
    return `exit code ${outcome.exitCode}`;
  }
  return undefined;
}

function refusalSection(refusals: string[]): string {
  if (refusals.length === 0) {
    return "";
  }
  return ["tool refusals (runtime-enforced):", ...refusals.map((text) => `  - ${text}`)].join("\n");
}

/**
 * Turn one child run into the parent tool result.
 *
 * The parent result is the only place the calling agent can see what happened:
 * a refusal the child produced in-process, a model that pi silently replaced and
 * an unrestricted tool set are all lifted into it as loud text.
 */
export function mapOutcome(outcome: SpawnOutcome, agentName: string, plan: SpawnPlan): MappedOutcome {
  const refusals = collectRefusals(outcome.messages);
  const modelWarning = firstModelWarning(outcome.stderr);
  const failure = failureReason(outcome, modelWarning);
  const details = {
    agent: agentName,
    tools: [...plan.tools],
    // A trace without a model field reads as "no model was used"; pi-default is
    // the truth here (RTL-09).
    model: plan.resolvedModel ?? "pi-default",
    usage: outcome.usage,
    exitCode: outcome.exitCode,
  };

  const notes: string[] = [];
  if (plan.tools.length === 0) {
    notes.push(
      `WARNING: agent "${agentName}" declares no tools — child runs with pi's FULL default tool set.`,
    );
  }
  // The warning doubles as the failure line when the model is the reason, so it
  // is only repeated here for a run that failed for another reason.
  if (modelWarning !== undefined && failure !== "model not found") {
    notes.push(`WARNING: ${modelWarning}`);
  }

  const sections: string[] = [];
  if (failure !== undefined) {
    const loud =
      outcome.errorMessage ||
      modelWarning ||
      outcome.stderr.trim() ||
      getFinalOutput(outcome.messages) ||
      "(no output)";
    sections.push(`Agent ${agentName} failed (${failure}): ${loud}`);
  } else {
    sections.push(getFinalOutput(outcome.messages) || "(no output)");
  }
  sections.push(...notes);
  const refusalsText = refusalSection(refusals);
  if (refusalsText !== "") {
    sections.push(refusalsText);
  }

  return {
    content: sections.join("\n\n"),
    isError: failure !== undefined,
    details,
    refusals,
  };
}

/**
 * Run one spawn plan and collect the child's protocol events.
 *
 * `spawner` is injectable so tests can drive the close/error/abort paths without
 * a real process; production always uses child_process.spawn.
 */
export async function runSpawnPlan(
  plan: SpawnPlan,
  abortSignal?: AbortSignal,
  spawner: Spawner = defaultSpawner,
): Promise<SpawnOutcome> {
  const outcome: SpawnOutcome = {
    exitCode: 0,
    messages: [],
    stderr: "",
    usage: emptyUsage(),
    aborted: false,
  };

  // spawn() reports a missing binary through the `error` event, never by
  // throwing, so there is no try/catch here.
  const child = spawner(plan.command, plan.args, {
    cwd: plan.cwd,
    shell: false,
    stdio: ["ignore", "pipe", "pipe"],
  });

  let buffer = "";

  const processLine = (line: string): void => {
    if (!line.trim()) {
      return;
    }
    let event: { type?: string; message?: Message };
    try {
      event = JSON.parse(line) as { type?: string; message?: Message };
    } catch {
      // Non-JSON lines carry no protocol meaning; stderr keeps the raw text.
      return;
    }

    // Tool results arrive as message_end too (agent-loop emits one per
    // finalized tool result), so this single branch covers both roles.
    if (event.type === "message_end" && event.message) {
      const message = event.message;
      outcome.messages.push(message);
      if (message.role === "assistant") {
        outcome.usage.turns += 1;
        const usage = message.usage;
        if (usage) {
          outcome.usage.input += usage.input || 0;
          outcome.usage.output += usage.output || 0;
          outcome.usage.cacheRead += usage.cacheRead || 0;
          outcome.usage.cacheWrite += usage.cacheWrite || 0;
          outcome.usage.cost += usage.cost?.total || 0;
          outcome.usage.contextTokens = usage.totalTokens || 0;
        }
        if (message.stopReason) {
          outcome.stopReason = message.stopReason;
        }
        if (message.errorMessage) {
          outcome.errorMessage = message.errorMessage;
        }
      }
    }
  };

  const killChild = (): void => {
    outcome.aborted = true;
    child.kill("SIGTERM");
    const force = setTimeout(() => {
      child.kill("SIGKILL");
    }, 5000);
    force.unref?.();
  };

  const exitCode = await new Promise<number>((resolveExit) => {
    let settled = false;
    const settle = (code: number): void => {
      if (!settled) {
        settled = true;
        resolveExit(code);
      }
    };

    child.stdout.on("data", (data: Buffer) => {
      buffer += data.toString();
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        processLine(line);
      }
    });

    child.stderr.on("data", (data: Buffer) => {
      outcome.stderr += data.toString();
    });

    child.on("close", (code, signalName) => {
      if (buffer.trim()) {
        processLine(buffer);
      }
      // A signal-terminated child that we did not abort still failed; 137 keeps
      // it from being read as a clean run with no output.
      abortSignal?.removeEventListener("abort", killChild);
      settle(code ?? (signalName ? 137 : 0));
    });

    child.on("error", (error) => {
      outcome.stderr += error instanceof Error ? error.message : String(error);
      abortSignal?.removeEventListener("abort", killChild);
      settle(127);
    });

    if (abortSignal) {
      if (abortSignal.aborted) {
        killChild();
      } else {
        abortSignal.addEventListener("abort", killChild, { once: true });
      }
    }
  });

  outcome.exitCode = exitCode;
  return outcome;
}
