/**
 * Headless tests for spawn execution and outcome mapping.
 *
 * runSpawnPlan takes an injectable spawner, so every branch that used to be
 * unreachable from a test (refusals, signal kills, stderr model fallback, empty
 * tool lists) is driven by a fake child. No pi, no network, no API key.
 *
 * Run: node --experimental-strip-types tests/spawn-run.test.ts
 */

import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";

import type { Message } from "@earendil-works/pi-ai";

import { buildSpawnPlan } from "../spawn-plan.ts";
import { mapOutcome, runSpawnPlan, type Spawner } from "../spawn-run.ts";
import type { AgentDef } from "../types.ts";

const promptDirs: string[] = [];

after(() => {
  for (const dir of promptDirs) {
    rmSync(dir, { recursive: true, force: true });
  }
});

function agent(changes: Partial<AgentDef> = {}): AgentDef {
  return {
    name: "reviewer",
    description: "Read-only reviewer",
    tools: ["read"],
    prompt: "",
    source: "project",
    path: "/repo/.pi/agents/reviewer.md",
    ...changes,
  };
}

function planFor(tools: string[], model?: string) {
  const plan = buildSpawnPlan(agent({ tools, model }), "check the diff", {
    cwd: "/repo",
    piCommand: { command: "pi", args: [] },
  });
  if (plan.tempPromptPath) {
    promptDirs.push(join(plan.tempPromptPath, ".."));
  }
  return plan;
}

/** Fake child: emits the protocol lines a real `pi --mode json` child would. */
class FakeChild extends EventEmitter {
  readonly stdout = new EventEmitter();
  readonly stderr = new EventEmitter();
  readonly killed: string[] = [];

  kill(signal?: string): boolean {
    this.killed.push(signal ?? "SIGTERM");
    return true;
  }

  line(event: unknown): void {
    this.stdout.emit("data", Buffer.from(`${JSON.stringify(event)}\n`));
  }

  assistant(text: string, stopReason = "stop"): void {
    this.line({
      type: "message_end",
      message: {
        role: "assistant",
        content: [{ type: "text", text }],
        api: "openai-completions",
        provider: "test",
        model: "test-model",
        usage: { input: 10, output: 5, cacheRead: 1, cacheWrite: 2, totalTokens: 18, cost: { input: 0, output: 0, total: 0.25 } },
        stopReason,
        timestamp: 1,
      },
    });
  }

  toolResult(toolName: string, text: string, isError: boolean): void {
    this.line({
      type: "message_end",
      message: {
        role: "toolResult",
        toolCallId: `call-${toolName}`,
        toolName,
        content: [{ type: "text", text }],
        isError,
        timestamp: 2,
      },
    });
  }

  writeStderr(text: string): void {
    this.stderr.emit("data", Buffer.from(text));
  }

  close(code: number | null, signal: string | null = null): void {
    this.emit("close", code, signal);
  }
}

function fakeSpawner(): { spawner: Spawner; child: FakeChild; calls: Array<{ command: string; args: string[]; cwd: string }> } {
  const child = new FakeChild();
  const calls: Array<{ command: string; args: string[]; cwd: string }> = [];
  const spawner: Spawner = (command, args, options) => {
    calls.push({ command, args, cwd: options.cwd });
    return child;
  };
  return { spawner, child, calls };
}

test("a tool refusal in the child surfaces in the parent content", async () => {
  const plan = planFor(["read"]);
  const { spawner, child } = fakeSpawner();

  const run = runSpawnPlan(plan, undefined, spawner);
  child.toolResult("edit", "Tool edit not found", true);
  child.assistant("I could not edit the file.");
  child.close(0);

  const outcome = await run;
  assert.equal(outcome.exitCode, 0);
  assert.equal(outcome.aborted, false);

  const mapped = mapOutcome(outcome, "reviewer", plan);
  assert.deepEqual(mapped.refusals, ["Tool edit not found"]);
  assert.match(mapped.content, /^I could not edit the file\./);
  assert.match(mapped.content, /tool refusals \(runtime-enforced\):\n {2}- Tool edit not found/);
  // The child exited cleanly; a refusal is visible but is not itself a failure.
  assert.equal(mapped.isError, false);
});

test("a successful run maps to the final assistant text, usage and a traced model", async () => {
  const plan = planFor(["read"]);
  const { spawner, child, calls } = fakeSpawner();

  const run = runSpawnPlan(plan, undefined, spawner);
  child.assistant("Looks good.");
  child.close(0);

  const outcome = await run;
  const mapped = mapOutcome(outcome, "reviewer", plan);

  assert.equal(calls.length, 1);
  assert.equal(calls[0].command, "pi");
  assert.deepEqual(calls[0].args, plan.args);
  assert.equal(calls[0].cwd, "/repo");
  assert.equal(mapped.content, "Looks good.");
  assert.equal(mapped.isError, false);
  assert.equal(mapped.details.model, "pi-default");
  assert.deepEqual(mapped.details.tools, ["read"]);
  assert.equal(mapped.details.usage.turns, 1);
  assert.equal(mapped.details.usage.input, 10);
  assert.equal(mapped.details.usage.contextTokens, 18);
  assert.equal(mapped.details.usage.cost, 0.25);
  assert.deepEqual(mapped.refusals, []);
});

test("a non-zero exit fails the run and keeps the child's stderr", async () => {
  const plan = planFor(["read"]);
  const { spawner, child } = fakeSpawner();

  const run = runSpawnPlan(plan, undefined, spawner);
  child.writeStderr("boom: provider unreachable\n");
  child.close(1);

  const mapped = mapOutcome(await run, "reviewer", plan);
  assert.equal(mapped.isError, true);
  assert.match(mapped.content, /^Agent reviewer failed \(exit code 1\): boom: provider unreachable/);
  assert.equal(mapped.details.exitCode, 1);
});

test("a signal kill with no abort is a failure, not an empty success", async () => {
  const plan = planFor(["read"]);
  const { spawner, child } = fakeSpawner();

  const run = runSpawnPlan(plan, undefined, spawner);
  child.close(null, "SIGSEGV");

  const outcome = await run;
  assert.equal(outcome.exitCode, 137);
  assert.equal(outcome.aborted, false);

  const mapped = mapOutcome(outcome, "reviewer", plan);
  assert.equal(mapped.isError, true);
  assert.match(mapped.content, /Agent reviewer failed \(exit code 137\)/);
});

test("stopReason aborted fails the run even with exit code 0", async () => {
  const plan = planFor(["read"]);
  const { spawner, child } = fakeSpawner();

  const run = runSpawnPlan(plan, undefined, spawner);
  child.assistant("partial answer", "aborted");
  child.close(0);

  const mapped = mapOutcome(await run, "reviewer", plan);
  assert.equal(mapped.isError, true);
  assert.match(mapped.content, /Agent reviewer failed \(aborted by the child\)/);
});

test("a model pi silently replaced is a loud failure", async () => {
  const plan = planFor(["read"], "anthropic/claude-does-not-exist");
  const { spawner, child } = fakeSpawner();

  const run = runSpawnPlan(plan, undefined, spawner);
  child.writeStderr('Warning: Model "claude-does-not-exist" not found for provider "anthropic". Using custom model id.\n');
  child.assistant("answered anyway");
  child.close(0);

  const mapped = mapOutcome(await run, "reviewer", plan);
  assert.equal(mapped.isError, true);
  assert.match(
    mapped.content,
    /^Agent reviewer failed \(model not found\): Model "claude-does-not-exist" not found for provider "anthropic"\. Using custom model id\.$/,
  );
  assert.equal(mapped.details.model, "anthropic/claude-does-not-exist");
});

test("a model warning still shows as a note when the run failed for another reason", async () => {
  const plan = planFor(["read"], "anthropic/claude-does-not-exist");
  const { spawner, child } = fakeSpawner();

  const run = runSpawnPlan(plan, undefined, spawner);
  child.writeStderr('Model "claude-does-not-exist" not found for provider "anthropic". Using custom model id.\n');
  child.assistant("answered anyway", "aborted");
  child.close(0);

  const mapped = mapOutcome(await run, "reviewer", plan);
  assert.equal(mapped.isError, true);
  assert.match(mapped.content, /Agent reviewer failed \(aborted by the child\)/);
  assert.match(mapped.content, /WARNING: Model "claude-does-not-exist" not found for provider/);
});

test("an empty tool list warns that the child runs with pi's defaults", async () => {
  const plan = planFor([]);
  const { spawner, child } = fakeSpawner();

  const run = runSpawnPlan(plan, undefined, spawner);
  child.assistant("done");
  child.close(0);

  const mapped = mapOutcome(await run, "reviewer", plan);
  assert.match(
    mapped.content,
    /^done\n\nWARNING: agent "reviewer" declares no tools — child runs with pi's FULL default tool set\.$/,
  );
  assert.deepEqual(mapped.details.tools, []);
  assert.equal(mapped.isError, false);
});

test("aborting the signal kills the child and fails the run", async () => {
  const plan = planFor(["read"]);
  const { spawner, child } = fakeSpawner();
  const controller = new AbortController();

  const run = runSpawnPlan(plan, controller.signal, spawner);
  controller.abort();
  child.close(null, "SIGTERM");

  const outcome = await run;
  assert.equal(outcome.aborted, true);
  assert.deepEqual(child.killed, ["SIGTERM"]);

  const mapped = mapOutcome(outcome, "reviewer", plan);
  assert.equal(mapped.isError, true);
  assert.match(mapped.content, /Agent reviewer failed \(aborted\)/);
});

test("a missing binary reports spawn failure instead of an empty run", async () => {
  const plan = planFor(["read"]);
  const child = new FakeChild();
  const spawner: Spawner = () => child;

  const run = runSpawnPlan(plan, undefined, spawner);
  child.emit("error", new Error("spawn pi ENOENT"));
  child.close(null, null);

  const outcome = await run;
  assert.equal(outcome.exitCode, 127);
  const mapped = mapOutcome(outcome, "reviewer", plan);
  assert.equal(mapped.isError, true);
  assert.match(mapped.content, /spawn pi ENOENT/);
});

test("a partial trailing line is processed when the child closes", async () => {
  const plan = planFor(["read"]);
  const { spawner, child } = fakeSpawner();

  const run = runSpawnPlan(plan, undefined, spawner);
  const message: Message = {
    role: "assistant",
    content: [{ type: "text", text: "no newline" }],
    api: "openai-completions",
    provider: "test",
    model: "test-model",
    usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2, cost: { input: 0, output: 0, total: 0 } },
    stopReason: "stop",
    timestamp: 3,
  };
  child.stdout.emit("data", Buffer.from(JSON.stringify({ type: "message_end", message })));
  child.close(0);

  const mapped = mapOutcome(await run, "reviewer", plan);
  assert.equal(mapped.content, "no newline");
});
