/**
 * Headless tests for the spawn half of the agent directory.
 *
 * The pure planning module is exercised directly, so the argument assembly and
 * the trace shape are checked without the pi runtime, a network, or API keys.
 * The live probe runs only with AGENT_DIRECTORY_LIVE=1.
 *
 * Run: node --experimental-strip-types tests/spawn.test.ts
 */

import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { after, test } from "node:test";

import { loadAgentDirectory } from "../loader.ts";
import { buildSpawnPlan, buildTraceEntry, formatAgentsList, resolvePiInvocation } from "../spawn-plan.ts";
import type { AgentCatalog, AgentDef } from "../types.ts";

const roots: string[] = [];

after(() => {
  for (const root of roots) {
    rmSync(root, { recursive: true, force: true });
  }
});

function makeRoot(prefix: string): string {
  const root = mkdtempSync(join(tmpdir(), prefix));
  roots.push(root);
  return root;
}

function agent(changes: Partial<AgentDef> = {}): AgentDef {
  return {
    name: "reviewer",
    description: "Read-only reviewer",
    tools: ["read", "bash"],
    prompt: "You review diffs.",
    source: "project",
    path: "/repo/.pi/agents/reviewer.md",
    ...changes,
  };
}

function cleanupPlan(plan: { tempPromptPath: string | null }): void {
  if (!plan.tempPromptPath) {
    return;
  }
  rmSync(plan.tempPromptPath, { force: true });
  rmSync(dirname(plan.tempPromptPath), { recursive: true, force: true });
}

/**
 * The live probes must never inherit the gate that started them: a child that
 * re-runs this file would recurse instead of probing pi.
 */
function liveEnvWithoutGate(): NodeJS.ProcessEnv {
  const env = { ...process.env };
  delete env.AGENT_DIRECTORY_LIVE;
  return env;
}

interface ChildRun {
  code: number | null;
  stdout: string;
  stderr: string;
  timedOut: boolean;
}

/** Run a real child with a hard timeout, so a hung probe fails instead of blocking. */
function runChild(command: string, args: string[], timeoutMs: number): Promise<ChildRun> {
  return new Promise<ChildRun>((resolvePromise) => {
    const child = spawn(command, args, {
      // Windows ships `pi` as a .cmd shim; Node refuses to launch it without a
      // shell (verified: spawn("pi") reports ENOENT even though PATH has it).
      shell: process.platform === "win32",
      stdio: ["ignore", "pipe", "pipe"],
      env: liveEnvWithoutGate(),
    });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGKILL");
    }, timeoutMs);

    child.stdout.on("data", (data) => {
      stdout += data.toString();
    });
    child.stderr.on("data", (data) => {
      stderr += data.toString();
    });
    child.on("error", (error) => {
      clearTimeout(timer);
      resolvePromise({ code: 127, stdout, stderr: `${stderr}${String(error)}`, timedOut });
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      resolvePromise({ code, stdout, stderr, timedOut });
    });
  });
}

function argAfter(args: string[], flag: string): string | undefined {
  const index = args.indexOf(flag);
  return index === -1 ? undefined : args[index + 1];
}

type JsonRecord = Record<string, unknown>;

function isJsonRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Strict JSONL framing: one JSON object per LF-terminated record. Unparseable
 * lines are diagnostics, not protocol, so they are dropped instead of guessed at.
 */
function parseJsonLines(stdout: string): JsonRecord[] {
  const records: JsonRecord[] = [];
  for (const line of stdout.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }
    try {
      const parsed: unknown = JSON.parse(trimmed);
      if (isJsonRecord(parsed)) {
        records.push(parsed);
      }
    } catch {
      // Non-JSON output carries no tool declaration; never a reason to fail here.
    }
  }
  return records;
}

/** Every message the child emitted, from `message_end` and the `agent_end` batch. */
function childMessages(records: JsonRecord[]): JsonRecord[] {
  const messages: JsonRecord[] = [];
  for (const record of records) {
    if (isJsonRecord(record.message)) {
      messages.push(record.message);
    }
    if (Array.isArray(record.messages)) {
      for (const entry of record.messages) {
        if (isJsonRecord(entry)) {
          messages.push(entry);
        }
      }
    }
  }
  return messages;
}

/**
 * Tool names the child actually declared to its model.
 *
 * `pi --tools read` prunes the declaration, so a tool outside the list never
 * reaches the model at all. The declaration lives in the system message's
 * `toolsAdded` array; the whole-line text is deliberately not string-matched.
 */
function declaredToolNames(records: JsonRecord[]): { names: string[]; systemMessages: number } {
  const names: string[] = [];
  let systemMessages = 0;
  for (const message of childMessages(records)) {
    if (message.role !== "system") {
      continue;
    }
    systemMessages += 1;
    if (Array.isArray(message.toolsAdded)) {
      for (const entry of message.toolsAdded) {
        if (isJsonRecord(entry) && typeof entry.name === "string") {
          names.push(entry.name);
        }
      }
    }
  }
  return { names, systemMessages };
}

/** Tool calls the child emitted for a tool its own runtime had pruned. */
function inactiveToolAttempts(records: JsonRecord[], allowedTools: string[]): string[] {
  const attempts: string[] = [];
  for (const message of childMessages(records)) {
    if (message.role === "assistant" && Array.isArray(message.content)) {
      for (const part of message.content) {
        if (isJsonRecord(part) && part.type === "toolCall" && typeof part.name === "string" && !allowedTools.includes(part.name)) {
          attempts.push(part.name);
        }
      }
    }
    if (message.role === "toolResult" && message.isError === true) {
      const text = Array.isArray(message.content)
        ? message.content
            .filter((part): part is JsonRecord => isJsonRecord(part) && part.type === "text")
            .map((part) => String(part.text ?? ""))
            .join("\n")
        : "";
      if (/Tool .* not found/.test(text)) {
        attempts.push(String(message.toolName ?? "unknown"));
      }
    }
  }
  return attempts;
}

test("agent model wins over the session default", () => {
  const plan = buildSpawnPlan(agent({ model: "anthropic/claude-sonnet-4" }), "check the diff", {
    cwd: "/repo",
    modelDefault: "openai/gpt-5",
    piCommand: { command: "pi", args: [] },
  });

  assert.equal(plan.command, "pi");
  assert.equal(argAfter(plan.args, "--model"), "anthropic/claude-sonnet-4");
  assert.equal(plan.resolvedModel, "anthropic/claude-sonnet-4");
  assert.deepEqual(plan.args.slice(0, 4), ["--mode", "json", "-p", "--no-session"]);
  cleanupPlan(plan);
});

test("session default is used when the agent names no model", () => {
  const plan = buildSpawnPlan(agent(), "check the diff", {
    cwd: "/repo",
    modelDefault: "openai/gpt-5",
    piCommand: { command: "pi", args: [] },
  });

  assert.equal(argAfter(plan.args, "--model"), "openai/gpt-5");
  assert.equal(plan.resolvedModel, "openai/gpt-5");
  cleanupPlan(plan);
});

test("no model flag when neither agent nor session supplies one", () => {
  const plan = buildSpawnPlan(agent(), "check the diff", {
    cwd: "/repo",
    piCommand: { command: "pi", args: [] },
  });

  assert.equal(plan.args.includes("--model"), false);
  assert.equal(plan.resolvedModel, undefined);
  cleanupPlan(plan);
});

test("tool list is passed as one comma-joined --tools value", () => {
  const plan = buildSpawnPlan(agent({ tools: ["read", "bash", "code-review"] }), "task", {
    cwd: "/repo",
    piCommand: { command: "pi", args: [] },
  });

  assert.equal(argAfter(plan.args, "--tools"), "read,bash,code-review");
  cleanupPlan(plan);
});

test("no --tools flag when the agent declares no tools", () => {
  const plan = buildSpawnPlan(agent({ tools: [] }), "task", {
    cwd: "/repo",
    piCommand: { command: "pi", args: [] },
  });

  assert.equal(plan.args.includes("--tools"), false);
  cleanupPlan(plan);
});

test("prompt goes to a 0o600 temp file referenced by --append-system-prompt", () => {
  const prompt = "You are the reviewer.\nReport findings only.";
  const plan = buildSpawnPlan(agent({ prompt }), "check the diff", {
    cwd: "/repo",
    piCommand: { command: "pi", args: [] },
  });

  assert.notEqual(plan.tempPromptPath, null);
  const tempPromptPath = plan.tempPromptPath as string;
  assert.equal(existsSync(tempPromptPath), true);
  assert.equal(readFileSync(tempPromptPath, "utf8"), prompt);
  // POSIX permission bits only; Windows reports synthetic modes.
  if (process.platform !== "win32") {
    assert.equal(statSync(tempPromptPath).mode & 0o777, 0o600);
  }
  assert.equal(argAfter(plan.args, "--append-system-prompt"), tempPromptPath);
  assert.equal(plan.args[plan.args.length - 1], "Task: check the diff");

  cleanupPlan(plan);
  assert.equal(existsSync(tempPromptPath), false);
});

test("an empty prompt adds no --append-system-prompt and no temp file", () => {
  const plan = buildSpawnPlan(agent({ prompt: "   \n" }), "task", {
    cwd: "/repo",
    piCommand: { command: "pi", args: [] },
  });

  assert.equal(plan.tempPromptPath, null);
  assert.equal(plan.args.includes("--append-system-prompt"), false);
});

test("explicit piCommand prefix is kept in front of the assembled args", () => {
  const plan = buildSpawnPlan(agent(), "task", {
    cwd: "/repo",
    piCommand: { command: "node", args: ["/usr/local/bin/pi.js"] },
  });

  assert.equal(plan.command, "node");
  assert.equal(plan.args[0], "/usr/local/bin/pi.js");
  assert.equal(plan.args[1], "--mode");
  assert.equal(plan.cwd, "/repo");
  cleanupPlan(plan);
});

test("resolvePiInvocation always yields a command and an args array", () => {
  const invocation = resolvePiInvocation(["--version"]);
  assert.equal(typeof invocation.command, "string");
  assert.equal(invocation.command.length > 0, true);
  assert.equal(Array.isArray(invocation.args), true);
  assert.equal(invocation.args[invocation.args.length - 1], "--version");
});

test("a restricted tool list also denies MCP tools, which --tools never removes", () => {
  const plan = buildSpawnPlan(agent({ tools: ["read"] }), "task", {
    cwd: "/repo",
    piCommand: { command: "pi", args: [] },
  });

  assert.equal(argAfter(plan.args, "--exclude-tools"), "mcp__*");
  cleanupPlan(plan);

  const mcpPlan = buildSpawnPlan(agent({ tools: ["read", "mcp__radius__query"] }), "task", {
    cwd: "/repo",
    piCommand: { command: "pi", args: [] },
  });
  assert.equal(mcpPlan.args.includes("--exclude-tools"), false);
  cleanupPlan(mcpPlan);
});

test("plan records the tool filter it hands to the child", () => {
  const plan = buildSpawnPlan(agent({ tools: ["read", "bash"] }), "task", {
    cwd: "/repo",
    piCommand: { command: "pi", args: [] },
  });
  assert.deepEqual(plan.tools, ["read", "bash"]);
  cleanupPlan(plan);

  const unrestricted = buildSpawnPlan(agent({ tools: [] }), "task", {
    cwd: "/repo",
    piCommand: { command: "pi", args: [] },
  });
  assert.deepEqual(unrestricted.tools, []);
  cleanupPlan(unrestricted);
});

test("trace entry carries the agent, a 12 char task digest, tools and model", () => {
  const trace = buildTraceEntry(agent({ model: "anthropic/claude-sonnet-4" }), "check the diff", "anthropic/claude-sonnet-4", "2026-10-23T10:00:00.000Z");

  assert.equal(trace.customType, "agent-directory");
  assert.equal(trace.content.agent, "reviewer");
  assert.equal(trace.content.taskDigest.length, 12);
  assert.match(trace.content.taskDigest, /^[0-9a-f]{12}$/);
  assert.deepEqual(trace.content.tools, ["read", "bash"]);
  assert.equal(trace.content.model, "anthropic/claude-sonnet-4");
  assert.equal(trace.content.source, "project");
  assert.equal(trace.content.startedAt, "2026-10-23T10:00:00.000Z");
  // The digest must not equal the task text, and equal tasks must digest equally.
  assert.notEqual(trace.content.taskDigest, "check the diff");
  assert.equal(buildTraceEntry(agent(), "check the diff", undefined, "x").content.taskDigest, trace.content.taskDigest);
  assert.notEqual(buildTraceEntry(agent(), "another task", undefined, "x").content.taskDigest, trace.content.taskDigest);
  assert.equal(trace.content.model === undefined, false);
});

test("trace entry keeps an unresolved model undefined instead of inventing one", () => {
  const trace = buildTraceEntry(agent(), "task", undefined, "2026-10-23T10:00:00.000Z");
  assert.equal(trace.content.model, undefined);
  assert.equal(trace.content.source, "project");
});

test("formatAgentsList renders agents, errors and shadowed sections", () => {
  const catalog: AgentCatalog = {
    agents: [
      agent({ name: "reviewer", tools: ["read", "bash"], model: "anthropic/claude-sonnet-4" }),
      agent({ name: "writer", description: "Writes docs", tools: [], source: "user", model: undefined }),
    ],
    errors: [{ path: "/repo/.pi/agents/broken.md", error: 'missing required string field "name"' }],
    shadowed: [{ name: "writer", shadowedBy: "/repo/.pi/agents/writer.md", path: "/home/u/.agents/agents/writer.md" }],
  };

  const output = formatAgentsList(catalog);
  const lines = output.split("\n");

  assert.match(lines[0], /^reviewer \(project\): Read-only reviewer — tools: \[read, bash\] — model: anthropic\/claude-sonnet-4$/);
  assert.match(lines[1], /^writer \(user\): Writes docs — tools: /);
  assert.match(lines[1], /^writer \(user\): Writes docs — tools: UNRESTRICTED \(pi defaults\) — model: \(pi default\)$/);
  assert.match(lines[1], /model: \(pi default\)$/);
  const errorsAt = lines.indexOf("errors:");
  assert.notEqual(errorsAt, -1);
  assert.match(lines[errorsAt + 1], /broken\.md: missing required string field/);
  const shadowedAt = lines.indexOf("shadowed:");
  assert.notEqual(shadowedAt, -1);
  assert.match(lines[shadowedAt + 1], /writer: .*writer\.md \(kept .*writer\.md\)$/);
});

test("formatAgentsList states an empty catalog without extra sections", () => {
  const output = formatAgentsList({ agents: [], errors: [], shadowed: [] });
  assert.equal(output, "No agents found.");
});

test("a catalog loaded from temp scopes builds a plan for the loaded agent", () => {
  const root = makeRoot("agent-directory-spawn-");
  const projectRoot = join(root, "project");
  const userRoot = join(root, "user");
  mkdirSync(join(projectRoot, ".pi", "agents"), { recursive: true });
  mkdirSync(join(userRoot, ".agents", "agents"), { recursive: true });

  writeFileSync(
    join(projectRoot, ".pi", "agents", "reviewer.md"),
    '---\nname: reviewer\ndescription: Read-only reviewer\ntools: [read, bash]\nmodel: anthropic/claude-sonnet-4\n---\nYou review diffs.\n',
    "utf8",
  );
  writeFileSync(
    join(userRoot, ".agents", "agents", "scribe.md"),
    "---\nname: scribe\ndescription: Writes notes\ntools: write\n---\nYou write notes.\n",
    "utf8",
  );

  const catalog = loadAgentDirectory({
    projectRoot,
    userRoot,
    manifestPath: join(projectRoot, ".pi", "agent-directory", "catalog.json"),
  });

  assert.equal(catalog.agents.length, 2);
  assert.equal(catalog.errors.length, 0);

  const reviewer = catalog.agents.find((entry) => entry.name === "reviewer") as AgentDef;
  assert.equal(existsSync(reviewer.path), true);

  const plan = buildSpawnPlan(reviewer, "check the diff since HEAD", {
    cwd: projectRoot,
    modelDefault: "openai/gpt-5",
    piCommand: { command: "pi", args: [] },
  });
  assert.equal(argAfter(plan.args, "--model"), "anthropic/claude-sonnet-4");
  assert.equal(argAfter(plan.args, "--tools"), "read,bash");
  assert.equal(readFileSync(plan.tempPromptPath as string, "utf8"), "You review diffs.");
  const trace = buildTraceEntry(reviewer, "check the diff since HEAD", plan.resolvedModel, "2026-10-23T10:00:00.000Z");
  assert.equal(trace.content.source, "project");
  cleanupPlan(plan);

  const scribe = catalog.agents.find((entry) => entry.name === "scribe") as AgentDef;
  const scribePlan = buildSpawnPlan(scribe, "write notes", { cwd: projectRoot, piCommand: { command: "pi", args: [] } });
  assert.equal(scribePlan.args.includes("--model"), false);
  assert.equal(argAfter(scribePlan.args, "--tools"), "write");
  cleanupPlan(scribePlan);
});

test(
  "live probe: an unknown model fails loudly in json mode",
  { skip: process.env.AGENT_DIRECTORY_LIVE !== "1" ? "set AGENT_DIRECTORY_LIVE=1 to run" : false },
  async () => {
    // Forced to the `pi` executable on purpose: resolving the invocation inside a
    // test file re-runs this suite instead of probing pi.
    const args = [
      "--mode",
      "json",
      "-p",
      "--no-session",
      "--model",
      "nonexistent-provider/nonexistent-model",
      "hi",
    ];

    const result = await runChild("pi", args, 120_000);

    assert.equal(result.timedOut, false, "the probe did not finish within 120 s");
    assert.notEqual(result.code, 0, `expected a non-zero exit, got stdout: ${result.stdout}`);
    assert.equal((result.stderr + result.stdout).trim().length > 0, true, "expected a loud message");
  },
);

// Tools the read-only fixture must never be able to reach. `edit`/`write` are
// the file mutators, `bash` is the escape hatch that could write through a shell.
const PRUNED_TOOLS = ["edit", "write", "bash"];

test(
  "live probe: a read-only child declares only `read` and cannot write (declaration pruning + behavioral absence)",
  { skip: process.env.AGENT_DIRECTORY_LIVE !== "1" ? "set AGENT_DIRECTORY_LIVE=1 to run" : false },
  async () => {
    // Why this probe asks for a write instead of a refusal: `pi --tools read`
    // removes `edit` from the model's tool *declaration*, so a compliant model
    // never emits the call and the child runtime never gets to refuse it (exit 0,
    // no toolCall — observed on deepseek-flash). The oracle is therefore double:
    // (1) the declaration carries only `read` and (2) the capability is absent
    // behaviorally — a task that cannot be fulfilled without a write leaves no
    // file behind (CS-30). The runtime refusal path itself is real and is
    // unit-covered with a fake-child JSONL in tests/spawn-run.test.ts.
    const root = makeRoot("agent-directory-readonly-");
    const projectRoot = join(root, "project");
    const userRoot = join(root, "user");
    mkdirSync(join(projectRoot, ".pi", "agents"), { recursive: true });
    mkdirSync(join(userRoot, ".agents", "agents"), { recursive: true });
    // Inside the child's own cwd on purpose: the write is denied because the tool
    // is absent, not because the path sits outside a writable root.
    const canaryDir = join(projectRoot, "canary");
    mkdirSync(canaryDir, { recursive: true });
    const canaryPath = join(canaryDir, "DONE.txt");

    writeFileSync(
      join(projectRoot, ".pi", "agents", "read-only.md"),
      "---\nname: read-only\ndescription: Read-only assistant\ntools: [read]\n---\nYou are a read-only assistant.\n",
      "utf8",
    );

    const catalog = loadAgentDirectory({
      projectRoot,
      userRoot,
      manifestPath: join(projectRoot, ".pi", "agent-directory", "catalog.json"),
    });
    const readOnly = catalog.agents.find((entry) => entry.name === "read-only");
    assert.notEqual(readOnly, undefined);

    // Fulfilling the task requires a write; a read-only child can only fail to do it.
    const task =
      `Write the word DONE into the file ${canaryPath} using your tools, then reply with the single word WRITTEN. ` +
      "Do not just describe the write — perform it.";
    // Forced to the `pi` executable: resolving the invocation inside a test file
    // re-runs this suite instead of probing pi. No --model, so pi's default applies.
    const plan = buildSpawnPlan(readOnly as AgentDef, task, {
      cwd: projectRoot,
      piCommand: { command: "pi", args: [] },
    });

    try {
      assert.equal(plan.command, "pi");
      const result = await runChild(plan.command, plan.args, 120_000);

      assert.equal(result.timedOut, false, "the child was killed by the 120 s timeout guard");

      // (a) Behavioral oracle: the disallowed capability did not land on disk.
      assert.equal(
        existsSync(canaryPath),
        false,
        `the read-only child wrote ${canaryPath}. exit=${result.code} stdout=${result.stdout}\n`,
      );

      // (b) Declaration pruning: the child told its model only about `read`.
      const records = parseJsonLines(result.stdout);
      const declared = declaredToolNames(records);
      const leaked = declared.names.filter((name) => PRUNED_TOOLS.includes(name));
      assert.deepEqual(
        leaked,
        [],
        `the child declared pruned tool(s) to its model: ${leaked.join(", ")} (declared: ${declared.names.join(", ") || "none"}) over ${declared.systemMessages} system message(s)`,
      );

      // (c) The run completed: absence of the capability is not a crash.
      assert.equal(result.code, 0, `expected a clean exit, got stdout=${result.stdout} stderr=${result.stderr}`);

      // Soft, model-dependent: a model that hallucinates a pruned tool call makes
      // the runtime refuse it in-band. Only then is the refusal text expected.
      const attempts = inactiveToolAttempts(records, plan.tools);
      if (attempts.length > 0) {
        assert.match(
          result.stdout,
          /Tool .* not found/,
          `the child attempted pruned tool(s) ${attempts.join(", ")} but reported no runtime refusal. exit=${result.code} stderr=${result.stderr}`,
        );
      }
    } finally {
      cleanupPlan(plan);
    }
  },
);
