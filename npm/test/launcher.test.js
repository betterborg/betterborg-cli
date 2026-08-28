"use strict";

const assert = require("node:assert/strict");
const childProcess = require("node:child_process");
const crypto = require("node:crypto");
const { EventEmitter } = require("node:events");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");

const metadata = require("../package.json");
const {
  launch,
  reportFailure,
  resolveCli,
  targetFor,
  translateVersionArguments,
  verifiedBinary,
} = require("../lib/launcher");

function temporaryDirectory(t) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "betterborg-npm-"));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  return directory;
}

function executable(directory, name, content = "") {
  const pathname = path.join(directory, name);
  fs.writeFileSync(pathname, content, { mode: 0o755 });
  return pathname;
}

test("package metadata exposes the public scoped borg command", () => {
  assert.equal(metadata.name, "@betterborg/cli");
  assert.equal(metadata.bin.borg, "bin/borg.js");
  assert.equal(metadata.publishConfig.access, "public");
  assert.deepEqual(metadata.repository, {
    type: "git",
    url: "git+https://github.com/betterborg/betterborg-cli.git",
    directory: "npm",
  });
  assert.match(metadata.version, /^\d+\.\d+\.\d+$/);
});

test("packed package includes the license and attribution notice", () => {
  const packageRoot = path.resolve(__dirname, "..");
  const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm";
  const output = childProcess.execFileSync(
    npmCommand,
    ["pack", "--dry-run", "--json", "."],
    { cwd: packageRoot, encoding: "utf8" },
  );
  const [{ files }] = JSON.parse(output);
  const packedPaths = new Set(files.map((file) => file.path));

  assert.equal(packedPaths.has("LICENSE"), true);
  assert.equal(packedPaths.has("NOTICE"), true);
  assert.equal(
    fs.readFileSync(path.join(packageRoot, "LICENSE"), "utf8"),
    fs.readFileSync(path.join(packageRoot, "..", "LICENSE"), "utf8"),
  );
  assert.equal(
    fs.readFileSync(path.join(packageRoot, "NOTICE"), "utf8"),
    fs.readFileSync(path.join(packageRoot, "..", "NOTICE"), "utf8"),
  );
});

test("target mapping accepts only released platforms and architectures", () => {
  assert.equal(targetFor("darwin", "arm64"), "borg-darwin-arm64");
  assert.equal(targetFor("darwin", "x64"), "borg-darwin-x86_64");
  assert.equal(targetFor("linux", "arm64"), "borg-linux-arm64");
  assert.equal(targetFor("linux", "x64"), "borg-linux-x86_64");
  assert.equal(targetFor("win32", "x64"), null);
  assert.equal(targetFor("linux", "ia32"), null);
});

test("only a lone exact version flag is translated", () => {
  assert.deepEqual(translateVersionArguments(["--version"]), ["version"]);
  assert.deepEqual(translateVersionArguments(["-V"]), ["version"]);
  for (const arguments_ of [
    [],
    ["version"],
    ["--version=1"],
    ["-v"],
    ["--version", "extra"],
    ["plan", "--version"],
  ]) {
    assert.deepEqual(translateVersionArguments(arguments_), arguments_);
  }
});

test("a cached binary must match its exact checksum sidecar", (t) => {
  const directory = temporaryDirectory(t);
  const target = "borg-linux-x86_64";
  const binary = path.join(directory, target);
  const checksum = `${binary}.sha256`;
  fs.writeFileSync(binary, "trusted bytes");
  const expected = crypto
    .createHash("sha256")
    .update("trusted bytes")
    .digest("hex");
  fs.writeFileSync(checksum, `${expected}  ${target}\n`);

  assert.equal(verifiedBinary(binary, checksum, target), true);
  fs.appendFileSync(binary, "tampered");
  assert.equal(verifiedBinary(binary, checksum, target), false);
  fs.writeFileSync(binary, "trusted bytes");
  fs.writeFileSync(checksum, `${expected}  another-file\n`);
  assert.equal(verifiedBinary(binary, checksum, target), false);
});

test("resolution prefers a compatible installed CLI", async (t) => {
  const directory = temporaryDirectory(t);
  const borg = executable(directory, "borg");
  let downloads = 0;
  const resolved = await resolveCli("1.2.3", {
    architecture: "x64",
    cacheDirectory: path.join(directory, "cache"),
    download: async () => { downloads += 1; },
    environment: { PATH: directory },
    launcherPath: path.join(directory, "launcher"),
    platform: "linux",
    spawnSync: (command, arguments_) => {
      assert.equal(command, borg);
      assert.deepEqual(arguments_, ["version"]);
      return { status: 0, stdout: "borg 1.2.3\n" };
    },
  });

  assert.deepEqual(resolved, {
    command: borg,
    argumentsPrefix: [],
    source: "installed",
  });
  assert.equal(downloads, 0);
});

test("Windows resolution skips its own npm shims and falls back to uvx", async () => {
  for (const [launcherPath, shimDirectory] of [
    ["C:\\npm\\node_modules\\@betterborg\\cli\\bin\\borg.js", "C:\\npm"],
    [
      "C:\\project\\node_modules\\@betterborg\\cli\\bin\\borg.js",
      "C:\\project\\node_modules\\.bin",
    ],
  ]) {
    const available = new Set([
      `${shimDirectory}\\borg.cmd`.toLowerCase(),
      "c:\\tools\\uvx.cmd",
    ]);
    const fileSystem = {
      constants: { X_OK: 1 },
      accessSync(candidate) {
        if (!available.has(candidate.toLowerCase())) {
          throw new Error("missing");
        }
      },
      realpathSync: (candidate) => candidate,
    };
    const resolved = await resolveCli("1.2.3", {
      architecture: "x64",
      environment: { PATH: `${shimDirectory};C:\\tools`, PATHEXT: ".CMD" },
      fileSystem,
      launcherPath,
      pathDelimiter: ";",
      pathModule: path.win32,
      platform: "win32",
      spawnSync: () => assert.fail("the package's own shim must not be probed"),
    });

    assert.deepEqual(resolved, {
      command: "C:\\tools\\uvx.CMD",
      argumentsPrefix: ["--from", "betterborg==1.2.3", "borg"],
      source: "uvx",
    });
  }
});

test("POSIX resolution skips its own regular npm shim and falls back to uvx", async () => {
  const shim = "/project/node_modules/.bin/borg";
  const uvx = "/tools/uvx";
  const available = new Set([shim, uvx]);
  const fileSystem = {
    constants: { X_OK: 1 },
    accessSync(candidate) {
      if (!available.has(candidate)) {
        throw new Error("missing");
      }
    },
    realpathSync: (candidate) => candidate,
  };
  const resolved = await resolveCli("1.2.3", {
    architecture: "ia32",
    environment: { PATH: "/project/node_modules/.bin:/tools" },
    fileSystem,
    launcherPath:
      "/project/node_modules/.pnpm/@betterborg+cli@1.2.3/node_modules/@betterborg/cli/bin/borg.js",
    pathDelimiter: ":",
    pathModule: path.posix,
    platform: "linux",
    spawnSync: () => assert.fail("the package's own shim must not be probed"),
  });

  assert.deepEqual(resolved, {
    command: uvx,
    argumentsPrefix: ["--from", "betterborg==1.2.3", "borg"],
    source: "uvx",
  });
});

test("resolution downloads and verifies the target into the cache", async (t) => {
  const directory = temporaryDirectory(t);
  const cache = path.join(directory, "cache");
  const content = Buffer.from("release binary");
  const expected = crypto.createHash("sha256").update(content).digest("hex");
  const downloads = [];
  const resolved = await resolveCli("1.2.3", {
    architecture: "arm64",
    cacheDirectory: cache,
    download: async (url, destination) => {
      downloads.push(url);
      fs.writeFileSync(
        destination,
        url.endsWith(".sha256")
          ? `${expected}  borg-darwin-arm64\n`
          : content,
      );
    },
    environment: { PATH: "" },
    platform: "darwin",
    randomBytes: () => Buffer.from("12345678"),
  });

  assert.equal(resolved.command, path.join(cache, "borg-darwin-arm64"));
  assert.equal(resolved.source, "release");
  assert.equal(verifiedBinary(
    resolved.command,
    `${resolved.command}.sha256`,
    "borg-darwin-arm64",
  ), true);
  assert.deepEqual(downloads.map((url) => url.split("/").at(-1)), [
    "borg-darwin-arm64",
    "borg-darwin-arm64.sha256",
  ]);
  assert.equal(fs.statSync(resolved.command).mode & 0o777, 0o755);

  fs.chmodSync(resolved.command, 0o600);
  const reused = await resolveCli("1.2.3", {
    architecture: "arm64",
    cacheDirectory: cache,
    download: async () => {
      assert.fail("a verified cache entry must not be downloaded again");
    },
    environment: { PATH: "" },
    platform: "darwin",
  });
  assert.equal(reused.command, resolved.command);
  assert.equal(fs.statSync(reused.command).mode & 0o777, 0o755);
});

test(
  "a failed release verification falls back to uvx with an exact version",
  async (t) => {
    const directory = temporaryDirectory(t);
    const uvx = executable(directory, "uvx");
    const resolved = await resolveCli("1.2.3", {
      architecture: "x64",
      cacheDirectory: path.join(directory, "cache"),
      download: async (_url, destination) => {
        fs.writeFileSync(destination, "bad");
      },
      environment: { PATH: directory },
      platform: "linux",
      spawnSync: () => ({ status: 1, stdout: "" }),
    });

    assert.deepEqual(resolved, {
      command: uvx,
      argumentsPrefix: ["--from", "betterborg==1.2.3", "borg"],
      source: "uvx",
    });
  },
);

test("missing prerequisites produce actionable errors without stack text", async (t) => {
  const directory = temporaryDirectory(t);
  await assert.rejects(
    resolveCli("1.2.3", {
      architecture: "x64",
      cacheDirectory: path.join(directory, "cache"),
      download: async () => {
        throw new Error("network unavailable");
      },
      environment: { PATH: "" },
      platform: "linux",
    }),
    (error) => {
      assert.match(error.message, /network unavailable/);
      assert.match(error.message, /uvx is not on PATH/);
      assert.match(error.message, /Install uv/);
      assert.doesNotMatch(error.message, /\n\s+at /);
      return true;
    },
  );
});

test("entrypoint failures print only an actionable message", () => {
  const messages = [];
  const processLike = { exitCode: null };
  reportFailure(new Error("install uv and retry"), {
    console: { error: (message) => messages.push(message) },
    process: processLike,
  });

  assert.deepEqual(messages, ["borg: install uv and retry"]);
  assert.equal(processLike.exitCode, 1);
});

test("launch forwards ordinary arguments, exit status, and signals", async () => {
  const child = new EventEmitter();
  child.killed = false;
  const childSignals = [];
  child.kill = (signal) => {
    childSignals.push(signal);
    child.killed = true;
  };
  const processLike = new EventEmitter();
  processLike.pid = 123;
  processLike.exitCode = null;
  const parentSignals = [];
  processLike.kill = (pid, signal) => parentSignals.push([pid, signal]);
  let invocation;
  const completed = launch(
    { command: "/cache/borg", argumentsPrefix: [] },
    ["plan", "--version=literal", "two words"],
    {
      process: processLike,
      spawn: (command, arguments_, options) => {
        invocation = { command, arguments_, options };
        return child;
      },
    },
  );

  processLike.emit("SIGTERM");
  processLike.emit("SIGTERM");
  assert.deepEqual(childSignals, ["SIGTERM", "SIGTERM"]);
  child.emit("close", null, "SIGTERM");
  await completed;

  assert.deepEqual(invocation, {
    command: "/cache/borg",
    arguments_: ["plan", "--version=literal", "two words"],
    options: { stdio: "inherit" },
  });
  assert.deepEqual(parentSignals, [[123, "SIGTERM"]]);
  assert.equal(processLike.listenerCount("SIGTERM"), 0);
});

test("launch forwards a numeric child exit status", async () => {
  const child = new EventEmitter();
  child.killed = false;
  child.kill = () => {};
  const processLike = new EventEmitter();
  processLike.exitCode = null;
  const completed = launch(
    {
      command: "uvx",
      argumentsPrefix: ["--from", "betterborg==1.2.3", "borg"],
    },
    ["tasks"],
    { process: processLike, spawn: () => child },
  );
  child.emit("close", 23, null);
  await completed;
  assert.equal(processLike.exitCode, 23);
});
