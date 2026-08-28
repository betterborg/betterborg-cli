"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { spawn, spawnSync } = require("node:child_process");

const REPOSITORY = "betterborg/betterborg-cli";
const FORWARDED_SIGNALS = ["SIGINT", "SIGTERM", "SIGHUP"];
const SIGNAL_EXIT_CODES = { SIGHUP: 129, SIGINT: 130, SIGTERM: 143 };

function targetFor(platform, architecture) {
  const operatingSystems = { darwin: "darwin", linux: "linux" };
  const architectures = { arm64: "arm64", x64: "x86_64" };
  const operatingSystem = operatingSystems[platform];
  const targetArchitecture = architectures[architecture];
  if (!operatingSystem || !targetArchitecture) {
    return null;
  }
  return `borg-${operatingSystem}-${targetArchitecture}`;
}

function translateVersionArguments(arguments_) {
  if (
    arguments_.length === 1 &&
    (arguments_[0] === "--version" || arguments_[0] === "-V")
  ) {
    return ["version"];
  }
  return [...arguments_];
}

function executableNames(name, platform, environment) {
  if (platform !== "win32") {
    return [name];
  }
  const extensions = (environment.PATHEXT || ".EXE;.CMD;.BAT;.COM")
    .split(";")
    .filter(Boolean);
  return [name, ...extensions.map((extension) => `${name}${extension}`)];
}

function samePath(left, right, platform) {
  if (platform === "win32") {
    return left.toLowerCase() === right.toLowerCase();
  }
  return left === right;
}

function launcherExecutables(dependencies) {
  if (!dependencies.launcherPath) {
    return [];
  }
  const executables = [dependencies.launcherPath];
  if (dependencies.platform !== "win32") {
    return executables;
  }

  let directory = dependencies.pathModule.dirname(dependencies.launcherPath);
  while (
    dependencies.pathModule.dirname(directory) !== directory &&
    dependencies.pathModule.basename(directory).toLowerCase() !== "node_modules"
  ) {
    directory = dependencies.pathModule.dirname(directory);
  }
  if (
    dependencies.pathModule.basename(directory).toLowerCase() !== "node_modules"
  ) {
    return executables;
  }

  for (const shimDirectory of [
    dependencies.pathModule.dirname(directory),
    dependencies.pathModule.join(directory, ".bin"),
  ]) {
    for (const name of ["borg", "borg.cmd", "borg.ps1"]) {
      executables.push(dependencies.pathModule.resolve(shimDirectory, name));
    }
  }
  return executables;
}

function executableOnPath(name, dependencies, excludedPaths = []) {
  const {
    environment,
    fileSystem,
    pathModule,
    platform,
    pathDelimiter,
  } = dependencies;
  const pathValue = environment.PATH || environment.Path || "";
  for (const directory of pathValue.split(pathDelimiter)) {
    if (!directory) {
      continue;
    }
    for (const executableName of executableNames(name, platform, environment)) {
      const candidate = pathModule.resolve(directory, executableName);
      try {
        fileSystem.accessSync(candidate, fileSystem.constants.X_OK);
        const resolvedCandidate = fileSystem.realpathSync(candidate);
        if (
          !excludedPaths.some((excludedPath) =>
            samePath(resolvedCandidate, excludedPath, platform),
          )
        ) {
          return candidate;
        }
      } catch {
        // A PATH entry that is missing or not executable is not a candidate.
      }
    }
  }
  return null;
}

function installedCli(version, dependencies) {
  const candidate = executableOnPath(
    "borg",
    dependencies,
    launcherExecutables(dependencies),
  );
  if (!candidate) {
    return null;
  }
  const completed = dependencies.spawnSync(candidate, ["version"], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "ignore"],
    timeout: 5000,
  });
  if (
    completed.status === 0 &&
    completed.stdout.trim() === `borg ${version}`
  ) {
    return candidate;
  }
  return null;
}

function digest(pathname, fileSystem = fs) {
  return crypto
    .createHash("sha256")
    .update(fileSystem.readFileSync(pathname))
    .digest("hex");
}

function verifiedBinary(binaryPath, checksumPath, target, fileSystem = fs) {
  try {
    const checksum = fileSystem.readFileSync(checksumPath, "utf8");
    const match = checksum.match(/^([a-f0-9]{64})  ([^\r\n]+)\n$/);
    return Boolean(
      match &&
        match[2] === target &&
        digest(binaryPath, fileSystem) === match[1],
    );
  } catch {
    return false;
  }
}

async function downloadToFile(url, destination) {
  const response = await fetch(url, { redirect: "follow" });
  if (!response.ok) {
    throw new Error(`download returned HTTP ${response.status} for ${url}`);
  }
  const content = Buffer.from(await response.arrayBuffer());
  fs.writeFileSync(destination, content, { flag: "wx" });
}

function defaultCacheDirectory(version, dependencies) {
  const root = dependencies.environment.XDG_CACHE_HOME
    ? dependencies.pathModule.resolve(dependencies.environment.XDG_CACHE_HOME)
    : dependencies.pathModule.join(dependencies.homeDirectory, ".cache");
  return dependencies.pathModule.join(root, "betterborg", "cli", version);
}

async function cachedRelease(version, target, dependencies) {
  const directory =
    dependencies.cacheDirectory || defaultCacheDirectory(version, dependencies);
  const binaryPath = dependencies.pathModule.join(directory, target);
  const checksumPath = dependencies.pathModule.join(directory, `${target}.sha256`);
  if (
    verifiedBinary(binaryPath, checksumPath, target, dependencies.fileSystem)
  ) {
    dependencies.fileSystem.chmodSync(binaryPath, 0o755);
    return binaryPath;
  }

  dependencies.fileSystem.mkdirSync(directory, { recursive: true, mode: 0o700 });
  const nonce = dependencies.randomBytes(8).toString("hex");
  const temporaryBinary = `${binaryPath}.${nonce}.tmp`;
  const temporaryChecksum = `${checksumPath}.${nonce}.tmp`;
  const releaseRoot = `https://github.com/${REPOSITORY}/releases/download/v${version}`;
  try {
    await dependencies.download(`${releaseRoot}/${target}`, temporaryBinary);
    await dependencies.download(
      `${releaseRoot}/${target}.sha256`,
      temporaryChecksum,
    );
    if (
      !verifiedBinary(
        temporaryBinary,
        temporaryChecksum,
        target,
        dependencies.fileSystem,
      )
    ) {
      throw new Error(`downloaded ${target} failed SHA-256 verification`);
    }
    dependencies.fileSystem.chmodSync(temporaryBinary, 0o755);
    dependencies.fileSystem.renameSync(temporaryBinary, binaryPath);
    dependencies.fileSystem.renameSync(temporaryChecksum, checksumPath);
    return binaryPath;
  } finally {
    dependencies.fileSystem.rmSync(temporaryBinary, { force: true });
    dependencies.fileSystem.rmSync(temporaryChecksum, { force: true });
  }
}

function withDefaults(overrides = {}) {
  let launcherPath = null;
  try {
    launcherPath = fs.realpathSync(process.argv[1]);
  } catch {
    // Tests and unusual embeddings may not have a filesystem entry in argv[1].
  }
  return {
    cacheDirectory: null,
    download: downloadToFile,
    environment: process.env,
    fileSystem: fs,
    homeDirectory: os.homedir(),
    launcherPath,
    pathDelimiter: path.delimiter,
    pathModule: path,
    platform: process.platform,
    architecture: process.arch,
    process,
    randomBytes: crypto.randomBytes,
    spawn,
    spawnSync,
    ...overrides,
  };
}

async function resolveCli(version, overrides = {}) {
  const dependencies = withDefaults(overrides);
  const installed = installedCli(version, dependencies);
  if (installed) {
    return { command: installed, argumentsPrefix: [], source: "installed" };
  }

  const target = targetFor(dependencies.platform, dependencies.architecture);
  let releaseFailure = null;
  if (target) {
    try {
      const binary = await cachedRelease(version, target, dependencies);
      return { command: binary, argumentsPrefix: [], source: "release" };
    } catch (error) {
      releaseFailure = error instanceof Error ? error.message : String(error);
    }
  }

  const uvx = executableOnPath("uvx", dependencies);
  if (uvx) {
    return {
      command: uvx,
      argumentsPrefix: ["--from", `betterborg==${version}`, "borg"],
      source: "uvx",
    };
  }

  const targetDescription = target
    ? `could not install the ${target} release${releaseFailure ? ` (${releaseFailure})` : ""}`
    : `no standalone release supports ${dependencies.platform}/${dependencies.architecture}`;
  throw new Error(
    `${targetDescription}, and uvx is not on PATH. Install uv from https://docs.astral.sh/uv/ or install betterborg==${version} so borg is on PATH.`,
  );
}

function launch(resolved, arguments_, overrides = {}) {
  const dependencies = withDefaults(overrides);
  return new Promise((resolve, reject) => {
    let settled = false;
    let child;
    try {
      child = dependencies.spawn(
        resolved.command,
        [...resolved.argumentsPrefix, ...arguments_],
        { stdio: "inherit" },
      );
    } catch (error) {
      reject(error);
      return;
    }

    const handlers = new Map(
      FORWARDED_SIGNALS.map((signal) => [
        signal,
        () => {
          child.kill(signal);
        },
      ]),
    );
    for (const [signal, handler] of handlers) {
      dependencies.process.on(signal, handler);
    }
    const cleanup = () => {
      for (const [signal, handler] of handlers) {
        dependencies.process.removeListener(signal, handler);
      }
    };

    child.once("error", (error) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(new Error(`could not start BetterBorg CLI: ${error.message}`));
    });
    child.once("close", (code, signal) => {
      if (settled) return;
      settled = true;
      cleanup();
      if (signal) {
        if (typeof dependencies.process.kill === "function") {
          dependencies.process.kill(dependencies.process.pid, signal);
        } else {
          dependencies.process.exitCode = SIGNAL_EXIT_CODES[signal] || 1;
        }
      } else {
        dependencies.process.exitCode = code === null ? 1 : code;
      }
      resolve();
    });
  });
}

async function main(arguments_, overrides = {}) {
  if (!overrides.version) {
    throw new Error("npm package version metadata is missing");
  }
  const resolved = await resolveCli(overrides.version, overrides);
  await launch(resolved, translateVersionArguments(arguments_), overrides);
}

function reportFailure(error, dependencies = {}) {
  const logger = dependencies.console || console;
  const processLike = dependencies.process || process;
  const message = error instanceof Error ? error.message : String(error);
  logger.error(`borg: ${message}`);
  processLike.exitCode = 1;
}

module.exports = {
  cachedRelease,
  launch,
  main,
  reportFailure,
  resolveCli,
  targetFor,
  translateVersionArguments,
  verifiedBinary,
};
