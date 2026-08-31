---
name: orchestrate
description: Use Betterborg to analyze, plan, and execute substantial repository work
---

# Orchestrate with Betterborg

Use the Borg MCP server for repository work that benefits from a durable PRD,
reviewed plan, executable task graph, or supervised execution.

1. Use `init` or `analyze` to establish the current repository state.
2. Use `create` to turn the user's goal into a named Borg and confirmed PRD.
3. Use `plan` for planning questions, plan inspection, and explicit approval.
4. Use `task_list` to inspect the approved executable work.
5. Use `execute` only after the user accepts the presented execution decision.

Treat structured statuses, artifacts, and next actions as authoritative. Ask the
user for any elicited input or approval; never manufacture consent. Report
blocked or setup-required results with their exact actionable guidance.
