# AGENTS.md

This file defines how agents should work in this repository.

## Core Preferences
- No backward compatibility unless explicitly requested.
- No fallbacks or silent failures. Fail fast with clear, explicit error messages.
- Prefer clean, direct solutions over workarounds.
- Do not add compatibility checks for deprecated parameters unless explicitly requested.
- Do not create commits or push. If asked for a commit message, use Conventional Commits.
- ROS 1 is used.
- Primary run path is Docker (docker compose).
- Image transport is raw-only; compressed transport is intentionally not supported.
- The system is interactive: laser detection is the user input method. Prefer solutions that balance low latency and accuracy.

## Workflow
- Before larger changes, present the plan and wait for approval.
- If the request is ambiguous, ask clarifying questions before acting.
- After any code change, review README and update it if needed.
- Keep docstrings accurate and updated with behavior changes.
- Avoid code duplication; refactor when duplication appears.
- Do not leave dead code; remove it or call it out explicitly.
- Point out possible optimizations when noticed.

## Error Handling
- Prefer explicit validation and early exits over implicit defaults.
- Do not ignore errors; surface them with actionable messages.
- Avoid "best effort" behavior that hides faults.

## Testing and Verification
- Prefer simple, deterministic checks that match current tooling.
- If tests cannot be run, state why and suggest what to run.

## Communication
- Explain what is changing and why.
- Flag risks and assumptions.
- If unsure about intent, stop and ask.
