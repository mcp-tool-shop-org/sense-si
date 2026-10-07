# sense-si

sense-si is the Python home for the seeing and hearing instruments. It is not a Rust port of ai-eyes. A Rust port would be a different project.

Two instruments, and one juror that can read what they write:

- **Eyes** (`ai-eyes`) measures a claim about pixels. The package is named `ai-eyes`, matching `ai-ears`. It is not in this checkout yet. The repository that holds the code today is `ai-eyes-mcp` at v1.2.0, which was not on PyPI, and both `ai-eyes` and `ai-ears` were free there on 2026-10-07. When the package moves here, the primary console script is `ai-eyes`. `ai-eyes-mcp` stays as a second entry point only until the Claude Code MCP config is switched to `ai-eyes`, and is removed in a later pull request. The import and the payloads stay as they are.
- **Ears** (`ai-ears`) turns a take into a hearing record: timing, pitch, a listener's transcript, and optional review marks. Every number names the instrument that produced it.
- **Decisions** is TypeSafe Jev, pinned. It answers a typed question over a record and returns probabilities. It does not hear, it does not see, and it does not pass or fail a take.

Jam keeps its own gates. A probability is something a person can read next to those gates. It is not a gate.

OpenRouter is not enabled here. The client exists so a test can fake the transport. A live call waits on a separate yes.

## Layout

```
packages/decisions/     pinned Jev client
packages/ears/          hearing record and the two questions
docs/contract.md        what the packages are allowed to decide
docs/adapter.md         how a jam receipt becomes a record
```

Install is not required to run the tests. From this directory:

```bash
python -m pytest
```

The import for the juror is `decisions`. Ears imports it. Eyes, when it arrives, imports neither.
