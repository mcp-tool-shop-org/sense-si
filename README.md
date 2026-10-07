# sense-si

sense-si is the Python home for the seeing and hearing instruments. It is not a Rust port of ai-eyes. A Rust port would be a different project.

Two instruments, and one juror that can read what they write:

- **Eyes** (`ai-eyes`) measures a claim about pixels. The package is in `packages/eyes`, named `ai-eyes`, matching `ai-ears`. The import is `ai_eyes_mcp`. The console script is `ai-eyes`. The payloads stay as they are.
- **Ears** (`ai-ears`) turns a take into a hearing record: timing, pitch, a listener's transcript, and optional review marks. Every number names the instrument that produced it.
- **Decisions** answers a typed question over a record and returns probabilities. The local engine is Kev-4B, pinned by revision. Hosted Jev stays an optional comparison. Decisions does not hear, does not see, and does not pass or fail a take.

Jam keeps its own gates. A probability is something a person can read next to those gates. It is not a gate.

OpenRouter is not enabled here. The client exists so a test can fake the transport. A live call waits on a separate yes.

## Layout

```
packages/decisions/     local Kev engine and optional pinned Jev client
packages/ears/          hearing record and the two questions
packages/eyes/          seeing instrument, SigLIP2
docs/contract.md        what the packages are allowed to decide
docs/adapter.md         how a jam receipt becomes a record
```

Install is not required to run the tests. From this directory:

```bash
python -m pytest
```

The import for the juror is `decisions`. Ears imports it. Eyes, when it arrives, imports neither.
