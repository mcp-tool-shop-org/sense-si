# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | Yes       |

## Reporting a Vulnerability

Email: **64996768+mcp-tool-shop@users.noreply.github.com**

Include:
- Description of the vulnerability
- Steps to reproduce
- Version affected
- Potential impact

### Response timeline

| Action | Target |
|--------|--------|
| Acknowledge report | 48 hours |
| Assess severity | 7 days |
| Release fix | 30 days |

## Scope

This tool operates **locally only**.
- **Data touched:** Local image files (read-only), HuggingFace model cache (read/write on first download)
- **No network egress** at runtime — model downloads happen once on first use via HuggingFace Hub, then all inference is local
- **No secrets handling** — does not read, store, or transmit credentials
- **No telemetry** is collected or sent
- **No file mutation** — image files are opened read-only, never modified
