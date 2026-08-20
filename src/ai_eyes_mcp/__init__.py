"""ai-eyes-mcp — Grounded visual evaluator MCP server."""

__version__ = "1.1.0"

# Single source for the shipped MCP tool set. CI, verify.sh, and tests import
# this rather than each maintaining a frozenset literal (W1-CITOOL-001).
EXPECTED_TOOL_NAMES = frozenset({
    "image_contains",
    "image_classify",
    "image_compare",
    "image_score_batch",
    "image_verify",
    "eyes_selftest",
    "eyes_status",
})
