"""Run the bundled Qdrant MCP server over stdio."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")
# Keep the model from guessing a collection when deployments omit this setting.
os.environ.setdefault("COLLECTION_NAME", "uda-hub-retrieval-eval")
os.environ.setdefault(
    "FASTEMBED_CACHE_PATH",
    str(PROJECT_ROOT / "mcp_services" / "qdrant" / ".cache" / "fastembed"),
)

from mcp_server_qdrant.server import mcp


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
