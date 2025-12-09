import asyncio
import os
import shutil
import sys
import tempfile
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# Mock modules BEFORE importing PrincipalInvestigator (or other scienceai modules that might depend on them)
sys.modules["scienceai.llm"] = MagicMock()
sys.modules["scienceai.analyst"] = MagicMock()
sys.modules["scienceai.reasoning"] = MagicMock()

from scienceai.database_manager import DatabaseManager


# Mock processor that always fails
async def mock_fail_processor(pdf_path):
    raise ValueError("Simulated metadata failure")


async def main():
    print("Starting error handling verification...")

    test_dir = os.path.join(tempfile.gettempdir(), "test_error_handling")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    # Create a dummy PDF
    with open(os.path.join(test_dir, "fail.pdf"), "wb") as f:
        f.write(b"%PDF-1.4 dummy content")

    # Initialize DatabaseManager with failing processor
    db = DatabaseManager(test_dir, mock_fail_processor, "test_project", storage_path=test_dir)

    db.ingest_papers()

    await db.process_all_papers()

    # Check results
    papers = db.get_all_papers()
    for paper in papers:
        print(f"Paper ID: {paper['paper_id']}")
        print(f"Status: {paper.get('status')}")
        print(f"Error: {paper.get('error')}")

        if paper.get("status") == "failed" and "Simulated metadata failure" in paper.get("error"):
            print("SUCCESS: Error correctly recorded.")
        else:
            print("FAILED: Error not recorded correctly.")


if __name__ == "__main__":
    asyncio.run(main())
