import asyncio
import json
import os
import shutil
import sys
import tempfile
from unittest.mock import MagicMock

# Mock modules BEFORE importing DatabaseManager
sys.modules["scienceai.llm"] = MagicMock()
# Create a mock for process_paper that is awaitable
mock_process_paper_module = MagicMock()


async def mock_process(pdf_path):
    return {"metadata": {"title": "Test Paper"}}


mock_process_paper_module.process_paper = mock_process
sys.modules["scienceai.process_paper"] = mock_process_paper_module

# Add src to path to ensure we load the local package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from scienceai.database_manager import DatabaseManager
from scienceai.process_paper import process_paper


async def main():
    print("Starting verification...")

    # Setup test environment
    test_dir = os.path.join(tempfile.gettempdir(), "test_ingestion")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    if not os.path.exists("test_paper.pdf"):
        print("test_paper.pdf not found.")
        return

    shutil.copy("test_paper.pdf", os.path.join(test_dir, "test_paper.pdf"))

    # Initialize DatabaseManager
    db = DatabaseManager(test_dir, process_paper, "test_project", storage_path=test_dir)

    # Ingest papers
    print("Ingesting papers...")
    db.ingest_papers()

    # Process papers
    print("Processing papers...")
    await db.process_all_papers()

    # Verify results
    papers = db.get_all_papers()
    for paper in papers:
        paper_id = paper["paper_id"]
        json_path = db.get_paper_json(paper_id)
        if json_path and os.path.exists(json_path):
            print(f"Paper {paper_id}: SUCCESS")
            with open(json_path) as f:
                data = json.load(f)
                print("METADATA EXTRACTED:")
                print(json.dumps(data.get("metadata", {}), indent=2))
        else:
            print(f"Paper {paper_id}: FAILED (No JSON found)")


if __name__ == "__main__":
    asyncio.run(main())
