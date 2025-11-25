import asyncio
import os
import shutil
import sys
import json

# Add src to path to ensure we load the local package
sys.path.insert(0, os.path.abspath("src"))

from scienceai.database_manager import DatabaseManager
from scienceai.process_paper import process_paper

async def main():
    print("Starting verification...")
    
    # Setup test environment
    test_dir = "test_ingestion"
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
        paper_id = paper['paper_id']
        json_path = db.get_paper_json(paper_id)
        if json_path and os.path.exists(json_path):
            print(f"Paper {paper_id}: SUCCESS")
            with open(json_path, 'r') as f:
                data = json.load(f)
                print("METADATA EXTRACTED:")
                print(json.dumps(data.get("metadata", {}), indent=2))
        else:
            print(f"Paper {paper_id}: FAILED (No JSON found)")

if __name__ == "__main__":
    asyncio.run(main())
