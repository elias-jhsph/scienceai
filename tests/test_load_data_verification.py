import os
import shutil
import sys
import tempfile
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# Mock modules BEFORE importing PrincipalInvestigator
sys.modules["scienceai.llm"] = MagicMock()
sys.modules["scienceai.analyst"] = MagicMock()
sys.modules["scienceai.reasoning"] = MagicMock()

from scienceai.principal_investigator import PrincipalInvestigator

# Setup paths
PROJECT_PATH = os.path.join(tempfile.gettempdir(), "scienceai_test_project_load")
if os.path.exists(PROJECT_PATH):
    shutil.rmtree(PROJECT_PATH)
os.makedirs(PROJECT_PATH)

DUMMY_CSV_PATH = os.path.join(PROJECT_PATH, "dummy_analyst_data.csv")
with open(DUMMY_CSV_PATH, "w") as f:
    f.write("col1,col2\n1,2\n3,4")


# Mock DatabaseManager
class MockDB:
    def __init__(self):
        self.project_path = PROJECT_PATH

    def get_all_analysts(self):
        return []

    def remove_old_default_messages(self, defaults):
        pass

    def get_pi_arbitrary_csv(self, name):
        return f"pi_arbitrary_csv/{name}"

    def create_pi_arbitrary_csv(self, name, content):
        pass

    def get_project_setting(self, key, default=None):
        return default

    def convert_analyst_tool_tracker(self, analyst_name, collection_name):
        # Mock returning the dummy CSV path
        if analyst_name == "Test Analyst" and collection_name == "test_collection":
            return DUMMY_CSV_PATH
        return None


def main():
    # Initialize PI with Mock DB
    mock_db = MockDB()
    pi = PrincipalInvestigator(mock_db)  # type: ignore

    # Test: Load Analyst Data
    print("Test: Load Analyst Data")
    code_load = """
import pandas as pd
filename = load_analyst_data('Test Analyst', 'test_collection')
print(f"Loaded filename: {filename}")
if filename:
    df = pd.read_csv(filename)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
"""

    result = pi.run_python_code(code_load)
    print(f"Result:\n{result}")

    # Assertions
    assert "Loaded filename: dummy_analyst_data.csv" in result
    assert "Data shape: (2, 2)" in result
    assert "Columns: ['col1', 'col2']" in result
    assert "Loaded data to dummy_analyst_data.csv" in result

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
