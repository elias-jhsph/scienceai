"""
ScienceAI - AI-Powered Research Assistant for Systematic Literature Analysis.

ScienceAI is a Python application that transforms how researchers analyze
scientific literature through an intelligent agent-based architecture.

Basic Usage:
    >>> from scienceai import ScienceAI
    >>> client = ScienceAI(project_name="MyProject")
    >>> client.upload_papers(["/path/to/paper.pdf"])
    >>> response = client.chat("What are the key findings?")

For more information, see the documentation at:
https://github.com/elias-jhsph/scienceai
"""

from scienceai.client import ScienceAI
from scienceai.database_manager import DatabaseManager, get_projects

__version__ = "0.4.3"
__author__ = "Elias Weston-Farber"
__email__ = "elias@eliastechlabs.com"

__all__ = [
    # Database utilities
    "DatabaseManager",
    # Main client interface
    "ScienceAI",
    "__author__",
    "__email__",
    # Package metadata
    "__version__",
    "get_projects",
]
