ScienceAI
=========

ScienceAI is a Python package designed to act as an AI-powered scientific literature search engine. It leverages the power of large language models (LLMs) to process and analyze research papers, enabling users to ask complex questions and receive insightful answers supported by evidence from the literature. The application can handle hundreds of papers included in the analysis without needing you to include any metadata for the uploaded papers, just provide the files!

Main Features
-------------

*   **Automated Paper Processing:** Automatically extract text, figures, tables, and metadata from research papers (PDFs).
*   **AI-Driven Analysis:** Utilize LLMs to summarize papers, interpret figures and tables, and extract relevant data points based on user-defined schemas.
*   **Analyst Agents:** Created and managed by the top-level AI to address specific research goals autonomously.
*   **Interactive Discussion:** Engage in a conversational interface with ScienceAI, asking questions and receiving detailed responses.
*   **Data Management:** Robust database system for efficient data retrieval and management.
*   **Visualization and Export:** Interactive, tree-like structure for exploring analysis results, with options to download extracted data, analysis summaries, and individual papers.
*   **Export Capabilities:** Export the data extracted by the AI in CSV format, and export all or subsets of the papers with meaningful file names and metadata detected by the system.

Installation
------------

ScienceAI requires Python 3.11 or higher and an OpenAI API key. To install the package, you can use pip:

```bash
pip install scienceai-llm
```

Usage
-----

ScienceAI is designed to be used through its user interface. After installation, start the application and use the web interface to upload PDF files and manage projects.

1.  Start the ScienceAI application:
    
    ```bash
    scienceai
    ```
    
2.  Open your web browser and navigate to `http://localhost:4242`.
    
3.  **Create a New Project:**
    
    *   Enter a project name and click "Start".
    *   Upload individual PDFs or a zip folder of PDFs for analysis.
    *   Click "Create Project" to begin the analysis.
4.  **Analyze Papers:**
    
    *   View the list of uploaded papers with their metadata.
    *   Use the "Science Discussion" panel to interact with the analysis framework.
    *   View and download extracted data in JSON and CSV formats.

Example Use Case
----------------

An example use case for ScienceAI is performing ad hoc literature reviews. A researcher can direct the AI to extract data from hundreds of papers simultaneously, which would be cumbersome using a simple chat interface. The researcher can upload a large set of PDFs, specify the data to be extracted, and let the AI handle the complex analysis. The extracted data can then be exported in CSV format for further investigation.

Detailed Documentation
----------------------

### Database Management

The `database_manager` module is the backbone of ScienceAI's data handling. It's responsible for:

*   **Ingesting Papers:** Adding research papers (PDFs) to the database.
*   **Processing Papers:** Extracting text, figures, tables, metadata, and generating summaries.
*   **Storing Data:** Persisting processed information in a structured format.
*   **Retrieving Data:** Providing access to papers, data extractions, and analysis results.
*   **Managing Analyst Agents:** Creating, storing, and retrieving Analyst Agent data.

### Principal Investigator (PI)

The `principle_investigator` module represents the main AI persona you interact with. The PI:

*   **Delegates Research:** Creates and manages Analyst Agents to address specific research questions.
*   **Interacts with User:** Conducts the conversation with the user, understanding their goals and relaying information from the Analyst Agents.
*   **Oversees Analysis:** Monitors the progress of Analyst Agents and ensures the research process is effective.

### Analyst Agents

The `analyst` module defines Analyst Agents, which are created by the top-level AI. Each Analyst:

*   **Has a Goal:** Is assigned a specific research question or objective by the top-level AI.
*   **Requests Data:** Directs the creation of data extraction schemas to gather relevant information from the papers.
*   **Analyzes Data:** Processes the extracted data to form answers to their assigned goal.
*   **Provides Evidence:** Presents the answer to their goal with supporting evidence from the research papers.

### Data Extraction

The `data_extractor` module handles the process of extracting structured data from papers based on user-defined schemas:

*   **Data Types:** Offers various data types (number, date, text\_block, etc.) for flexible data extraction.
*   **Schema Generation:** Assists in generating a schema that outlines the data to be extracted.
*   **Data Extraction:** Uses LLMs to extract the specified data from research papers.

### Language Model Interaction

The `llm` module manages interactions with the OpenAI API:

*   **API Calls:** Handles requests to the OpenAI API for tasks such as text generation and function calling.
*   **Token Management:** Tracks token usage to stay within API limits.
*   **Error Handling:** Provides error handling for API requests.

Configuring the OpenAI API Key
------------------------------

ScienceAI requires an OpenAI API key to interact with the OpenAI language models. Follow these steps to configure your API key:

1.  **Obtain an API Key:** Sign up for an API key from OpenAI if you don't already have one.
2.  **Enter the API Key:** The first time you run ScienceAI, you will be prompted to enter your OpenAI API key. Paste your API key into the prompt.

Contributing
------------

We welcome contributions to ScienceAI! Here's how you can get involved:

*   **Report Bugs:** If you find any issues or bugs, please open an issue on our GitHub repository.
*   **Feature Requests:** Have an idea for a new feature? Submit a feature request on GitHub.
*   **Pull Requests:** Want to contribute code? Fork the repository, make your changes, and submit a pull request.
