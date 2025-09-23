# Real-Time Market Sentiment Analyzer using LangChain & MLflow

This project implements a LangChain-powered pipeline that analyzes real-time market sentiment for a given company. It fetches the company's stock code, retrieves the latest news, and uses Google's Gemini LLM via Vertex AI to generate a structured JSON sentiment profile. All steps are traced and logged using MLflow for observability.

## Tech Stack & Tools
- **Framework**: LangChain
- **LLM**: Google Gemini 2.0 Flash (via Vertex AI)
- **Data Source**: Yahoo Finance News
- **Observability**: MLflow
- **Environment**: Python 3.10+

## Features
- **Dynamic Stock Code Extraction**: Automatically finds the stock ticker for any company name.
- **Real-Time News Fetching**: Uses the `YahooFinanceNewsTool` to get the latest market news.
- **Structured JSON Output**: Leverages LangChain's `PydanticOutputParser` to ensure a consistent and well-structured JSON response.
- **Deep Tracing with MLflow**: Integrates `MlflowCallbackHandler` to automatically trace each step of the chain, including LLM calls and tool usage, providing a clear view of the pipeline's execution.
- **Comprehensive Logging**: Logs all inputs, outputs, metrics (like confidence score), and metadata to MLflow for monitoring and debugging.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
Create a `requirements.txt` file with the following content:
```
langchain
langchain-google-vertexai
langchain-community
mlflow
pydantic<2
google-cloud-aiplatform
yfinance
```
Then, install the packages:
```bash
pip install -r requirements.txt
```
*Note: `pydantic<2` is specified as some LangChain components have better compatibility with v1.*

---

## Configuration

### 1. Google Cloud Authentication
This script requires access to Google Cloud Vertex AI. Ensure you have the `gcloud` CLI installed and configured.

Authenticate your local environment by running:
```bash
gcloud auth application-default login
```
This command will open a browser window for you to log in to your Google account and grant permissions.

### 2. Set Environment Variables
You must set the following environment variables. You can add them to your shell profile (e.g., `.bashrc`, `.zshrc`) or set them in your terminal session.

```bash
# Your Google Cloud Project ID
export VERTEX_AI_PROJECT="your-gcp-project-id"

# The region for your Vertex AI services (optional, defaults to us-central1)
export VERTEX_AI_LOCATION="us-central1"
```

---

## How to Run the Application

### 1. Start the MLflow UI
To monitor the runs and view the traces, start the MLflow UI in a separate terminal. The UI will read from the `mlflow.db` file that the script creates.

```bash
# Run this from the project's root directory
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Navigate to `http://127.0.0.1:5000` in your web browser to view the MLflow dashboard.

### 2. Run the Sentiment Analyzer Script
Execute the Python script from your terminal, passing the company name as a command-line argument.

**Example:**
```bash
python sentiment_analyzer.py "Google"
```

**Another Example:**
```bash
python sentiment_analyzer.py "NVIDIA Corporation"
```

After the script finishes, refresh the MLflow UI. You will see a new run under the "Market Sentiment Analysis" experiment. Click on it to see the logged parameters, metrics, the output JSON artifact, and a detailed trace of the LangChain execution under the "Traces" tab.

!

---
