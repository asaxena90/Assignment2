# Real-Time Market Sentiment Analyzer using LangChain & MLflow

This project implements a LangChain-powered pipeline that analyzes real-time market sentiment for a given company. It fetches the company's stock code, retrieves the latest news, and uses Google's Gemini LLM via the Generative AI API to generate a structured JSON sentiment profile. All steps are traced and logged using MLflow for observability.

## Tech Stack & Tools
- **Framework**: LangChain
- **LLM**: Google Gemini 2.0 Flash 
- **Data Source**: Yahoo Finance News
- **Observability**: MLflow
- **Environment**: Python 3.10+

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  
```

### 3. Install Dependencies
Create a `requirements.txt` file with the following content. Note the change from `langchain-google-vertexai` to `langchain-google-genai`.

```
langchain
langchain-google-genai
langchain-community
mlflow
pydantic<2
yfinance
```
Then, install the packages:
```bash
pip install -r requirements.txt
```

---

## Configuration



### 1. Set Environment Variables
You must set the `GOOGLE_API_KEY` environment variable. You can add it to your shell profile (e.g., `.bashrc`, `.zshrc`) or set it in your terminal session for testing.

```bash
# Your Google API Key from AI Studio
export GOOGLE_API_KEY="AIzaSy...your...key...here"
```

---

## How to Run the Application

### 1. Start the MLflow UI
To monitor the runs, start the MLflow UI in a separate terminal.

```bash
# Run this from the project's root directory
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Navigate to `http://1227.0.0.1:5000` in your web browser.

### 2. Run the Sentiment Analyzer Script
Execute the Python script, passing the company name as an argument.

**Example:**
```bash
python sentiment_analyzer.py "Microsoft"
```

After the script finishes, refresh the MLflow UI to see the new run, including its parameters, metrics, and detailed traces.
