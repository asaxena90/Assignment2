import os
import sys
import json
import logging
from typing import List, Optional

# LangChain Imports
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import YahooFinanceNewsTool
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough

# MLflow Integration
import mlflow
mlflow.langchain.autolog()

# --- Configuration ---
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Google API Key Configuration
if not os.getenv("GOOGLE_API_KEY"):
    sys.exit("Error: GOOGLE_API_KEY environment variable is not set.")

# MLflow Configuration
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Market Sentiment Analysis")

# --- Pydantic Model for Structured Output ---
class SentimentProfile(BaseModel):
    """A structured profile of market sentiment for a company."""
    company_name: str = Field(description="The name of the company being analyzed.")
    stock_code: str = Field(description="The stock market ticker symbol for the company.")
    sentiment: str = Field(description="Overall sentiment, must be 'Positive', 'Negative', or 'Neutral'.")
    market_implications: str = Field(description="A brief summary of the market implications based on the news.")
    confidence_score: float = Field(description="Confidence in the sentiment analysis, from 0.0 to 1.0.")
    people_names: Optional[List[str]] = Field(description="List of key people mentioned in the news.")
    places_names: Optional[List[str]] = Field(description="List of key geographical places mentioned.")
    other_companies_referred: Optional[List[str]] = Field(description="List of other companies mentioned.")
    related_industries: Optional[List[str]] = Field(description="List of related industries impacted.")
    news_summary: str = Field(description="A concise summary of the news articles provided.")


# --- Core Functions & Chain Links ---

def get_stock_ticker(company_name: str) -> str:
    """Uses an LLM to find the stock ticker for a given company name."""
    logging.info(f"Generating stock ticker for: {company_name}")
    ticker_prompt = PromptTemplate.from_template(
        "What is the stock market ticker for {company_name}? Respond with only the ticker symbol."
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, convert_system_message_to_human=True)
    ticker_chain = ticker_prompt | llm | StrOutputParser()
    ticker = ticker_chain.invoke({"company_name": company_name}).strip()
    logging.info(f"Found ticker: {ticker}")
    return ticker

def fetch_news(ticker: str) -> str:
    """Fetches recent news articles for a given stock ticker."""
    logging.info(f"Fetching news for ticker: {ticker}")
    try:
        tool = YahooFinanceNewsTool()
        news_results = tool.run(ticker)
        if isinstance(news_results, list) and all(isinstance(i, dict) for i in news_results):
            return "\n\n".join(
                f"Title: {item.get('title', 'N/A')}\n"
                f"Published: {item.get('published', 'N/A')}\n"
                f"Summary: {item.get('summary', 'No summary available.')}"
                for item in news_results[:5]
            )
        else:
            logging.warning(f"No structured news found for {ticker}. Returning raw output.")
            return str(news_results)
    except Exception as e:
        logging.error(f"Failed to fetch news for {ticker}: {e}")
        return "No news could be fetched."

# --- Main Analysis Pipeline ---

def build_sentiment_analyzer_chain():
    """Constructs the full LangChain pipeline for sentiment analysis."""
    # 1. Initialize the LLM using ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, convert_system_message_to_human=True)

    # 2. Set up the Pydantic parser
    parser = PydanticOutputParser(pydantic_object=SentimentProfile)

    # 3. Create the main analysis prompt template
    analysis_prompt = PromptTemplate(
        template="""You are an expert financial market analyst.
Analyze the following news articles for the company '{company_name}' with stock code '{stock_code}'.
Based on the news, generate a structured sentiment profile.

{format_instructions}

News Articles:
---
{news_desc}
---
""",
        input_variables=["company_name", "stock_code", "news_desc"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # 4. Define the complete chain using LangChain Expression Language (LCEL)
    chain = (
        {"company_name": RunnablePassthrough(), "stock_code": RunnablePassthrough(), "news_desc": RunnablePassthrough()}
        | RunnablePassthrough.assign(
            stock_code=lambda x: get_stock_ticker(x["company_name"])
        )
        | RunnablePassthrough.assign(
            news_desc=lambda x: fetch_news(x["stock_code"])
        )
        | analysis_prompt
        | llm
        | parser
    )
    return chain

# --- Execution ---

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sentiment_analyzer_api_key.py \"<Company Name>\"")
        sys.exit(1)

    company_input = sys.argv[1]

    # mlflow_callback_handler = MlflowCallbackHandler()

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logging.info(f"Starting MLflow Run ID: {run_id}")
        logging.info(f"To view logs, run: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")

        mlflow.log_param("company_name", company_input)

        try:
            analyzer_chain = build_sentiment_analyzer_chain()
            result = analyzer_chain.invoke(company_input)
            mlflow.log_metric("sentiment_confidence", result.confidence_score)
            mlflow.log_dict(result.dict(), "sentiment_profile.json")
            mlflow.set_tag("status", "SUCCESS")

            print("\n--- Market Sentiment Analysis Result ---")
            print(json.dumps(result.dict(), indent=2))
            print("\n--- End of Report ---")

        except Exception as e:
            logging.error(f"An error occurred during the analysis: {e}", exc_info=True)
            mlflow.set_tag("status", "FAILED")
            mlflow.log_param("error_message", str(e))
            sys.exit(1)
