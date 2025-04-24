import os
import getpass
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

FOUNDATION_MODEL = "gemini-2.0-flash-lite"

def get_api_key():
    load_env()

    return os.environ["GOOGLE_API_KEY"]

def load_env():
    _ = load_dotenv()

    if "GOOGLE_API_KEY" not in os.environ:                       
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google Gemini API key: ")

def get_chat_model(temperature=1.0, top_k=1, top_p=0.9):
    load_env()

    return ChatGoogleGenerativeAI(
        max_tokens=None,
        model=FOUNDATION_MODEL,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )

def get_embeddings_model():
    load_env()

    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")