import os
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal, Annotated
import getpass
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def load_env():
    _ = load_dotenv()

    if "GOOGLE_API_KEY" not in os.environ:                       
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google Gemini API key: ")

def get_foundation_model():
    load_env()

    return ChatGoogleGenerativeAI(
        max_tokens=None,
        model="gemini-2.0-flash-exp",
        temperature=1,
        top_k=1,
        top_p=0.9
    )


class Router(BaseModel):
    """Analyze the unread email and route it according to its content."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )


def get_router_model(base_model):
    return base_model.with_structured_output(Router)

def get_embeddings_model():
    load_env()

    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")