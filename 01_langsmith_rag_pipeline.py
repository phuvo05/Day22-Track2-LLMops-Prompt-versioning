"""
Step 1 — LangSmith-instrumented RAG Pipeline
=============================================
TASK:
  1. Load knowledge base, split into chunks, index with FAISS
  2. Build a RAG chain: retriever -> prompt -> LLM -> output parser
  3. Decorate the query function with @traceable so every call is traced
  4. Run all 50 questions -> generates >= 50 LangSmith traces

DELIVERABLE: Open https://smith.langchain.com and confirm traces appear.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "ai20k-lab22")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

if not OPENAI_API_KEY or OPENAI_API_KEY == "<your-openai-api-key>":
    print("ERROR: OPENAI_API_KEY is not set in .env file.")
    print("Please set a valid OpenAI API key to run this pipeline.")
    sys.exit(1)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable


llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    temperature=0.0,
)

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)


def build_vectorstore():
    """
    Load the knowledge base, split into chunks, embed and index with FAISS.

    Steps:
      a) Read the knowledge base file
      b) Split text with RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
      c) Call FAISS.from_texts(chunks, embeddings) to build the index
      d) Return the vectorstore
    """
    kb_path = Path("data/knowledge_base.txt")
    if not kb_path.exists():
        print(f"ERROR: Knowledge base not found at {kb_path}")
        sys.exit(1)

    text = kb_path.read_text(encoding="utf-8")
    print(f"Loaded knowledge base ({len(text)} characters)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks")

    vectorstore = FAISS.from_texts(chunks, embeddings)
    print("FAISS vector store created")

    return vectorstore


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the context below to answer the question.\n\nContext:\n{context}"),
    ("human", "{question}"),
])


def build_rag_chain(vectorstore):
    """
    Build a LangChain RAG chain using LCEL (pipe operator).

    Chain structure:
        {"context": retriever | format_docs, "question": passthrough}
        | prompt
        | llm
        | StrOutputParser()

    Returns: (chain, retriever)
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever


@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """
    Run the RAG chain on a single question.
    The @traceable decorator sends input/output/latency to LangSmith.
    """
    return chain.invoke(question)


SAMPLE_QUESTIONS = [
    "What are the three main types of machine learning?",
    "What is overfitting in machine learning?",
    "Explain the bias-variance tradeoff.",
    "How does regularization prevent overfitting?",
    "What is cross-validation?",
    "What is backpropagation?",
    "What are Convolutional Neural Networks primarily used for?",
    "How do LSTM networks address the vanishing gradient problem?",
    "What activation functions are commonly used in neural networks?",
    "What is the role of pooling layers in CNNs?",
    "What is the transformer architecture?",
    "What are word embeddings?",
    "What is transfer learning in NLP?",
    "How does BERT handle language understanding?",
    "What is self-attention in transformers?",
    "What is GPT and how is it trained?",
    "What is instruction tuning?",
    "What is RLHF?",
    "What is chain-of-thought prompting?",
    "What is the context length of GPT-4?",
    "What is Retrieval-Augmented Generation?",
    "What are the main components of a RAG pipeline?",
    "What is dense retrieval?",
    "Why is chunking strategy important in RAG?",
    "What advanced RAG techniques exist beyond basic retrieval?",
    "What are vector databases used for?",
    "What is FAISS?",
    "How do text embeddings capture semantic meaning?",
    "What is HNSW?",
    "What is hybrid search in vector databases?",
    "What is LangChain?",
    "What is LangChain Expression Language (LCEL)?",
    "What is LangGraph?",
    "What memory types does LangChain support?",
    "What are LangChain retrievers?",
    "What is LangSmith?",
    "What information do LangSmith traces capture?",
    "What is the LangSmith Prompt Hub?",
    "How does LangSmith help monitor production LLM applications?",
    "What are LangSmith datasets used for?",
    "What is RAGAS?",
    "How does RAGAS compute faithfulness?",
    "What is answer relevancy in RAGAS?",
    "What is context recall in RAGAS?",
    "What inputs does RAGAS evaluation require?",
    "What is Guardrails AI?",
    "What is PII and why is it important to detect in LLM responses?",
    "What does structured output validation ensure?",
    "What is Constitutional AI?",
    "What are common AI safety concerns with LLMs?",
]


def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)
    print(f"  LangSmith Project: {os.environ['LANGCHAIN_PROJECT']}")
    print(f"  LLM Model: {LLM_MODEL}")
    print(f"  Embedding Model: {EMBEDDING_MODEL}")
    print("=" * 60)
    print()

    print("[1/3] Building vector store...")
    vectorstore = build_vectorstore()
    print()

    print("[2/3] Building RAG chain...")
    chain, retriever = build_rag_chain(vectorstore)
    print()

    print("[3/3] Running questions through RAG pipeline...")
    print("-" * 60)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        answer = ask(chain, question)
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:60]}{'...' if len(question) > 60 else ''}")
        print(f"       A: {answer[:150]}{'...' if len(answer) > 150 else ''}")
        print()

    print("=" * 60)
    print(f"Successfully processed {len(SAMPLE_QUESTIONS)} questions")
    print(f"All traces sent to LangSmith project '{os.environ['LANGCHAIN_PROJECT']}'")
    print("Open https://smith.langchain.com to view traces.")
    print("=" * 60)


if __name__ == "__main__":
    main()
