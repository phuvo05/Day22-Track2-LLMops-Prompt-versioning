"""
Step 2 — Prompt Hub & A/B Routing
===================================
TASK:
  1. Write two distinct system prompts (V1: concise, V2: structured)
  2. Push both to LangSmith Prompt Hub via client.push_prompt()
  3. Pull them back via client.pull_prompt()
  4. Implement deterministic A/B routing: hash(request_id) % 2 → V1 or V2
  5. Run all 50 questions through the router → ≥ 50 more LangSmith traces

DELIVERABLE: 2 named prompts visible in https://smith.langchain.com Prompt Hub
"""

import os
import sys
import hashlib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "ai20k-lab22")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

PROMPT_V1_NAME = "ai20k-rag-prompt-v1"
PROMPT_V2_NAME = "ai20k-rag-prompt-v2"

SYSTEM_V1 = (
    "You are a helpful AI assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer concise (2-4 sentences). "
    "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)

SYSTEM_V2 = (
    "You are an expert AI tutor. Answer using ONLY the provided context below.\n"
    "IMPORTANT: Do NOT add any information not present in the context. "
    "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Instructions:\n"
    "1. Read the context carefully and identify only the facts directly relevant to the question.\n"
    "2. Synthesize those facts into a clear, accurate, well-organized answer (3-5 sentences).\n"
    "3. Avoid speculation, external knowledge, or elaboration beyond what the context states.\n"
    "4. If the context lacks sufficient information, respond: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human", "{question}"),
])

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human", "{question}"),
])

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


def push_prompts_to_hub(client: Client) -> None:
    """Upload both prompt versions to LangSmith Prompt Hub."""
    try:
        url = client.push_prompt(
            PROMPT_V1_NAME,
            object=PROMPT_V1,
            description="V1 - concise 2-4 sentence answers"
        )
        print(f"[OK] Pushed V1 -> {url}")
    except Exception as e:
        print(f"[WARNING] V1 push failed: {e}")

    try:
        url = client.push_prompt(
            PROMPT_V2_NAME,
            object=PROMPT_V2,
            description="V2 - structured expert 3-5 sentence answers"
        )
        print(f"[OK] Pushed V2 -> {url}")
    except Exception as e:
        print(f"[WARNING] V2 push failed: {e}")


def pull_prompts_from_hub(client: Client) -> dict:
    """Download both prompt versions from LangSmith Prompt Hub. Fall back to local templates if unavailable."""
    prompts = {}

    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"[DOWN] Pulled '{PROMPT_V1_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"[INFO] Using local fallback for '{PROMPT_V1_NAME}'")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"[DOWN] Pulled '{PROMPT_V2_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"[INFO] Using local fallback for '{PROMPT_V2_NAME}'")

    return prompts


def get_prompt_version(request_id: str) -> str:
    """Route a request to prompt V1 or V2 based on the MD5 hash of request_id."""
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


def build_vectorstore() -> FAISS:
    """Load the knowledge base, split into chunks, embed and index with FAISS."""
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    text = Path("data/knowledge_base.txt").read_text()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks")

    vectorstore = FAISS.from_texts(chunks, embeddings)
    print(f"Indexed {len(chunks)} chunks in FAISS")
    return vectorstore


@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """Run the RAG chain using the given prompt version."""
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )

    return {"question": question, "answer": answer, "version": version}


def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    client = Client(api_key=os.environ.get("LANGCHAIN_API_KEY", ""))

    print("\n--- Pushing prompts to LangSmith Prompt Hub ---")
    push_prompts_to_hub(client)

    print("\n--- Pulling prompts from LangSmith Prompt Hub ---")
    prompts = pull_prompts_from_hub(client)

    print("\n--- Building vectorstore ---")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )
    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("\n--- Running A/B routing for all questions ---")

    v1_count = 0
    v2_count = 0

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        request_id = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt = prompts[version_key]

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1

        result = ask_ab(retriever, llm, prompt, question, version_tag)
        print(f"[{i:02d}] [prompt-{version_tag}] {question[:55]}...")

    print("\n" + "=" * 60)
    print("  Routing Summary")
    print("=" * 60)
    print(f"  Total questions: {len(SAMPLE_QUESTIONS)}")
    print(f"  Prompt V1 (concise): {v1_count} queries")
    print(f"  Prompt V2 (structured): {v2_count} queries")
    print(f"  Split ratio: {v1_count/len(SAMPLE_QUESTIONS)*100:.1f}% / {v2_count/len(SAMPLE_QUESTIONS)*100:.1f}%")
    print("=" * 60)
    print(f"\n[OK] {len(SAMPLE_QUESTIONS)} traces sent to LangSmith project '{os.environ['LANGCHAIN_PROJECT']}'")
    print("   Open https://smith.langchain.com to view traces.")


if __name__ == "__main__":
    main()
