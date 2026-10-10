import os
import re
from typing import TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


# Step 1: Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "Gemini API key not found. Check your .env file."
    )


# Step 2: Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key,
    temperature=0,
)


# Step 3: Define workflow state
class AssistantState(TypedDict):
    question: str
    context: str
    answer: str
    chat_history: list
    chunks: list


# Step 4: Load and split PDF
def load_pdf_chunks(pdf_path="data/sample.pdf"):

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    documents = PyPDFLoader(pdf_path).load()

    print(f"PDF pages loaded: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(documents)

    print(f"Text chunks created: {len(chunks)}")

    return chunks


chunks = load_pdf_chunks()
gemini_quota_exhausted = False

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "did",
    "do", "does", "for", "from", "give", "has", "have", "i", "in",
    "is", "it", "me", "of", "on", "or", "please", "show", "tell",
    "that", "the", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "why", "with",
}


def search_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOP_WORDS
    }


# Step 5: Retrieve relevant PDF content
def retrieve_context(state: AssistantState):
    question_terms = search_terms(state["question"])
    if "phone" in question_terms or "mobile" in question_terms:
        question_terms.add("contact")

    scored_chunks = []

    for document in state["chunks"]:
        content = document.page_content
        words = search_terms(content)
        score = len(question_terms & words)
        if state["question"].lower().strip(" ?!.,") in content.lower():
            score += 2

        if score > 0:
            scored_chunks.append(
                (score, content)
            )

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    top_chunks = scored_chunks[:3]

    context = "\n\n".join(
        text for score, text in top_chunks
    )

    if not context:
        context = ""

    print("PDF retrieval completed.")

    return {"context": context}


# Step 6: Generate an answer with Gemini
def generate_answer(state: AssistantState):
    global gemini_quota_exhausted

    current_chunks = state.get("chunks", [])
    question = state["question"]
    context = state["context"]
    chat_history = state.get("chat_history", [])

    if not context:
        return {"answer": "Information not found in the document."}

    if gemini_quota_exhausted:
        return {
            "answer": (
                "Gemini's request quota is exhausted. "
                "Here are the matching PDF passages:\n\n"
                + context
            )
        }

    previous_messages = "\n".join(
        f"User: {item['question']}\n"
        f"Assistant: {item['answer']}"
        for item in chat_history[-4:]
    )

    prompt = f"""
You are a helpful AI document assistant.

Answer the question using the PDF context below.
Give a clear, detailed answer when the context supports it.
Do not invent facts that are not in the PDF.

If the answer is not present, say:
"Information not found in the document."

Previous conversation:
{previous_messages}

PDF context:
{context}

Current question:
{question}
"""

    try:
        response = llm.invoke(prompt)
    except Exception as error:
        error_message = str(error)
        if "RESOURCE_EXHAUSTED" not in error_message and "429" not in error_message:
            raise
        gemini_quota_exhausted = True
        return {
            "answer": (
                "Gemini's request quota is exhausted. "
                "Here are the matching PDF passages:\n\n"
                + context
            )
        }

    content = response.content

    if isinstance(content, list):
        content = "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )

    return {"answer": str(content)}


# Step 7: Build the LangGraph workflow
builder = StateGraph(AssistantState)

builder.add_node("retrieve_context", retrieve_context)
builder.add_node("generate_answer", generate_answer)

builder.add_edge(START, "retrieve_context")
builder.add_edge("retrieve_context", "generate_answer")
builder.add_edge("generate_answer", END)

workflow = builder.compile()


# Step 8: Run the assistant
if __name__ == "__main__":
    chat_history = []

    print("\nGemini AI Document Assistant")
    print("Ask questions about your PDF.")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            result = workflow.invoke({
                "question": question,
                "context": "",
                "answer": "",
                "chat_history": chat_history,
                "chunks": chunks,
            })

            answer = result["answer"]

            print("\nGemini Answer:")
            print(answer)
            print()

            chat_history.append({
                "question": question,
                "answer": answer,
            })

        except Exception as error:
            print("\nSomething went wrong:")
            print(error)