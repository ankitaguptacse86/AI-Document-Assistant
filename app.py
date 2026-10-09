from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
# Step 1: Load the PDF
loader = PyPDFLoader("data/sample.pdf")
documents = loader.load()

print(f"PDF pages loaded: {len(documents)}")

# Step 2: Split the document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"Text chunks created: {len(chunks)}")

# Step 3: Display the first chunk
if chunks:
    print("\nFirst chunk:")
    print(chunks[0].page_content[:500])
else:
    print("No text found in the PDF.")
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    print("Gemini API key not found.")
    print("Add it to the .env file later.")

else:
    # Add Step 11.3 code here
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    print("FAISS vector database created successfully!")

    def retrieve_documents(question):
        results = vector_store.similarity_search(
            question,
            k=3
        )
        return results

    # Test document retrieval
    question = st.text_input("Ask a question about your PDF: ")
    results = retrieve_documents(question)

    print("\nRelevant document content:")

    for document in results:
        print(document.page_content)
        print("-" * 40)
    # Step 13: Generate an answer using Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0
    )

    def generate_answer(question, documents):
        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        prompt = f"""
        You are an AI document assistant.

        Answer the question using only the context below.
        If the answer is not present, say:
        "Information not found in the document."

        Context:
        {context}

        Question:
        {question}
        """

        response = llm.invoke(prompt)
        return response.content

    answer = generate_answer(question, results)
    st.write(answer)
    print("\nGemini Answer:")
    print(answer)
