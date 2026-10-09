import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("API key not found. Check your .env file.")
else:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            google_api_key=api_key,
            temperature=0,
        )

        response = llm.invoke(
            "Explain artificial intelligence in 3 simple sentences."
        )

        print("\nGemini API connected successfully!")
        print("\nGemini response:")
        content = response.content
        if isinstance(content, list):
            content = "\n".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        print(content)

    except Exception as error:
        print("\nGemini connection failed.")
        print("Error:", error)
