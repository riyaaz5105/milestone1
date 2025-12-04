# ===== WEEK-1 =====
# basic_gemini_llm.py 
import os
from dotenv import load_dotenv
import google.generativeai as genai

# 1. Load environment variables from .env
load_dotenv()

# 2. Read API key from environment and configure Gemini
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment. Check your .env file.")
genai.configure(api_key=GEMINI_KEY)

# 3. Create a Gemini model instance
model = genai.GenerativeModel("gemini-2.5-flash")  

# 4. Ask a question
prompt = "You are a helpful AI assistant. Explain machine learning in simple words."

# 5. Get the response
response = model.generate_content(prompt)

# 6. Print the response text
print("=== GEMINI SDK OUTPUT ===")
print(response.text)
print("=========================")