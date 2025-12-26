import os
import datetime
from typing import TypedDict, Annotated, List, Union

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# --- 1. Shared State & Long-Term Memory ---

class AgentState(TypedDict):
    """The 'Shared Scratchpad' for all agents."""
    task: str
    research_data: str
    final_output: str
    chat_history: List[BaseMessage]

# Initialize Vector Store as a "Shared Long-Term Memory"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# Create an empty vectorstore
shared_vector_db = FAISS.from_texts(["Initial System Start"], embeddings)

def save_to_memory(info: str):
    """Saves key findings into the shared vector store."""
    shared_vector_db.add_texts([info])

def search_memory(query: str) -> str:
    """Retrieves relevant past info from the shared vector store."""
    docs = shared_vector_db.similarity_search(query, k=1)
    return docs[0].page_content if docs else "No relevant past memory found."

# --- 2. Specialized Agent Definitions ---

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

def research_agent(state: AgentState) -> dict:
    """Focuses on gathering external data and saving it to memory."""
    print("🔍 [Research Agent]: Searching for information...")
    search = DuckDuckGoSearchAPIWrapper()
    
    # Logic: Search the web
    query = state["task"]
    web_results = search.run(query)
    
    # Save to shared memory for future decision making
    save_to_memory(f"Research for {query}: {web_results[:500]}")
    
    return {"research_data": web_results}

def summarizer_agent(state: AgentState) -> dict:
    """Focuses on synthesizing research and memory into a final answer."""
    print("✍️ [Summarizer Agent]: Compiling final response...")
    
    # Check long-term memory to see if we knew anything else
    past_context = search_memory(state["task"])
    
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Summarizer Agent. Use the provided research and past context to create a professional report."),
        ("human", "Research: {research}\nPast Context: {context}\n\nTask: {task}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "research": state["research_data"],
        "context": past_context,
        "task": state["task"]
    })
    
    return {"final_output": response.content}

# --- 3. Orchestration Logic ---

def run_multimodal_scenario(user_goal: str):
    """Orchestrates the collaboration between agents."""
    
    # Initialize State
    state: AgentState = {
        "task": user_goal,
        "research_data": "",
        "final_output": "",
        "chat_history": []
    }

    # Step 1: Research (Collaborative Task Execution)
    research_results = research_agent(state)
    state.update(research_results)

    # Step 2: Summarize (Using Shared Memory & Research)
    final_results = summarizer_agent(state)
    state.update(final_results)

    return state["final_output"]

# --- 4. Main Execution ---

if __name__ == "__main__":
    print("🤖 Milestone 3: Multi-Agent Collaboration Active")
    print("------------------------------------------------")
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        answer = run_multimodal_scenario(user_input)
        
        print(f"\n✅ COLLABORATIVE RESULT:\n{answer}\n")
        print("--- Memory Updated. Ready for next task. ---\n")