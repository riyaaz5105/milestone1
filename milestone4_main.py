from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

app = FastAPI(title="Multi-Agent Workflow API")

# --- Data Models ---
class WorkflowRequest(BaseModel):
    topic: str

class WorkflowResponse(BaseModel):
    status: str
    research_summary: str
    final_output: str

# --- Mock Agent Logic (Replace with LLM calls) ---
def researcher_agent(topic: str):
    # Simulating a search/scrape task
    time.sleep(2) 
    return f"Found 3 key trends about {topic}: 1. Automation, 2. Scalability, 3. Integration."

def writer_agent(data: str):
    # Simulating a formatting/writing task
    time.sleep(1)
    return f"Subject: Latest Update\n\nBased on our research: {data}\n\nBest regards, AI Agent."

# --- API Endpoints ---
@app.post("/run-workflow", response_model=WorkflowResponse)
async def trigger_workflow(request: WorkflowRequest):
    try:
        # Step 1: Research
        raw_data = researcher_agent(request.topic)
        
        # Step 2: Write
        email_content = writer_agent(raw_data)
        
        return {
            "status": "success",
            "research_summary": raw_data,
            "final_output": email_content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)