import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from google import genai
from dotenv import load_dotenv

# 1. SETUP: Load environment variables and configure Gemini
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="AI Task Architect")

# 2. THE SCHEMA: This is our "Source of Truth"
# We define exactly what a 'Task' looks like.
class Task(BaseModel):
    task_name: str = Field(description="A concise title for the task")
    category: str = Field(description="Categorize as: Work, Personal, or Urgent")
    priority: int = Field(description="1 to 5, where 5 is highest")
    due_date: Optional[str] = Field(description="Extract a date if mentioned, else 'TBD'")

# --- THE HARD MODE GUARDRAIL ---
@field_validator('priority')
@classmethod
def check_priority_range(cls, v: int):
    if not (1 <= v <= 5):
        # This error will be caught by our 'try/except' block below
        raise ValueError(f"Priority must be between 1 and 5, but got {v}")
    return v

class TaskList(BaseModel):
    tasks: List[Task]

# 3. THE LOGIC: The "Reasoning" function
@app.post("/architect")
async def extract_tasks(user_input: str):
    attempts = 0
    max_retries = 3
    error_context = "" # We will fill this if the AI fails

    current_prompt = f"""
    You are an expert administrative assistant. 
    Extract structured tasks from the following messy text:
    "{user_input}"
    
    Return the data strictly as a JSON object matching the TaskList schema.
    """
    
    while attempts < max_retries:
        try:
            # We add the error context to the prompt on retries
            full_prompt = current_prompt + f"\n\nPREVIOUS ERROR: {error_context}" if error_context else current_prompt
            
            # We use 'request_mime_type' to force Gemini to return valid JSON
            # New SDK syntax for Gemini 3 Flash
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=full_prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': TaskList, # Look! We can pass the Pydantic model directly now!
                }
            )

            # This is where our 'field_validator' runs!
            # The SDK now handles the JSON parsing for us automatically
            # --- THE FIX: MANUALLY TRIGGER VALIDATION ---
            # We take the parsed data and 're-validate' it strictly.
            # This forces the @field_validator to actually raise the ValueError.
            validated_data = TaskList.model_validate(response.parsed.model_dump())
            
            return validated_data
        
        except Exception as e:
            attempts += 1
            error_context = str(e)
            print(f"DEBUG: Attempt {attempts} failed with error: {error_context}")
    
    raise HTTPException(status_code=500, detail=f"Failed after {max_retries} attempts. Last error: {error_context}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)