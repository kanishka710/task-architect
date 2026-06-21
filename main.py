import os
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator
from google import genai
from dotenv import load_dotenv

# 1. SETUP: Load environment variables and configure Gemini
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="AI Task Architect")

class Task(BaseModel):
    task_name: str = Field(description="A concise title for the task")
    category: str = Field(description="Categorize as: Work, Personal, or Urgent")
    priority: int = Field(description="1 to 5, where 5 is highest")
    due_date: str = Field(default="TBD", description="Extract a date if mentioned, else 'TBD'")

    @field_validator("priority")
    @classmethod
    def check_priority_range(cls, value: int) -> int:
        if not 1 <= value <= 5:
            raise ValueError(f"Priority must be between 1 and 5, but got {value}")
        return value

    @model_validator(mode="after")
    def normalize_due_date(self):
        if not self.due_date:
            self.due_date = "TBD"
        return self

class TaskList(BaseModel):
    tasks: List[Task]

def build_task_prompt(user_input: str, error_context: str = "") -> str:
    prompt = f"""
    You are an expert administrative assistant.
    Extract structured tasks from the following messy text:
    "{user_input}"

    Return the data strictly as a JSON object matching the TaskList schema.
    """

    if error_context:
        prompt += f"\n\nPREVIOUS ERROR: {error_context}"

    return prompt


def validate_task_list(response: object) -> TaskList:
    parsed_response = getattr(response, "parsed", None)
    if parsed_response is None:
        raise ValueError("Gemini did not return parsed task data")

    return TaskList.model_validate(parsed_response.model_dump())


@app.post("/architect")
async def extract_tasks(user_input: str):
    attempts = 0
    max_retries = 3
    error_context = ""
    
    while attempts < max_retries:
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=build_task_prompt(user_input, error_context),
                config={
                    "response_mime_type": "application/json",
                    "response_schema": TaskList,
                }
            )

            return validate_task_list(response)
        
        except Exception as exc:
            attempts += 1
            error_context = str(exc)
            print(f"DEBUG: Attempt {attempts} failed with error: {error_context}")
    
    raise HTTPException(status_code=500, detail=f"Failed after {max_retries} attempts. Last error: {error_context}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)