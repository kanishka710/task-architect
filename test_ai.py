import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load the key from your .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure the AI
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-3-flash-preview')

# Simple test prompt
response = model.generate_content("Say 'System Online' if you can hear me.")
print(f"AI Response: {response.text}")