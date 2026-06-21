import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

def run_connection_test():
	response = client.models.generate_content(
		model="gemini-3-flash-preview",
		contents="Say 'System Online' if you can hear me.",
	)
	print(f"AI Response: {response.text}")


if __name__ == "__main__":
	run_connection_test()