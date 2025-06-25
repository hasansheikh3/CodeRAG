from google import genai
from coderag.config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

result = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents="What is the meaning of life?")

print(result.embeddings)
print(result.embeddings[0])