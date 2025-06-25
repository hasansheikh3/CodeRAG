import numpy as np
from coderag.config import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL
from google import genai
from google.genai import types

client = genai.Client(
    api_key=GEMINI_API_KEY
)

def generate_embeddings(text):
    """Generate embeddings using the Gemini API."""
    try:
        response = client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=1024,  # Specify the output dimensionality
            )
        )
        # Extract the embedding from the response
        embeddings = response.embeddings[0]
        embeddings = np.array(embeddings.values, dtype=np.float32)
        return embeddings.reshape(1, -1)  # Reshape to 2D array
    except Exception as e:
        print(f"Error generating embeddings with Gemini: {e}")
        return None
