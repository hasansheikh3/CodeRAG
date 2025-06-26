import numpy as np
from coderag.config import VOYAGEAI_API_KEY, VOYAGEAI_EMBEDDING_MODEL
import voyageai as vo


client = vo.Client(
    api_key=VOYAGEAI_API_KEY
)

def generate_embeddings(text):
    """Generate embeddings using the Gemini API."""
    try:
        response = client.embed(
            texts=[text],
            model=VOYAGEAI_EMBEDDING_MODEL,
            output_dimension=1024  # Specify the output dimensionality
        )

        # Extract the embedding from the response
        embeddings = response.embeddings[0]
        embeddings = np.array(embeddings, dtype=np.float32)
        return embeddings.reshape(1, -1)  # Reshape to 2D array
    except Exception as e:
        print(f"Error generating embeddings with Gemini: {e}")
        return None
