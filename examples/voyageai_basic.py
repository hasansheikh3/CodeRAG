import voyageai
from coderag.config import VOYAGEAI_API_KEY, VOYAGEAI_EMBEDDING_MODEL, EMBEDDING_DIM

vo = voyageai.Client(api_key=VOYAGEAI_API_KEY)

documents = [
    "The Mediterranean diet emphasizes fish, olive oil, and vegetables, believed to reduce chronic diseases.",
    "Photosynthesis in plants converts light energy into glucose and produces essential oxygen.",
    "20th-century innovations, from radios to smartphones, centered on electronic advancements.",
    "Rivers provide water, irrigation, and habitat for aquatic species, vital for ecosystems.",
    "Apple’s conference call to discuss fourth fiscal quarter results and business updates is scheduled for Thursday, November 2, 2023 at 2:00 p.m. PT / 5:00 p.m. ET.",
    "Shakespeare's works, like 'Hamlet' and 'A Midsummer Night's Dream,' endure in literature."
]


# Generate embeddings for the documents
embeddings = vo.embed(
    documents, 
    model=VOYAGEAI_EMBEDDING_MODEL, 
    output_dimension=EMBEDDING_DIM
)

query = "When is Apple's conference call scheduled?"

# Perform a similarity search with the query
query_embedding = vo.embed([query], model=VOYAGEAI_EMBEDDING_MODEL, input_type="query").embeddings[0]
