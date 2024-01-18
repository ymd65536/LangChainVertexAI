PROJECT_ID = ""  # @param {type:"string"}
ME_REGION = "asia-northeast1"  # @param {type:"string"}
ME_INDEX_NAME = f"{PROJECT_ID}-me-index"  # @param {type:"string"}

ME_EMBEDDING_DIR = f"{PROJECT_ID}-me-bucket"  # @param {type:"string"}
ME_DIMENSIONS = 768  # when using Vertex PaLM Embedding

GCS_BUCKET_DOCS = f"{PROJECT_ID}-documents"
folder_prefix = "documents/google-research-pdfs/"

# Create chain to answer questions
NUMBER_OF_RESULTS = 1
SEARCH_DISTANCE_THRESHOLD = 0.6

template = """SYSTEM: You are an intelligent assistant helping the users with their questions on research papers.

Question: {question}

Strictly Use ONLY the following pieces of context to answer the question at the end. Think step-by-step and then answer.

Do not try to make up an answer:
 - If the answer to the question cannot be determined from the context alone, say "I cannot determine the answer to that."
 - If the context is empty, just say "I do not know the answer to that."

=============
{context}
=============

Question: {question}
Helpful Answer:"""
