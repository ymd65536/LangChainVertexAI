import textwrap

from langchain.llms import VertexAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Import custom Matching Engine packages
from utils.matching_engine import MatchingEngine
from utils.matching_engine_utils import MatchingEngineUtils

from embeddings import embedding_cls as embedding
from ai_config import config as config

mengine = MatchingEngineUtils(
    config.PROJECT_ID,
    config.ME_REGION,
    config.ME_INDEX_NAME
)

embeddings = embedding.CustomVertexAIEmbeddings(
    requests_per_minute=embedding.EMBEDDING_QPM,
    num_instances_per_batch=embedding.EMBEDDING_NUM_BATCH,
)

# initialize vector store
ME_INDEX_ID, ME_INDEX_ENDPOINT_ID = mengine.get_index_and_endpoint()
me = MatchingEngine.from_components(
    project_id=config.PROJECT_ID,
    region=config.ME_REGION,
    gcs_bucket_name=f"gs://{config.ME_EMBEDDING_DIR}".split("/")[2],
    embedding=embeddings,
    index_id=ME_INDEX_ID,
    endpoint_id=ME_INDEX_ENDPOINT_ID,
)

# Expose index to the retriever
retriever = me.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": config.NUMBER_OF_RESULTS,
        "search_distance": config.SEARCH_DISTANCE_THRESHOLD,
    },
)

# Text model instance integrated with langChain
llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=1024,
    temperature=0.2,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

# Uses LLM to synthesize results from the search index.
# Use Vertex PaLM Text API for LLM
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    verbose=True,
    chain_type_kwargs={
        "prompt": PromptTemplate(
            template=config.template,
            input_variables=["context", "question"],
        ),
    },
)

# Enable for troubleshooting
qa.combine_documents_chain.verbose = True
qa.combine_documents_chain.llm_chain.verbose = True
qa.combine_documents_chain.llm_chain.llm.verbose = True


def wrap(s):
    return "\n".join(textwrap.wrap(s, width=120, break_long_words=False))


def formatter(result):
    print(f"Query: {result['query']}")
    print("." * 80)
    if "source_documents" in result.keys():
        for idx, ref in enumerate(result["source_documents"]):
            print("-" * 80)
            print(f"REFERENCE #{idx}")
            print("-" * 80)
            if "score" in ref.metadata:
                print(f"Matching Score: {ref.metadata['score']}")
            if "source" in ref.metadata:
                print(f"Document Source: {ref.metadata['source']}")
            if "document_name" in ref.metadata:
                print(f"Document Name: {ref.metadata['document_name']}")
            print("." * 80)
            print(f"Content: \n{wrap(ref.page_content)}")
    print("." * 80)
    print(f"Response: {wrap(result['result'])}")
    print("." * 80)


def ask(query, qa=qa, k=config.NUMBER_OF_RESULTS, 
        search_distance=config.SEARCH_DISTANCE_THRESHOLD):
    qa.retriever.search_kwargs["search_distance"] = search_distance
    qa.retriever.search_kwargs["k"] = k
    result = qa({"query": query})
    return formatter(result)


ask("What are video localized narratives?")
