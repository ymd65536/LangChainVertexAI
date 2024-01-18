from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import GCSDirectoryLoader

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

#index = mengine.create_index(
#    embedding_gcs_uri=f"gs://{config.ME_EMBEDDING_DIR}/init_index",
#    dimensions=config.ME_DIMENSIONS,
#    index_update_method="streaming",
#    index_algorithm="tree-ah",
#)
#if index:
#    print(index.name)

index_endpoint = mengine.deploy_index()
if index_endpoint:
    print(f"Index endpoint resource name: {index_endpoint.name}")
    print(
        f"Index endpoint public domain name: {index_endpoint.public_endpoint_domain_name}"
    )
    print("Deployed indexes on the index endpoint:")
    for d in index_endpoint.deployed_indexes:
        print(f"    {d.id}")

GCS_BUCKET_DOCS = f"{config.PROJECT_ID}-documents"
# ! set -x && gsutil mb -p $PROJECT_ID -l asia-northeast1 gs://$GCS_BUCKET_DOCS

folder_prefix = "documents/google-research-pdfs/"
# !gsutil rsync -r gs://github-repo/documents/google-research-pdfs/ gs://$GCS_BUCKET_DOCS/$folder_prefix

print(f"Processing documents from {config.GCS_BUCKET_DOCS}")
loader = GCSDirectoryLoader(
    project_name=config.PROJECT_ID,
    bucket=config.GCS_BUCKET_DOCS,
    prefix=folder_prefix
)
documents = loader.load()

# Add document name and source to the metadata
for document in documents:
    doc_md = document.metadata
    document_name = doc_md["source"].split("/")[-1]
    # derive doc source from Document loader
    doc_source_prefix = "/".join(GCS_BUCKET_DOCS.split("/")[:3])
    doc_source_suffix = "/".join(doc_md["source"].split("/")[4:-1])
    source = f"{doc_source_prefix}/{doc_source_suffix}"
    document.metadata = {"source": source, "document_name": document_name}

print(f"# of documents loaded (pre-chunking) = {len(documents)}")

# split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
)
doc_splits = text_splitter.split_documents(documents)

# Add chunk number to metadata
for idx, split in enumerate(doc_splits):
    split.metadata["chunk"] = idx

# Store docs as embeddings in Matching Engine index
# It may take a while since API is rate limited
texts = [doc.page_content for doc in doc_splits]
metadatas = [
    [
        {"namespace": "source", "allow_list": [doc.metadata["source"]]},
        {"namespace": "document_name", "allow_list": [doc.metadata["document_name"]]},
        {"namespace": "chunk", "allow_list": [str(doc.metadata["chunk"])]},
    ]
    for doc in doc_splits
]

print(f"# of documents = {len(doc_splits)}")
embeddings = embedding.CustomVertexAIEmbeddings(
    requests_per_minute=embedding.EMBEDDING_QPM,
    num_instances_per_batch=embedding.EMBEDDING_NUM_BATCH,
)

ME_INDEX_ID, ME_INDEX_ENDPOINT_ID = mengine.get_index_and_endpoint()
me = MatchingEngine.from_components(
    project_id=config.PROJECT_ID,
    region=config.ME_REGION,
    gcs_bucket_name=f"gs://{config.ME_EMBEDDING_DIR}".split("/")[2],
    embedding=embeddings,
    index_id=ME_INDEX_ID,
    endpoint_id=ME_INDEX_ENDPOINT_ID,
)

doc_ids = me.add_texts(texts=texts, metadatas=metadatas)
similar_doc = me.similarity_search("What are video localized narratives?", k=2)[0]

print(similar_doc.page_content)
