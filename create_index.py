from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import GCSDirectoryLoader

# Import custom Matching Engine packages
from utils.matching_engine_utils import MatchingEngineUtils

from ai_config import config as config


mengine = MatchingEngineUtils(
    config.PROJECT_ID,
    config.ME_REGION,
    config.ME_INDEX_NAME
)

index = mengine.create_index(
    embedding_gcs_uri=f"gs://{config.ME_EMBEDDING_DIR}/init_index",
    dimensions=config.ME_DIMENSIONS,
    index_update_method="streaming",
    index_algorithm="tree-ah",
)
if index:
    print(index.name)

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

# split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
)
doc_splits = text_splitter.split_documents(documents)
print(f"# of documents = {len(doc_splits)}")
