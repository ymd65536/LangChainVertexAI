# LangChainVertexAI

LangChainでVertexAIを扱う

## Setup

```bash
pip install -r requirements.txt
pip list | grep "langchain"
pip list --user | grep unstructured
```

requirements.txtの内容

```bash
pip install --user langchain==0.1.1 langchain-google-vertexai==0.0.1.post1
pip install --user unstructured==0.7.5 pdf2image==1.16.3 pytesseract==0.3.10 pdfminer.six==20221105
```

```bash
PROJECT_ID=`gcloud config list --format 'value(core.project)'`
ME_EMBEDDING_DIR="$PROJECT_ID-me-bucket"
set -x && gsutil mb -p $PROJECT_ID -l asia-northeast1 gs://$ME_EMBEDDING_DIR
```

```bash
GCS_BUCKET_DOCS="$PROJECT_ID-documents"
set -x && gsutil mb -p $PROJECT_ID -l asia-northeast1 gs://$GCS_BUCKET_DOCS
folder_prefix="documents/google-research-pdfs/"
gsutil rsync -r gs://github-repo/documents/google-research-pdfs/ gs://$GCS_BUCKET_DOCS/$folder_prefix
```


```python

# write embedding to Cloud Storage
! set -x && gsutil cp embeddings_0.json gs://{ME_EMBEDDING_DIR}/init_index/embeddings_0.json
```