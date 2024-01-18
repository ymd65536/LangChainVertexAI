# LangChainVertexAI

LangChainでVertexAIを扱う

## Setup

```bash
pip install -r requirements.txt
pip list | grep "langchain"
pip list --user | grep unstructured
```

requirements.txtの内容（pip installした場合）

```bash
pip install --user langchain==0.1.1 langchain-google-vertexai==0.0.1.post1
pip install --user unstructured==0.7.5 pdf2image==1.16.3 pytesseract==0.3.10 pdfminer.six==20221105
```

### EMBEDDING_DIRの作成

```bash
PROJECT_ID=`gcloud config list --format 'value(core.project)'`
ME_EMBEDDING_DIR="$PROJECT_ID-me-bucket"
set -x && gsutil mb -p $PROJECT_ID -l asia-northeast1 gs://$ME_EMBEDDING_DIR
```

### GCS_BUCKET_DOCSの作成

```bash
GCS_BUCKET_DOCS="$PROJECT_ID-documents"
set -x && gsutil mb -p $PROJECT_ID -l asia-northeast1 gs://$GCS_BUCKET_DOCS
folder_prefix="documents/google-research-pdfs/"
gsutil rsync -r gs://github-repo/documents/google-research-pdfs/ gs://$GCS_BUCKET_DOCS/$folder_prefix
```

### EMBEDDING_DIRにjsonファイルをアップロード

```bash
python create_embeddings_json.py
```

```bash
PROJECT_ID=`gcloud config list --format 'value(core.project)'`
ME_EMBEDDING_DIR="$PROJECT_ID-me-bucket"
index="init_index"
json_file="embeddings_0.json"
set -x && gsutil cp $json_file gs://$ME_EMBEDDING_DIR/$index/$json_file
```
