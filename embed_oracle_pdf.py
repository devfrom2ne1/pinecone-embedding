from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os

# 1. Load environment variables
load_dotenv()
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_env = os.environ.get("PINECONE_ENV")

# 2. Load PDF
pdf_path = "./sql-tuning-guide.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 3. Chunk
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
)
chunked_documents = text_splitter.split_documents(documents)

# 4. Embedding
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

# 5. Pinecone v3 방식
pc = Pinecone(api_key=pinecone_api_key)

index_name = "oracle-tuning-index"  # 이미 만들어진 인덱스

# 6. Connect VectorStore
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

# 7. Upload chunks
batch_size = 100
for i in range(0, len(chunked_documents), batch_size):
    print(f"Uploading batch {i} ~ {i + batch_size}")
    batch = chunked_documents[i:i + batch_size]
    vectorstore.add_documents(batch)

print("✅ PDF 문서를 Pinecone에 임베딩 완료!")
