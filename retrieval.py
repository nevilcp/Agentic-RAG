from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from documents import doc_splits
from langchain.tools import tool

embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits, embedding=embedding
)
retriever = vectorstore.as_retriever()

@tool
def retrieve_blog_posts(query: str) -> str:
    """Search and return information about Lilian Weng blog posts."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

retriever_tool = retrieve_blog_posts