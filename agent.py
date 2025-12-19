from langgraph.graph import MessagesState
from langchain_ollama import ChatOllama
from retrieval import retriever_tool

response_model = ChatOllama(
    model="llama3.1",
    temperature=0
)

def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = (
        response_model.bind_tools([retriever_tool]).invoke(state["messages"])  
    )
    return {"messages": [response]}