from langchain.vectorstores import FAISS
from langchain_ollama import ChatOllama 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub

# Load the vector store from disk
db_name = r"./myVectorStore"
vector_store = FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)

# Retrieve similar documents
question = "how to gain muscle mass?"
docs = vector_store.similarity_search(query=question, k=3, search_type="similarity")

# Retrieve documents using a retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
retriever.invoke(question)

question = "how to lose weight?"
retriever.invoke(question)

# RAG with ollama
# prompt = hub.pull("rlm/rag-prompt")

prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Answer in bullet points. Make sure your answer is relevant to the question and it is answered from the context only.
    Question: {question} 
    Context: {context} 
    Answer:
"""

prompt = ChatPromptTemplate.from_template(prompt)

llm = ChatOllama(model='gemma3:1b', base_url='http://localhost:11434')

def format_docs(docs):
    return '\n\n'.join([doc.page_content for doc in docs])

context = format_docs(docs)

rag_chain = (
    {"context": retriever|format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = "how to lose weight?"
response = rag_chain.invoke(question)

print(response)

question = "how to gain muscle mass?"
response = rag_chain.invoke(question)

print(response)





