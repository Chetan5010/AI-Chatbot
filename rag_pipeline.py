from langchain_groq import ChatGroq
from vector_database import faiss_db
from langchain_core.prompts import ChatPromptTemplate

#Step1: Setup LLM 
llm_model=ChatGroq(model="deepseek-r1-distill-llama-70b", api_key="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")

#Step2: Retrieve Docs

def retrieve_docs(query):
    return faiss_db.similarity_search(query)

def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

#Step3: Answer Question

custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context
Question: {question} 
Context: {context} 
Answer:
"""

def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    return chain.invoke({"question": query, "context": context})

