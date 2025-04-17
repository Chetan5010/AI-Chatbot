import streamlit as st
from rag_pipeline import answer_query, retrieve_docs, llm_model

st.title("Doctor Chatbot")

user_query = st.text_area("Enter your prompt:", height=150, placeholder="Ask Anything!")
ask_question = st.button("Ask Doctor")

if ask_question:
    if user_query.strip() != "":
        st.chat_message("user").write(user_query)
        retrieved_docs = retrieve_docs(user_query)  # No file upload now
        response = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)
        st.chat_message("Doctor").write(response)
    else:
        st.warning("Please enter a question.")
