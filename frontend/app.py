import streamlit as st
import requests
import os

API_HOST = os.environ.get("API_HOST", "localhost")
API_URL = f"http://{API_HOST}:8000/query"


st.title("RAG-based HR Policy Chatbot")

# Input box for user query
query = st.text_input("Enter your question about HR policies:")

if st.button("Ask"):
    if not query:
        st.warning("Please enter a question.")
    else:
        # Prepare payload
        payload = {"question": query}

        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            data = response.json()

            st.subheader("Answer:")
            st.write(data.get("answer", "No answer received."))

            # st.subheader("Source Text:")
            # Show truncated source, with expand option
            # with st.expander("Show context"):
            #     st.write(data.get("context", "No context available."))

        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
