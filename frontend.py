# frontend.py

import streamlit as st
from backend import build_index, query_llm

# Emojis for user and bot
USER_EMOJI = "👤"
BOT_EMOJI = "🤖"

# Streamlit page config
st.set_page_config(page_title="Document Chatbot", layout="wide")

# Sidebar for chat history
st.sidebar.title("Chat History")
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Main UI
st.title("Chat with your Documents - RAG Chatbot")

# Prompt the user to enter the folder path
folder_path = st.text_input("Enter the folder path containing your PDF files:", value="test/")

# Build the index when the user provides the folder path
if folder_path:
    st.write("Building the index from files in the folder...")
    vector_db = build_index(folder_path)
    st.write("Index has been built successfully!")

    # Allow the user to ask questions
    user_query = st.text_input("Ask a question about your PDFs:")

    if st.button("Send"):
        if user_query:
            # Store user question in chat history
            st.session_state["chat_history"].append((USER_EMOJI, user_query))
            
            # Query the Claude Foundation Model
            st.write("Searching for the best match and querying the LLM...")
            response = query_llm(vector_db, user_query)

            # Store bot response in chat history
            st.session_state["chat_history"].append((BOT_EMOJI, response))
        
        # Display the chat history in the sidebar
        st.sidebar.subheader("Conversation Log")
        for emoji, message in st.session_state["chat_history"]:
            st.sidebar.markdown(f"{emoji} {message}")

    # Display the conversation in the main area
    st.write("---")
    st.subheader("Conversation")
    for emoji, message in st.session_state["chat_history"]:
        st.markdown(f"{emoji} **{message}**")

