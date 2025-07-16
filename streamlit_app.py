import streamlit as st
from main import process_query

# with st.sidebar:
#     openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
#     "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
#     "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
#     "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 TrendLens")
st.caption("🚀 A Streamlit chatbot")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    # Build chat_history
    chat_history = []
    messages = st.session_state["messages"]
    for i in range(len(messages) - 1):
        if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
            chat_history.append((messages[i]["content"], messages[i+1]["content"]))
    # Call your research pipeline with the user's input
    result = process_query(prompt, chat_history)
    st.session_state.messages.append({"role": "assistant", "content": result})
    st.chat_message("assistant").write(result)