import streamlit as st
from agent import run_agent


st.set_page_config(
    page_title="AtomBot",
    page_icon="🤖",
    layout="centered"
)


st.title("AtomBot")
st.caption("Agentic Research & Knowledge Assistant")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


question = st.chat_input(
    "Ask a question..."
)


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)


    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer = run_agent(question)

        st.write(answer.content)


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )