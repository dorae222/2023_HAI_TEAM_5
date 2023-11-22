"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import faiss
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import VectorDBQAWithSourcesChain
import pickle
from dotenv import load_dotenv, find_dotenv
import time
load_dotenv(find_dotenv())

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store)


# From here down is all the StreamLit UI.
st.set_page_config(page_title="HAI Notion QA Bot", page_icon=":robot:")
st.header("HAI Notion QA bot")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.chat_input("질문을 입력하세요", key='input')
    return input_text


user_input = get_text()

if user_input:
    st.session_state.past.append(user_input)

    # 사용자의 질문을 먼저 보이게 하기
    message(user_input, is_user=True, key="user_input")

    result = chain({"question": user_input})
    output = f"{result['answer']}\nSources: {result['sources']}"

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        time.sleep(0.1)
        