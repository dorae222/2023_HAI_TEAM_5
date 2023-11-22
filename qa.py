"""Ask a question to the notion database."""
import faiss
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import pickle
import argparse
from dotenv import load_dotenv, find_dotenv
import re
load_dotenv(find_dotenv())

parser = argparse.ArgumentParser(description='Ask a question to the notion DB.')
parser.add_argument('question', type=str, help='The question to ask the notion DB')
args = parser.parse_args()

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
chain = RetrievalQAWithSourcesChain.from_chain_type(llm=ChatOpenAI(temperature=0), retriever=store.as_retriever())
result = chain({"question": args.question})
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")

# result = chain({"question": args.question})
# # print('result:',result)

# # re.split()로 분리한 결과를 모두 가져옵니다.
# split_results = re.split(r"SOURCES:\s", result['answer'])
# print('split_results:',split_results)

# # 첫 번째 결과를 '답변'으로 간주합니다.
# answer = split_results[0]

# # 나머지 결과(있다면)를 '출처'로 간주합니다.
# # '출처'가 여러 개일 수 있으므로, 이들을 다시 합치는 과정이 필요합니다.
# sources = ' '.join(split_results[1:]) if len(split_results) > 1 else "No sources found."

# print(f"Answer: {answer}")
# print(f"Sources: {sources}")
# # print('------------')
# # print(f"split_results: {split_results}")