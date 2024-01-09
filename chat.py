import os

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate

load_dotenv()

st.set_page_config(page_title=f'Chat with dental assistant')
st.title('Chat with your dental assistant')

@st.cache_resource(ttl='1h')
def get_retriever():
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory='db', embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_type='mmr')
    return retriever

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ''):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


retriever = get_retriever()

msgs = StreamlitChatMessageHistory()

# memory = ConversationBufferMemory(memory_key='chat_history', chat_memory=msgs, return_messages=True)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
llm = ChatOpenAI(temperature=0, streaming=True)

# Build prompt
template = """You are an assistant who helps people learn about dental health and oral care.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm, 
    retriever=retriever, 
    memory=memory, 
    combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
    # get_chat_history=lambda h : h
    # return_source_documents=True
    verbose=False
)

if st.sidebar.button('Clear message history') or len(msgs.messages) == 0:
    msgs.clear()
    msgs.add_ai_message(f'Ask me anything about oral health care!')

avatars = {'human': 'user', 'ai': 'assistant'}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder='Ask me anything!'):
    st.chat_message('user').write(user_query)

    with st.chat_message('assistant'):
        stream_handler = StreamHandler(st.empty())
        # response = qa_chain.run(user_query, callbacks=[stream_handler])
        result = qa_chain({"question": user_query}, callbacks=[stream_handler])
        response = result["answer"]