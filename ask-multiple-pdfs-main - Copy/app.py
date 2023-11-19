import streamlit as st
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"  

def get_webpage_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching content from {url}: {e}")
        return ''

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_text_chunks_from_urls(urls):
    text_chunks = []
    for url in urls:
        if not url:
            st.warning("Skipping empty URL.")
            continue

        text = get_webpage_text(url)
        text_chunks.extend(get_text_chunks(text))
    return text_chunks

def get_vectorstore(text_chunks):
    if not text_chunks:
        st.warning("No text chunks found. Unable to create vector store.")
        return None

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    if vectorstore is None:
        st.warning("No vectorstore available. Unable to create conversation chain.")
        return None

    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    try:
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain
    except AttributeError as e:
        if "verbose" in str(e):  # Check if the error is related to 'verbose'
            st.warning("Attribute 'verbose' not found. Please check 'langchain' library version.")
        else:
            st.error(f"Error creating conversation chain: {e}")
        return None

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with URLs", page_icon=":globe_with_meridians:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with URLs :globe_with_meridians:")
    url_input = st.text_area("Enter URLs (separated by commas):")
    urls = [url.strip() for url in url_input.split(',')]

    if urls:
        text_chunks = get_text_chunks_from_urls(urls)
        vectorstore = get_vectorstore(text_chunks)
        st.session_state.conversation = get_conversation_chain(vectorstore)

    user_question = st.text_input("Ask a question about the URLs:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your URLs")
        st.write('\n'.join(urls))

if __name__ == '__main__':
    main()
