import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template, bot_template_content
 
def get_pdf_text(pdf_docs):
    text = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text.append({
                    "file_name": pdf.name,
                    "page_number": page_num + 1,
                    "text": page_text
                })
    return text
 
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = []
    for item in text:
        chunked_text = text_splitter.split_text(item["text"])
        for chunk in chunked_text:
            chunks.append({
                "file_name": item["file_name"],
                "page_number": item["page_number"],
                "chunk": chunk
            })
    return chunks
 
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(
        texts=[chunk["chunk"] for chunk in text_chunks],
        embedding=embeddings,
        metadatas=[{"file_name": chunk["file_name"], "page_number": chunk["page_number"]} for chunk in text_chunks]
    )
    # print("Metadata in vectorstore:", [chunk["file_name"] for chunk in text_chunks])
    return vectorstore
 
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(return_source_documents=True, top_k=10),
        memory=memory
    )
    return conversation_chain
 
def get_generic_llm_response(user_question):
    llm = ChatOpenAI()
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="You are a helpful AI assistant. Answer the following question without any specific context: {question}"
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    response = llm_chain.run(user_question)
    return response
 
def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process your PDFs first.")
        return
    
    retriever = st.session_state.conversation.retriever
    retriever_results = retriever.get_relevant_documents(user_question)
    if retriever_results:
        for doc in retriever_results:
            metadata = doc.metadata
            s = f"""
                    Document Name - {metadata.get('file_name', 'Unknown')}.
                    Page Number- {metadata.get('page_number', 'Unknown')}.
                    Excerpt- {doc.page_content[:100]}..."""
    else:
        st.write("No documents retrieved by the retriever.")
    
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
 
    st.subheader("Contextual Response:")
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", "USER MESSAGE : " + message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", "Innovant Dynamic Document generator response : " + message.content), unsafe_allow_html=True)
            st.write(bot_template_content.replace("{{MSG}}", "Reference : " + s), unsafe_allow_html=True)
    generic_response = get_generic_llm_response(user_question)
    st.write(bot_template.replace("{{MSG}}", "Open ai response : " + generic_response), unsafe_allow_html=True)
 
 
 
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", layout="wide")
    st.write(css, unsafe_allow_html=True)
 
    # Add logo to the top right corner
    logo_path = os.path.join(os.getcwd(), "logo.png")  # Assuming the logo is named "logo.png" in the current directory
    if os.path.exists(logo_path):
        col1, col2 = st.columns([4, 1])
        with col2:
            st.image(logo_path, width=150)  # Adjust width as needed
    else:
        st.warning("Logo file not found. Please add 'logo.png' to the current directory.")
 
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
 
    st.header("Chat with multiple PDFs")
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        handle_userinput(user_question)
 
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
 
if __name__ == '__main__':
    main()
 #uploaded on 10:37 pm.
 