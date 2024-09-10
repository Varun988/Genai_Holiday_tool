%%writefile app.py
import os
import streamlit as st
import pickle
import time
import langchain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# load_dotenv()  # take environment variables from .env (especially groq api key)
llm = ChatGroq(
    temperature=0.9,
    groq_api_key='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',
    model_name="llama-3.1-70b-versatile"
)

st.title("Holiday📈")
st.sidebar.title("Holiday Tool")
# process_file_clicked = st.sidebar.button("Process File")
main_placeholder = st.empty()
if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def set_clicked():
    st.session_state.clicked = True

st.button('Upload File', on_click=set_clicked)
if st.session_state.clicked:
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    print(uploaded_file)
    if uploaded_file is not None:
         # load data
#       file_path = (
#     "/content/sample_data/Holidays.pdf"
# )
      st.write("File uploaded.....")
      temp_file = "./temp.pdf"
      with open(temp_file, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name
      loader = PyPDFLoader(temp_file)
      pages = loader.load_and_split()
      main_placeholder.text("Data Loading...Started...✅✅✅")

      # split data

      main_placeholder.text("Text Splitter...Started...✅✅✅")
      # create embeddings and save it to FAISS index
      embeddings =  HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

      vectorstore = FAISS.from_documents(pages, embeddings)
      main_placeholder.text("Embedding Vector Started Building...✅✅✅")
      time.sleep(2)

      # Save the FAISS index to a pickle file
      with open(file_path, "wb") as f:
          pickle.dump(vectorstore, f)
          # print(uploaded_file)

file_path = "/content/sample_data/vector_index.pkl"#run jupyter notebook to generte this file

main_placeholder = st.empty()

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)