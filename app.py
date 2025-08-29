import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import shutil
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "I dont have enough information about that", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    # Check if there are uploaded files in session state
    if 'uploaded_files' not in st.session_state or not st.session_state.uploaded_files:
        st.warning("⚠️ Please upload and process PDF files first before asking questions!")
        return
    
    # Check if FAISS index exists
    if not os.path.exists("faiss_index"):
        st.warning("⚠️ Please upload and process PDF files first before asking questions!")
        return
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()

        response = chain(
            {"input_documents":docs, "question": user_question}
            , return_only_outputs=True)

        print(response)
        st.write("Reply: ", response["output_text"])
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure you have uploaded and processed PDF files first.")


def remove_file_from_session(file_name):
    """Remove a specific file from session state"""
    if 'uploaded_files' in st.session_state:
        st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if f.name != file_name]
        # Reprocess remaining files
        reprocess_remaining_files()


def reprocess_remaining_files():
    """Reprocess all remaining files in session state"""
    if 'uploaded_files' in st.session_state and len(st.session_state.uploaded_files) > 0:
        try:
            raw_text = get_pdf_text(st.session_state.uploaded_files)
            if raw_text.strip():
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success(f"✅ Reprocessed {len(st.session_state.uploaded_files)} file(s) successfully!")
            else:
                st.error("❌ No readable text found in remaining files.")
        except Exception as e:
            st.error(f"❌ Error reprocessing files: {str(e)}")
    else:
        # Remove FAISS index if no files left
        cleanup_faiss_index()
        st.info("All files removed. Upload new files to continue.")


def cleanup_faiss_index():
    """Remove FAISS index directory"""
    if os.path.exists("faiss_index"):
        try:
            shutil.rmtree("faiss_index")
        except Exception as e:
            st.error(f"Error cleaning up FAISS index: {str(e)}")


def display_uploaded_files():
    """Display uploaded files with remove buttons"""
    if 'uploaded_files' in st.session_state and st.session_state.uploaded_files:
        st.subheader("📄 Uploaded Files:")
        
        files_to_remove = []
        
        for i, file in enumerate(st.session_state.uploaded_files):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.write(f"📄 {file.name}")
                
            with col2:
                if st.button("❌", key=f"remove_{i}", help=f"Remove {file.name}"):
                    files_to_remove.append(file.name)
        
        # Remove files after iteration to avoid modifying list during iteration
        for file_name in files_to_remove:
            remove_file_from_session(file_name)
            st.rerun()


def initialize_session():
    """Initialize session state and clean up any orphaned FAISS index"""
    # Initialize session state for uploaded files
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
        # Clean up any existing FAISS index on fresh session
        cleanup_faiss_index()


def main():
    st.set_page_config("PDF Reader")
    st.header("Hello, I am PewDF 🤓")
    
    # Initialize session and cleanup
    initialize_session()
    
    # Add description
    st.markdown("Upload PDF files and ask questions about their content!")

    user_question = st.text_input("Please ask a Question about the PDF File. I am here to answer.")

    if user_question:
        if user_question.strip() == "":
            st.warning("⚠️ Please enter a valid question!")
        else:
            user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        
        # File uploader
        new_pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            if new_pdf_docs:
                if len(new_pdf_docs) > 0:
                    # Add new files to session state
                    for new_file in new_pdf_docs:
                        # Check if file is not already in session state
                        if not any(f.name == new_file.name for f in st.session_state.uploaded_files):
                            st.session_state.uploaded_files.append(new_file)
                    
                    with st.spinner("Processing..."):
                        try:
                            raw_text = get_pdf_text(st.session_state.uploaded_files)
                            if raw_text.strip() == "":
                                st.error("❌ Could not extract text from the uploaded PDF files. Please make sure the PDFs contain readable text.")
                            else:
                                text_chunks = get_text_chunks(raw_text)
                                get_vector_store(text_chunks)
                                st.success(f"✅ {len(st.session_state.uploaded_files)} PDF file(s) processed successfully! You can now ask questions.")
                        except Exception as e:
                            st.error(f"❌ Error processing PDF files: {str(e)}")
                else:
                    st.warning("⚠️ Please upload at least one PDF file before processing!")
            else:
                st.warning("⚠️ Please upload PDF files first!")
        
        # Display uploaded files with remove buttons
        st.markdown("---")
        display_uploaded_files()
        
        # Clear all button
        if st.session_state.uploaded_files:
            if st.button("🗑️ Clear All Files"):
                st.session_state.uploaded_files = []
                cleanup_faiss_index()
                st.success("✅ All files cleared!")
                st.rerun()
        
        # Show current status
        if st.session_state.uploaded_files and os.path.exists("faiss_index"):
            st.success(f"✅ {len(st.session_state.uploaded_files)} PDF file(s) loaded and ready!")
        else:
            st.info("📁 No PDF files loaded yet")


if __name__ == "__main__":

    main()
