import streamlit as st
import os
import uuid
import json
from typing import List, Dict
import tempfile
from datetime import datetime
from groq import Groq

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# Initialize Groq client
GROQ_API_KEY = "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"

class SimpleDocumentManager:
    """Manages document storage and retrieval"""
    
    def __init__(self):
        self.documents_file = "medical_documents.json"
        self.load_documents()
    
    def load_documents(self):
        """Load stored documents from file"""
        try:
            if os.path.exists(self.documents_file):
                with open(self.documents_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
            else:
                self.documents = {}
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")
            self.documents = {}
    
    def save_documents(self):
        """Save documents to file"""
        try:
            with open(self.documents_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"Error saving documents: {str(e)}")
    
    def add_document(self, doc_id: str, doc_name: str, vector_store_path: str, metadata: Dict):
        """Add a new document"""
        self.documents[doc_id] = {
            'doc_name': doc_name,
            'vector_store_path': vector_store_path,
            'metadata': metadata,
            'created_at': datetime.now().isoformat(),
            'chat_history': []
        }
        self.save_documents()
    
    def get_document(self, doc_id: str) -> Dict:
        """Get a specific document"""
        return self.documents.get(doc_id)
    
    def get_all_documents(self) -> Dict:
        """Get all documents"""
        return self.documents
    
    def update_chat_history(self, doc_id: str, user_message: str, assistant_message: str):
        """Update chat history for a document"""
        if doc_id in self.documents:
            self.documents[doc_id]['chat_history'].append({
                'timestamp': datetime.now().isoformat(),
                'user': user_message,
                'assistant': assistant_message
            })
            self.save_documents()
    
    def document_exists(self, doc_id: str) -> bool:
        """Check if document exists"""
        return doc_id in self.documents

class MedicalRAGSystem:
    """Main RAG system for medical document processing"""
    
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self.vector_store = None
        self.current_doc_id = None
        
    def extract_text_from_file(self, file_path: str, file_type: str) -> str:
        """Extract text from different file types"""
        try:
            if file_type == "pdf":
                loader = PyPDFLoader(file_path)
            elif file_type == "docx":
                loader = Docx2txtLoader(file_path)
            else:  # txt
                loader = TextLoader(file_path, encoding='utf-8')
                
            documents = loader.load()
            return "\n".join([doc.page_content for doc in documents])
        except Exception as e:
            st.error(f"Error extracting text: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """Split text into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        chunks = text_splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]
        return documents
    
    def create_vector_store(self, documents: List[Document], doc_id: str) -> str:
        """Create FAISS vector store from documents and save it"""
        self.vector_store = FAISS.from_documents(documents, self.embedding_model)
        self.current_doc_id = doc_id
        
        # Save vector store to disk
        vector_store_path = f"vector_stores/{doc_id}"
        os.makedirs("vector_stores", exist_ok=True)
        self.vector_store.save_local(vector_store_path)
        
        return vector_store_path
    
    def load_vector_store(self, vector_store_path: str, doc_id: str) -> bool:
        """Load FAISS vector store from disk"""
        try:
            if os.path.exists(vector_store_path):
                self.vector_store = FAISS.load_local(
                    vector_store_path, 
                    self.embedding_model, 
                    allow_dangerous_deserialization=True
                )
                self.current_doc_id = doc_id
                return True
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
        return False
    
    def semantic_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform semantic search on the vector store"""
        if self.vector_store is None:
            return []
        
        docs = self.vector_store.similarity_search(query, k=k)
        return docs
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using Groq LLM"""
        
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a medical assistant. Based on the provided medical context, answer the user's question accurately and professionally.

Medical Context:
{context}

User Question: {question}

Instructions:
1. Answer based ONLY on the provided medical context
2. If the answer is not in the context, say "I don't have enough information in the provided document to answer this question accurately."
3. Provide clear, concise medical information
4. Always recommend consulting healthcare professionals for medical advice

Answer:"""
        )
        
        try:
            # Format prompt
            formatted_prompt = prompt_template.format(
                context=context,
                question=query
            )
            
            # Call Groq API
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=1024
            )
            
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def process_query(self, query: str) -> str:
        """Process user query through RAG pipeline"""
        # Semantic search
        relevant_docs = self.semantic_search(query)
        
        if not relevant_docs:
            return "I don't have enough information in the provided document to answer this question. Please make sure a document has been uploaded and processed."
        
        # Combine context from relevant documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Generate answer
        answer = self.generate_answer(query, context)
        return answer

class UIManager:
    """Manages all UI components and layout"""
    
    @staticmethod
    def setup_custom_css():
        """Setup custom CSS styling"""
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: 700;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #2e86ab;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 300;
        }
        .card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 1.5rem;
            border-left: 4px solid #1f77b4;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .success-card {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
        }
        .info-card {
            background-color: #d1ecf1;
            border-left: 4px solid #17a2b8;
        }
        .user-message {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            border-left: 4px solid #2196f3;
        }
        .assistant-message {
            background-color: #f3e5f5;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            border-left: 4px solid #9c27b0;
        }
        .stButton button {
            background-color: #1f77b4;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        .stButton button:hover {
            background-color: #1668a0;
            color: white;
        }
        .document-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def setup_sidebar(doc_manager, current_doc_id, doc_selected):
        """Setup the sidebar content"""
        with st.sidebar:
            st.markdown("<h2 style='text-align: center; color: #1f77b4;'>🏥 MediChat AI</h2>", unsafe_allow_html=True)
            st.markdown("---")
            
            # Current document info
            if doc_selected and current_doc_id:
                current_doc = doc_manager.get_document(current_doc_id)
                if current_doc:
                    st.markdown("### 📄 Current Document")
                    st.markdown(f"**Name:** {current_doc['doc_name']}")
                    st.markdown(f"**ID:** `{current_doc_id}`")
                    st.markdown(f"**Uploaded:** {current_doc['created_at'][:10]}")
                    
                    if st.button("🔄 Switch Document", use_container_width=True):
                        return True  # Indicate switch requested
            
            # Statistics
            st.markdown("---")
            st.markdown("### 📊 Statistics")
            all_docs = doc_manager.get_all_documents()
            st.markdown(f"**Total Documents:** {len(all_docs)}")
            
            if doc_selected and current_doc_id:
                current_doc = doc_manager.get_document(current_doc_id)
                if current_doc:
                    st.markdown(f"**Chat History:** {len(current_doc['chat_history'])} messages")
            
            # Instructions
            st.markdown("---")
            st.markdown("### 💡 How to Use")
            st.markdown("""
            1. **Upload** a medical document
            2. **Process** it to create AI knowledge
            3. **Chat** with your document
            4. **Save** your Document ID for later
            """)
            
            return False
    
    @staticmethod
    def show_document_selection_ui(rag_system, doc_manager):
        """Show document selection UI"""
        st.markdown("---")
        
        # Welcome section
        st.markdown("""
        <div class='card info-card'>
        <h3>👋 Welcome to MediChat AI!</h3>
        <p>Upload your medical documents and chat with them using advanced AI. Get instant answers from your medical reports, research papers, and healthcare documents.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Two column layout for upload options
        col_left, col_right = st.columns(2)
        
        with col_left:
            UIManager._show_upload_section(rag_system, doc_manager)
        
        with col_right:
            UIManager._show_load_section(doc_manager)
            
            # Quick tips
            st.markdown("""
            <div class='card info-card'>
            <h4>💡 Quick Tips</h4>
            <ul>
            <li>Supported formats: PDF, DOCX, TXT</li>
            <li>Save your Document ID for future access</li>
            <li>Chat in multiple languages</li>
            <li>Get instant medical insights</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Show available documents
        UIManager._show_existing_documents(rag_system, doc_manager)
    
    @staticmethod
    def _show_upload_section(rag_system, doc_manager):
        """Show document upload section"""
        st.markdown("### 📤 Upload New Document")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose your medical document",
            type=['pdf', 'docx', 'txt'],
            key="new_upload",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            st.info(f"**Selected:** {uploaded_file.name}")
            
            if st.button("🚀 Process Document", type="primary", use_container_width=True):
                UIManager._process_uploaded_file(uploaded_file, rag_system, doc_manager)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    @staticmethod
    def _process_uploaded_file(uploaded_file, rag_system, doc_manager):
        """Process uploaded file"""
        with st.spinner("🔄 Processing your document... This may take a moment."):
            try:
                # Generate unique document ID
                doc_id = str(uuid.uuid4())[:8]
                
                # Save uploaded file temporarily
                file_extension = uploaded_file.name.split('.')[-1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Extract text
                text = rag_system.extract_text_from_file(tmp_file_path, file_extension)
                
                if text:
                    # Chunk text
                    documents = rag_system.chunk_text(text)
                
                    # Create and save vector store
                    vector_store_path = rag_system.create_vector_store(documents, doc_id)
                
                    # Store document information
                    metadata = {
                        'file_type': file_extension,
                        'chunk_count': len(documents),
                        'file_size': len(uploaded_file.getvalue())
                    }
                
                    doc_manager.add_document(doc_id, uploaded_file.name, vector_store_path, metadata)
                
                    st.session_state.current_doc_id = doc_id
                    st.session_state.doc_selected = True
                
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                    st.rerun()
                else:
                    st.error("❌ Could not extract text from the document. Please try another file.")
                
            except Exception as e:
                st.error(f"❌ Error processing document: {str(e)}")
    
    @staticmethod
    def _show_load_section(doc_manager):
        """Show document load section"""
        st.markdown("### 🔍 Load Existing Document")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        doc_id_input = st.text_input(
            "Enter your Document ID:",
            placeholder="Paste your Document ID here...",
            help="Enter the ID that was generated when you uploaded the document",
            label_visibility="collapsed"
        )
        
        if st.button("📂 Load Document", use_container_width=True):
            UIManager._load_existing_document(doc_id_input, doc_manager)
        
        st.markdown("</div>")
    
    @staticmethod
    def _load_existing_document(doc_id_input, doc_manager):
        """Load existing document by ID"""
        if doc_id_input.strip():
            if doc_manager.document_exists(doc_id_input):
                # Load vector store
                doc_data = doc_manager.get_document(doc_id_input)
                success = st.session_state.rag_system.load_vector_store(
                    doc_data['vector_store_path'], doc_id_input
                )
            
                if success:
                    st.session_state.current_doc_id = doc_id_input
                    st.session_state.doc_selected = True
                    st.rerun()
                else:
                    st.error("❌ Failed to load document")
            else:
                st.error("❌ Document ID not found")
        else:
            st.warning("⚠️ Please enter a document ID")
    
    @staticmethod
    def _show_existing_documents(rag_system, doc_manager):
        """Show list of existing documents"""
        all_docs = doc_manager.get_all_documents()
        if all_docs:
            st.markdown("---")
            st.markdown("### 📂 Your Documents")
            
            for doc_id, doc_data in all_docs.items():
                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.markdown(f"**{doc_data['doc_name']}**")
                        st.caption(f"Uploaded: {doc_data['created_at'][:10]}")
                    with col2:
                        st.code(f"ID: {doc_id}")
                    with col3:
                        if st.button("Select", key=f"select_{doc_id}"):
                            success = rag_system.load_vector_store(
                                doc_data['vector_store_path'], doc_id
                            )
                            if success:
                                st.session_state.current_doc_id = doc_id
                                st.session_state.doc_selected = True
                                st.rerun()
    
    @staticmethod
    def show_chat_interface(rag_system, doc_manager, current_doc_id):
        """Show chat interface"""
        current_doc = doc_manager.get_document(current_doc_id)
        if not current_doc:
            st.error("❌ Document not found")
            if st.button("← Go Back"):
                st.session_state.doc_selected = False
                st.session_state.current_doc_id = None
                st.rerun()
            return
        
        # Header with document info
        st.markdown("---")
        
        # Document info card
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"### 💬 Chat with: **{current_doc['doc_name']}**")
        with col2:
            st.metric("Chat Messages", len(current_doc['chat_history']))
        with col3:
            if st.button("🔄 Change Document", use_container_width=True):
                st.session_state.doc_selected = False
                st.session_state.current_doc_id = None
                st.rerun()
        
        # Chat container
        chat_container = st.container(height=500)
        
        with chat_container:
            if current_doc['chat_history']:
                for i, chat in enumerate(current_doc['chat_history']):
                    # User message
                    st.markdown(f"""
                    <div class='user-message'>
                    <strong>👤 You:</strong> {chat['user']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Assistant message
                    st.markdown(f"""
                    <div class='assistant-message'>
                    <strong>🤖 MediChat:</strong> {chat['assistant']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Timestamp
                    st.caption(f"🕒 {chat['timestamp'][:19]}")
                    
                    if i < len(current_doc['chat_history']) - 1:
                        st.markdown("---")
            else:
                st.markdown("""
                <div style='text-align: center; padding: 2rem; color: #666;'>
                <h3>🎉 Ready to Chat!</h3>
                <p>Start asking questions about your medical document. I'm here to help you understand the content better.</p>
                <p>Try questions like:</p>
                <ul style='text-align: left; display: inline-block;'>
                <li>What are the main findings in this document?</li>
                <li>Can you summarize the key points?</li>
                <li>What treatments are recommended?</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        st.markdown("---")
        user_query = st.chat_input("💭 Ask a question about your medical document...")
        
        if user_query:
            UIManager._handle_user_query(user_query, rag_system, doc_manager, current_doc_id, chat_container)
    
    @staticmethod
    def _handle_user_query(user_query, rag_system, doc_manager, current_doc_id, chat_container):
        """Handle user query and generate response"""
        # Display user message immediately
        with chat_container:
            st.markdown(f"""
            <div class='user-message'>
            <strong>👤 You:</strong> {user_query}
            </div>
            """, unsafe_allow_html=True)
        
        # Generate and display response
        with st.spinner("🔍 Searching document and generating response..."):
            response = rag_system.process_query(user_query)
        
        # Update chat history
        doc_manager.update_chat_history(current_doc_id, user_query, response)
        
        st.rerun()

def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="MediChat AI - Medical Document Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Setup custom CSS
    UIManager.setup_custom_css()
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = MedicalRAGSystem()
    if 'doc_manager' not in st.session_state:
        st.session_state.doc_manager = SimpleDocumentManager()
    if 'current_doc_id' not in st.session_state:
        st.session_state.current_doc_id = None
    if 'doc_selected' not in st.session_state:
        st.session_state.doc_selected = False
    
    # Setup sidebar
    switch_requested = UIManager.setup_sidebar(
        st.session_state.doc_manager,
        st.session_state.current_doc_id,
        st.session_state.doc_selected
    )
    
    if switch_requested:
        st.session_state.doc_selected = False
        st.session_state.current_doc_id = None
        st.rerun()
    
    # Main content area
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<h1 class='main-header'>🏥 MediChat AI</h1>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Your Intelligent Medical Document Assistant</h2>", unsafe_allow_html=True)
        
        # Show appropriate UI based on state
        if not st.session_state.doc_selected:
            # Document selection UI
            UIManager.show_document_selection_ui(
                st.session_state.rag_system,
                st.session_state.doc_manager
            )
        else:
            # Chat interface
            UIManager.show_chat_interface(
                st.session_state.rag_system,
                st.session_state.doc_manager,
                st.session_state.current_doc_id
            )

if __name__ == "__main__":
    main()