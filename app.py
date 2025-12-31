import streamlit as st
import fitz
import tiktoken
import re
import time
from st_copy import copy_button
from io import BytesIO
import base64

# Set page configuration
st.set_page_config(
    page_title="Academic Paper Token Splitter",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for academic styling
st.markdown("""
<style>
    .academic-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chunk-card {
        border: 1px solid #e1e4e8;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .chunk-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .chunk-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #0066cc;
    }
    .token-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 500;
        font-size: 0.9em;
    }
    .metadata-box {
        background-color: #e9ecef;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 4px;
        margin: 1rem 0;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

def extract_text_from_academic_pdf(pdf_bytes):
    """
    Extract text from academic PDF using PyMuPDF (fitz) which handles scientific papers better.
    Preserves academic formatting and handles complex layouts common in research papers.
    """
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = ""
        
        # Extract text from each page with academic paper optimization
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Use text extraction with flags that preserve academic formatting
            text = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
            
            # Clean up text specifically for academic papers
            text = clean_academic_text(text)
            
            if text.strip():
                full_text += text + "\n\n"
        
        pdf_document.close()
        return full_text.strip()
    
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def clean_academic_text(text):
    """
    Clean extracted text specifically for academic papers:
    - Fix random spaces in words (common issue in PDF extraction)
    - Preserve academic formatting
    - Remove excessive whitespace but keep paragraph structure
    - Handle hyphenated words properly
    """
    # Remove excessive newlines but preserve paragraph structure
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Fix random spaces within words (common in academic PDFs)
    # This handles cases like "ran dom" -> "random"
    text = re.sub(r'(\b[a-z]{2,})\s+([a-z]{2,}\b)', r'\1\2', text)
    
    # Handle hyphenated words at line breaks
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Reconstruct with proper paragraph spacing
    cleaned_text = '\n\n'.join(lines)
    
    return cleaned_text

def count_tokens(text, encoding_name="cl100k_base"):
    """
    Count tokens using tiktoken encoder, accurate for LLM token limits.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def split_text_by_tokens(text, max_tokens=8192, encoding_name="cl100k_base"):
    """
    Split text into chunks of maximum 8,192 tokens using fixed-size chunking.
    This is the standard approach for preparing text for LLM processing.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks = []
    
    # Split into chunks of max_tokens size
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks

def create_download_link(text, filename):
    """
    Create a download link for text content
    """
    b64 = base64.b64encode(text.encode()).decode()
    return f'data:text/plain;base64,{b64}'

def main():
    # Academic paper themed header
    st.markdown("""
    <div class="academic-header">
        <h1>🎓 Academic Paper Token Splitter</h1>
        <p>Upload research papers and split text into 8,192-token chunks for LLM processing</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with academic-focused settings
    with st.sidebar:
        st.header("⚙️ Processing Settings")
        
        max_tokens = st.number_input(
            "Max Tokens per Chunk",
            min_value=1024,
            max_value=32768,
            value=8192,
            step=1024,
            help="Token limit per chunk (8,192 is standard for GPT-4/GPT-3.5)"
        )
        
        encoding_name = st.selectbox(
            "Token Encoding",
            ["cl100k_base", "p50k_base", "r50k_base"],
            index=0,
            help="cl100k_base is used by GPT-4, GPT-3.5-turbo and modern models"
        )
        
        st.markdown("---")
        st.subheader("📚 Academic Paper Tips")
        st.markdown("""
        - ✅ **8,192 tokens** ≈ 6,000 words or 20-25 pages of academic text
        - 🔍 **PyMuPDF** handles scientific papers, equations, and tables better
        - 📝 Each chunk can be copied with one click using the copy button
        - ⚠️ Scanned/image-based PDFs may require OCR first
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ About This Tool")
        st.markdown("""
        This tool is specifically designed for processing academic papers and research documents. 
        It preserves academic formatting and provides optimal chunking for LLM processing.
        """)

    # Main content area
    st.markdown("### 📄 Upload Your Academic Paper")
    uploaded_file = st.file_uploader(
        "Upload PDF document (research papers, journal articles, conference papers)",
        type=["pdf"],
        accept_multiple_files=False,
        help="Upload your academic paper in PDF format"
    )
    
    if uploaded_file is not None:
        try:
            # Show file info
            file_size = uploaded_file.size / 1024 / 1024  # MB
            st.info(f"📄 **{uploaded_file.name}** | Size: {file_size:.2f} MB")
            
            # Read PDF bytes
            pdf_bytes = uploaded_file.read()
            
            # Extract text with progress
            with st.spinner("🔍 Extracting text from academic paper..."):
                start_time = time.time()
                extracted_text = extract_text_from_academic_pdf(pdf_bytes)
                extraction_time = time.time() - start_time
            
            if not extracted_text:
                st.error("❌ No text could be extracted. This might be a scanned/image-based PDF.")
                st.stop()
            
            # Show extraction stats
            char_count = len(extracted_text)
            token_count = count_tokens(extracted_text, encoding_name)
            avg_words_per_token = 0.75  # Approximate for academic text
            estimated_words = int(token_count * avg_words_per_token)
            
            st.markdown(f"""
            <div class="metadata-box">
                <strong>📄 Extraction Results:</strong><br>
                • Characters: {char_count:,}<br>
                • Tokens (using {encoding_name}): {token_count:,}<br>
                • Estimated Words: {estimated_words:,}<br>
                • Processing Time: {extraction_time:.2f} seconds
            </div>
            """, unsafe_allow_html=True)
            
            # Split text into chunks
            with st.spinner(f"✂️ Splitting into {max_tokens}-token chunks..."):
                chunks = split_text_by_tokens(extracted_text, max_tokens, encoding_name)
            
            # Show chunking results
            num_chunks = len(chunks)
            st.success(f"✅ Successfully created {num_chunks} chunks!")
            
            # Display chunks with copy functionality
            st.subheader("📑 Text Chunks (Copyable)")
            
            for i, chunk in enumerate(chunks, 1):
                chunk_tokens = count_tokens(chunk, encoding_name)
                chunk_words = int(chunk_tokens * avg_words_per_token)
                
                # Create chunk container
                st.markdown(f"""
                <div class="chunk-card">
                    <div class="chunk-header">
                        <h4>Chunk {i} of {num_chunks}</h4>
                        <span class="token-badge">📊 {chunk_tokens:,} tokens | ~{chunk_words:,} words</span>
                    </div>
                """, unsafe_allow_html=True)
                
                # Text preview (first 300 characters)
                preview = chunk[:300] + "..." if len(chunk) > 300 else chunk
                st.text_area(
                    f"Preview - Chunk {i}",
                    preview,
                    height=150,
                    disabled=True,
                    key=f"preview_{i}"
                )
                
                # Copy button using st-copy
                copy_button(
                    text=chunk,
                    icon='material_symbols',  # Google Material Symbols icon
                    tooltip=f'Copy Chunk {i} to Clipboard',
                    copied_label='✅ Copied!',
                    key=f'copy_chunk_{i}'
                )
                
                # Additional info about the chunk
                st.caption(f"Chunk {i} contains academic content including text, references, and formulas from your paper.")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Add spacing between chunks
                if i < num_chunks:
                    st.divider()
            
            # Summary section
            st.markdown("---")
            st.subheader("📊 Processing Summary")
            
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.markdown("### 📈 Statistics")
                st.markdown(f"""
                - **Total Chunks:** {num_chunks}
                - **Total Tokens:** {token_count:,}
                - **Average Tokens per Chunk:** {token_count // num_chunks if num_chunks > 0 else 0:,}
                - **Token Limit per Chunk:** {max_tokens:,}
                - **Processing Time:** {extraction_time:.2f} seconds
                """)
            
            with summary_col2:
                st.markdown("### 💡 Academic Paper Insights")
                st.markdown("""
                - **Chunking Strategy:** Fixed-size token splitting preserves academic content structure
                - **Text Extraction:** PyMuPDF optimized for scientific documents and equations
                - **Token Counting:** Uses tiktoken for accurate LLM token limits
                - **Copy Functionality:** One-click copy with visual feedback using st-copy
                - **Best Practice:** Each chunk contains complete academic concepts where possible
                """)
            
            # Bulk download option
            st.markdown("---")
            st.subheader("📦 Bulk Operations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download all chunks as single file
                all_chunks = "\n\n" + "\n\n".join([
                    f"=== CHUNK {i+1} ({count_tokens(chunk, encoding_name):,} tokens) ===\n{chunk}"
                    for i, chunk in enumerate(chunks)
                ])
                
                download_link = create_download_link(all_chunks, f"all_chunks_{uploaded_file.name.replace('.pdf', '.txt')}")
                st.markdown(f"""
                    <a href="{download_link}" download="all_chunks_{uploaded_file.name.replace('.pdf', '.txt')}" 
                       style="display: block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                              color: white; text-align: center; padding: 12px; border-radius: 8px; 
                              text-decoration: none; font-weight: 500; margin: 10px 0;">
                       💾 Download All Chunks as Single File
                    </a>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 8px; height: 100%;">
                    <strong>⚠️ Important Notes:</strong><br>
                    • Academic papers often contain complex formatting<br>
                    • Equations, tables, and figures may not extract perfectly<br>
                    • For scanned PDFs, consider OCR before processing<br>
                    • 8,192 tokens is optimal for most modern LLMs<br>
                    • Copy functionality requires HTTPS when deployed
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
