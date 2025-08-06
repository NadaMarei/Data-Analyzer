import streamlit as st
import fitz  # PyMuPDF
import re
import io
import sys
import shutil
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, PageBreak, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
import traceback
import os
import tempfile
import time

# Robust NLTK resource handling with collocations.tab fix
def setup_nltk_resources():
    try:
        # Create a proper NLTK data directory structure
        nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Set NLTK data path
        nltk.data.path = [nltk_data_dir] + nltk.data.path
        
        # Create necessary subdirectories
        tokenizers_dir = os.path.join(nltk_data_dir, "tokenizers")
        os.makedirs(tokenizers_dir, exist_ok=True)
        
        punkt_dir = os.path.join(tokenizers_dir, "punkt")
        os.makedirs(punkt_dir, exist_ok=True)
        
        # Create special punkt_tab directory for compatibility
        punkt_tab_dir = os.path.join(tokenizers_dir, "punkt_tab", "english")
        os.makedirs(punkt_tab_dir, exist_ok=True)
        
        # Download required resources
        resources = [
            ('punkt', 'tokenizers/punkt'),
            ('stopwords', 'corpora/stopwords'),
            ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
            ('wordnet', 'corpora/wordnet')
        ]
        
        for resource_name, nltk_path in resources:
            # Check if resource exists in our custom directory
            resource_path = os.path.join(nltk_data_dir, nltk_path)
            if not os.path.exists(resource_path):
                nltk.download(resource_name, download_dir=nltk_data_dir, quiet=True)
                
        # Copy punkt files to punkt_tab for compatibility
        for filename in os.listdir(punkt_dir):
            if filename.startswith("english"):
                src = os.path.join(punkt_dir, filename)
                dst = os.path.join(punkt_tab_dir, filename)
                if not os.path.exists(dst):
                    shutil.copy(src, dst)
        
        # Create missing collocations.tab file
        collocations_path = os.path.join(punkt_tab_dir, "collocations.tab")
        if not os.path.exists(collocations_path):
            with open(collocations_path, "w") as f:
                f.write("# Empty collocations file\n")
        
        # Verify resources are properly downloaded
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        return True
    except Exception as e:
        st.error(f"NLTK setup failed: {str(e)}")
        st.error("Please check internet connection or try restarting the app")
        return False

# Set page configuration
st.set_page_config(
    page_title="Qualitative Data Analyzer",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MAX_MEMORY_CHUNK = 50 * 1024 * 1024  # 50MB chunks for large files
LARGE_FILE_THRESHOLD = 100 * 1024 * 1024  # 100MB

def extract_text_from_pdf(uploaded_file):
    """Extract text from an uploaded PDF file with efficient memory handling"""
    try:
        # Reset file pointer to beginning before reading
        uploaded_file.seek(0)
        
        # Check if file is large
        if uploaded_file.size > LARGE_FILE_THRESHOLD:
            return process_large_pdf(uploaded_file)
        else:
            # Process small files directly in memory
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                return text
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return None

def process_large_pdf(uploaded_file):
    """Process large PDF files using temporary files and streaming"""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            # Write in chunks to avoid memory overload
            uploaded_file.seek(0)
            while True:
                chunk = uploaded_file.read(MAX_MEMORY_CHUNK)
                if not chunk:
                    break
                tmp_file.write(chunk)
            tmp_file_path = tmp_file.name
        
        # Process the temporary file
        text = ""
        with fitz.open(tmp_file_path) as doc:
            for page in doc:
                text += page.get_text()
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        return text
    except Exception as e:
        st.error(f"Error processing large file: {str(e)}")
        return None

def extract_words_from_pdf(uploaded_file):
    """Extract word list from PDF (one word per line) with efficient memory handling"""
    text = extract_text_from_pdf(uploaded_file)
    if text is None:
        return None
    
    words = []
    for line in text.split('\n'):
        cleaned_line = line.strip()
        if cleaned_line:
            words.append(cleaned_line)
    return words

def preprocess_text(text):
    """Clean and tokenize text with robust error handling"""
    if not text:
        return []
    
    try:
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        return tokens
    except Exception as e:
        st.error(f"Error in text preprocessing: {str(e)}")
        return []

def count_words_in_text(text):
    """Count words in text using simple splitting method"""
    if not text:
        return 0
    return len(text.split())

def create_word_embeddings(documents):
    """Create word embeddings using Bag-of-Words and LDA"""
    if not documents:
        return {}
    
    # Create document-term matrix
    vectorizer = CountVectorizer(max_features=1000)
    try:
        dtm = vectorizer.fit_transform(documents)
    except ValueError as ve:
        st.warning(f"Vectorization warning: {str(ve)}")
        return {}
    except Exception as e:
        st.error(f"Error creating document-term matrix: {str(e)}")
        return {}
    
    # Apply LDA to get topic distributions
    try:
        lda = LatentDirichletAllocation(n_components=10, random_state=42)
        topic_distributions = lda.fit_transform(dtm)
    except Exception as e:
        st.error(f"LDA failed: {str(e)}")
        return {}
    
    # Get vocabulary
    vocabulary = vectorizer.get_feature_names_out()
    
    # Create word embeddings
    word_embeddings = {}
    for word in vocabulary:
        word_idx = vectorizer.vocabulary_.get(word)
        if word_idx is not None:
            word_topics = []
            for i in range(len(documents)):
                if dtm[i, word_idx] > 0:
                    word_topics.append(topic_distributions[i])
            
            if word_topics:
                word_embeddings[word] = np.mean(word_topics, axis=0)
    
    return word_embeddings

def find_similar_words(target_word, word_embeddings, threshold=0.7):
    """Find similar words based on cosine similarity"""
    similar_words = []
    if target_word in word_embeddings:
        target_embedding = word_embeddings[target_word]
        for word, embedding in word_embeddings.items():
            if word != target_word:
                try:
                    similarity = cosine_similarity([target_embedding], [embedding])[0][0]
                    if similarity > threshold:
                        similar_words.append(word)
                except Exception:
                    continue
    return similar_words

def count_word_frequencies(text, word_list, word_embeddings=None, use_synonyms=False):
    """Count occurrences of words with optional synonym detection"""
    frequencies = defaultdict(int)
    if not text:
        return frequencies
    
    text_lower = text.lower()
    
    for word in word_list:
        # Always count exact matches
        try:
            exact_pattern = r'\b' + re.escape(word.lower()) + r'\b'
            exact_matches = re.findall(exact_pattern, text_lower)
            frequencies[word] += len(exact_matches)
        except re.error:
            continue
        
        # Detect and count synonyms if enabled
        if use_synonyms and word_embeddings:
            similar_words = find_similar_words(word.lower(), word_embeddings)
            for synonym in similar_words:
                try:
                    syn_pattern = r'\b' + re.escape(synonym) + r'\b'
                    syn_matches = re.findall(syn_pattern, text_lower)
                    frequencies[word] += len(syn_matches)
                except re.error:
                    continue
    
    return frequencies

def generate_pdf_report(summary_data, file_analysis_data, word_counts, target_words):
    """Generate professional PDF report in memory"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
        styles = getSampleStyleSheet()
        elements = []
        
        # Add title
        title = Paragraph("Qualitative Data Analysis Report", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 12))
        
        # Document Statistics Section
        elements.append(Paragraph("Document Statistics", styles['Heading2']))
        elements.append(Spacer(1, 8))
        
        # Prepare document stats table
        stats_table_data = [
            ["Document Type", "File Name", "Word Count"],
            ["Target Words", "Word List", str(word_counts['word_list'])]
        ]
        
        for report in word_counts['reports']:
            stats_table_data.append([
                "Company Report", 
                report['name'], 
                str(report['word_count'])
            ])
        
        stats_table_data.append([
            "Total", 
            f"{len(word_counts['reports'])} Reports", 
            str(word_counts['total_report_words'])
        ])
        
        # Create stats table
        stats_table = Table(stats_table_data)
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#ecf0f1")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), 
             [colors.whitesmoke, colors.HexColor("#f9f9f9")]),
        ]))
        elements.append(stats_table)
        
        elements.append(Spacer(1, 24))
        
        # Summary table - PRESERVE USER'S WORD ORDER
        elements.append(Paragraph("Summary Report", styles['Heading2']))
        elements.append(Spacer(1, 8))
        
        # Prepare summary table data in the original word order
        summary_table_data = [["Target Word", "Frequency Count", "Percentage"]]
        for word in target_words:
            # Find the matching summary data for this word
            word_data = next((item for item in summary_data if item["Target Word"] == word), None)
            if word_data:
                summary_table_data.append([
                    word_data["Target Word"], 
                    str(word_data["Frequency Count"]), 
                    f"{word_data['Percentage']:.2f}%"
                ])
        
        # Create summary table
        summary_table = Table(summary_table_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#ecf0f1")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), 
             [colors.whitesmoke, colors.HexColor("#f9f9f9")]),
        ]))
        elements.append(summary_table)
        
        # Add space before next section
        elements.append(PageBreak())
        
        # File analysis section - PRESERVE USER'S WORD ORDER
        elements.append(Paragraph("Per-File Analysis", styles['Heading2']))
        elements.append(Spacer(1, 8))
        
        # Prepare file analysis table data in the original word order
        if file_analysis_data:
            file_names = list(next(iter(file_analysis_data.values())).keys())
            file_table_data = [["Target Word"] + list(file_names) + ["Total"]]
            
            # Add rows in the original word order
            for word in target_words:
                if word in file_analysis_data:
                    counts = file_analysis_data[word]
                    row = [word]
                    total = 0
                    for file_name in file_names:
                        count = counts.get(file_name, 0)
                        row.append(str(count))
                        total += count
                    row.append(str(total))
                    file_table_data.append(row)
            
            # Create file analysis table
            file_table = Table(file_table_data)
            file_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#3498db")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), 
                 [colors.whitesmoke, colors.HexColor("#f1f8ff")]),
            ]))
            elements.append(file_table)
        
        doc.build(elements)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"PDF generation failed: {str(e)}")
        return None

def generate_excel_report(summary_data, file_analysis_data, word_counts, target_words):
    """Generate Excel report with multiple sheets in memory"""
    try:
        buffer = io.BytesIO()
        
        # Create a workbook and add sheets
        wb = Workbook()
        
        # Document Statistics Sheet
        ws_stats = wb.active
        ws_stats.title = "Document Statistics"
        
        # Prepare stats data
        stats_data = [
            ["Document Type", "File Name", "Word Count"],
            ["Target Words", "Word List", word_counts['word_list']]
        ]
        
        for report in word_counts['reports']:
            stats_data.append(["Company Report", report['name'], report['word_count']])
        
        stats_data.append(["Total", f"{len(word_counts['reports'])} Reports", word_counts['total_report_words']])
        
        # Add data to sheet
        for row in stats_data:
            ws_stats.append(row)
        
        # Format stats sheet
        header_fill = PatternFill(start_color="2C3E50", end_color="2C3E50", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True)
        border = Border(left=Side(style='thin'), 
                       right=Side(style='thin'), 
                       top=Side(style='thin'), 
                       bottom=Side(style='thin'))
        center_aligned = Alignment(horizontal='center')
        
        for cell in ws_stats[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center_aligned
        
        for row in ws_stats.iter_rows(min_row=2, max_row=len(stats_data)):
            for cell in row:
                cell.border = border
                cell.alignment = center_aligned
        
        # Summary Report Sheet - PRESERVE USER'S WORD ORDER
        ws_summary = wb.create_sheet(title="Summary Report")
        
        # Prepare summary data in the original word order
        summary_headers = ["Target Word", "Frequency Count", "Percentage"]
        ws_summary.append(summary_headers)
        
        for word in target_words:
            # Find the matching summary data for this word
            word_data = next((item for item in summary_data if item["Target Word"] == word), None)
            if word_data:
                ws_summary.append([
                    word_data["Target Word"], 
                    word_data["Frequency Count"], 
                    word_data["Percentage"]
                ])
        
        # Format summary sheet
        for cell in ws_summary[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center_aligned
        
        for row in ws_summary.iter_rows(min_row=2, max_row=len(target_words)+1):
            for cell in row:
                cell.border = border
                cell.alignment = center_aligned
        
        # Per-File Analysis Sheet - PRESERVE USER'S WORD ORDER
        if file_analysis_data:
            ws_analysis = wb.create_sheet(title="Per-File Analysis")
            
            # Prepare headers
            file_names = list(next(iter(file_analysis_data.values())).keys())
            headers = ["Target Word"] + list(file_names) + ["Total"]
            ws_analysis.append(headers)
            
            # Add data in the original word order
            for word in target_words:
                if word in file_analysis_data:
                    counts = file_analysis_data[word]
                    row = [word]
                    total = 0
                    for file_name in file_names:
                        count = counts.get(file_name, 0)
                        row.append(count)
                        total += count
                    row.append(total)
                    ws_analysis.append(row)
            
            # Format analysis sheet
            for cell in ws_analysis[1]:
                cell.fill = PatternFill(start_color="3498DB", end_color="3498DB", fill_type="solid")
                cell.font = Font(bold=True)
                cell.alignment = center_aligned
            
            for row in ws_analysis.iter_rows(min_row=2, max_row=len(target_words)+1):
                for cell in row:
                    cell.border = border
                    cell.alignment = center_aligned
        
        # Save to buffer
        wb.save(buffer)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Excel generation failed: {str(e)}")
        return None

def main():
    try:
        # Initialize NLTK resources
        if not setup_nltk_resources():
            st.error("Critical error: NLTK setup failed. App cannot continue.")
            return
            
        st.title(":mag: Qualitative Data Analysis Tool")
        st.markdown("""
        **Upload company reports and a word list to analyze word frequencies across documents.**
        *Supports large files up to 1TB with efficient streaming*
        """)
        
        # File upload sections
        with st.expander("Upload Documents", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Company Reports")
                report_files = st.file_uploader(
                    "Upload one or more company reports (PDF)", 
                    type="pdf", 
                    accept_multiple_files=True,
                    help="Upload PDF documents containing the text to analyze"
                )
            
            with col2:
                st.subheader("Target Word List")
                word_file = st.file_uploader(
                    "Upload word list (PDF)", 
                    type="pdf",
                    help="PDF containing one target word per line"
                )
        
        # Analysis options
        with st.expander("Analysis Options", expanded=True):
            analysis_mode = st.radio(
                "Analysis Mode:",
                ["Exact words only", "Exact words and detected synonyms"],
                index=0,
                horizontal=True
            )
            
            if analysis_mode == "Exact words and detected synonyms":
                st.info("The system will automatically detect contextually similar words using semantic analysis.")
                similarity_threshold = st.slider(
                    "Similarity Sensitivity",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.7,
                    step=0.05,
                    help="Higher values detect only very similar words, lower values detect broader synonyms"
                )
        
        # Processing section
        if st.button("Analyze Documents", type="primary", use_container_width=True):
            # Validate inputs
            if not report_files:
                st.warning("Please upload at least one company report")
                return
            if not word_file:
                st.warning("Please upload a word list PDF")
                return
            
            # Check for very large files
            large_files = [f for f in report_files if f.size > LARGE_FILE_THRESHOLD]
            if large_files:
                st.info(f"Processing {len(large_files)} large file(s). This may take longer...")
            
            try:
                # Initialize progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Extract words and count them
                status_text.text("Extracting word list...")
                target_words = extract_words_from_pdf(word_file)
                if not target_words:
                    st.error("Failed to extract target words. Please check your word list PDF.")
                    return
                word_list_count = len(target_words)
                
                # Reset file pointer for report files
                for file in report_files:
                    file.seek(0)
                
                # Step 2: Process company reports - extract text and count words
                status_text.text("Processing company reports...")
                report_data = []
                total_report_words = 0
                total_files = len(report_files)
                start_time = time.time()
                
                # Extract and store text from all reports with word counts
                for i, report_file in enumerate(report_files):
                    # Show file processing status
                    file_size = report_file.size / (1024 * 1024)
                    status_text.text(f"Processing {report_file.name} ({file_size:.1f} MB)...")
                    
                    # Reset file pointer before reading
                    report_file.seek(0)
                    progress_bar.progress((i + 1) / (total_files * 2 + 2))
                    
                    # Process file with appropriate method
                    text = extract_text_from_pdf(report_file)
                    if text:
                        word_count = count_words_in_text(text)
                        report_data.append({
                            "name": report_file.name,
                            "text": text,
                            "word_count": word_count
                        })
                        total_report_words += word_count
                
                # Check if we have valid documents
                if not report_data:
                    st.error("No valid text extracted from company reports. Please check your PDF files.")
                    return
                
                # Create combined text
                combined_text = "\n\n".join([item["text"] for item in report_data])
                
                # Prepare document statistics
                document_stats = {
                    "word_list": word_list_count,
                    "reports": [{"name": item["name"], "word_count": item["word_count"]} for item in report_data],
                    "total_report_words": total_report_words
                }
                
                # Create word embeddings if needed
                word_embeddings = None
                if analysis_mode == "Exact words and detected synonyms":
                    status_text.text("Building semantic model...")
                    # Preprocess texts for embedding creation
                    processed_texts = []
                    for item in report_data:
                        tokens = preprocess_text(item["text"])
                        if tokens:
                            processed_texts.append(" ".join(tokens))
                    
                    if processed_texts:
                        word_embeddings = create_word_embeddings(processed_texts)
                    else:
                        st.warning("No valid text for semantic analysis. Using exact words only.")
                    progress_bar.progress((total_files + 1) / (total_files * 2 + 2))
                
                # Initialize file analysis structure
                file_analysis = defaultdict(lambda: defaultdict(int))
                
                # Count words in each file
                for i, data in enumerate(report_data):
                    status_text.text(f"Analyzing file {i+1}/{len(report_data)}...")
                    progress_bar.progress((total_files + i + 2) / (total_files * 2 + 2))
                    
                    # Count words for this specific file
                    file_counts = count_word_frequencies(
                        data["text"], 
                        target_words, 
                        word_embeddings, 
                        use_synonyms=(analysis_mode == "Exact words and detected synonyms")
                    )
                    
                    # Store in analysis structure
                    for word, count in file_counts.items():
                        file_analysis[word][data["name"]] = count
                
                # Step 3: Count overall word frequencies
                status_text.text("Analyzing word frequencies...")
                overall_frequencies = count_word_frequencies(
                    combined_text, 
                    target_words, 
                    word_embeddings, 
                    use_synonyms=(analysis_mode == "Exact words and detected synonyms")
                )
                total_occurrences = sum(overall_frequencies.values())
                
                # Prepare summary data with percentages - PRESERVE USER'S WORD ORDER
                summary_data = []
                for word in target_words:
                    count = overall_frequencies.get(word, 0)
                    percentage = (count / total_occurrences * 100) if total_occurrences > 0 else 0
                    summary_data.append({
                        "Target Word": word,
                        "Frequency Count": count,
                        "Percentage": percentage
                    })
                
                progress_bar.progress(100)
                
                # Display results
                processing_time = time.time() - start_time
                st.success(f"Analysis complete! Processed {total_report_words:,} words in {processing_time:.1f} seconds")
                
                # Document Statistics
                st.subheader("Document Statistics")
                stats_df = pd.DataFrame({
                    "Document Type": ["Word List"] + ["Company Report"] * len(report_data),
                    "File Name": ["Word List"] + [item["name"] for item in report_data],
                    "Word Count": [word_list_count] + [item["word_count"] for item in report_data]
                })
                st.dataframe(stats_df, height=200)
                st.markdown(f"**Total Words in Reports:** {total_report_words:,}")
                
                # Summary Report - PRESERVE USER'S WORD ORDER
                st.subheader("Summary Report")
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, height=300)
                
                # File Analysis Report - PRESERVE USER'S WORD ORDER
                st.subheader("Per-File Analysis")
                if file_analysis:
                    # Create DataFrame with words in original order
                    analysis_rows = []
                    for word in target_words:
                        if word in file_analysis:
                            row = {"Target Word": word}
                            for file_name in file_analysis[word]:
                                row[file_name] = file_analysis[word][file_name]
                            row['Total'] = sum(file_analysis[word].values())
                            analysis_rows.append(row)
                    
                    file_analysis_df = pd.DataFrame(analysis_rows)
                    file_analysis_df.set_index("Target Word", inplace=True)
                    st.dataframe(file_analysis_df, height=400)
                else:
                    st.warning("No file analysis data available")
                
                # Export options
                st.subheader("Export Full Report")
                col1, col2 = st.columns(2)
                
                # Generate reports in memory
                with st.spinner("Preparing reports..."):
                    # Generate PDF report in memory - pass target_words for ordering
                    pdf_buffer = generate_pdf_report(summary_data, file_analysis, document_stats, target_words)
                    
                    # Generate Excel report in memory - pass target_words for ordering
                    excel_buffer = generate_excel_report(summary_data, file_analysis, document_stats, target_words)
                
                if pdf_buffer:
                    with col1:
                        # Download PDF report
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_buffer,
                            file_name="Qualitative_Analysis_Report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                else:
                    st.error("PDF report generation failed")
                
                if excel_buffer:
                    with col2:
                        # Download Excel report
                        st.download_button(
                            label="Download Excel Report",
                            data=excel_buffer,
                            file_name="Qualitative_Analysis_Report.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                else:
                    st.error("Excel report generation failed")
                
                # Additional CSV exports - PRESERVE USER'S WORD ORDER
                st.subheader("Export Data Tables")
                col_csv1, col_csv2 = st.columns(2)
                
                with col_csv1:
                    # Download summary as CSV
                    csv = summary_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Summary (CSV)",
                        data=csv,
                        file_name="word_frequency_summary.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col_csv2:
                    # Download file analysis as CSV
                    if file_analysis:
                        file_analysis_csv = file_analysis_df.to_csv().encode('utf-8')
                        st.download_button(
                            label="Download File Analysis (CSV)",
                            data=file_analysis_csv,
                            file_name="per_file_analysis.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                
                status_text.empty()
                progress_bar.empty()
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.error("Please try again or check your files")
                st.text(traceback.format_exc())
    except Exception as e:
        st.error(f"Critical application error: {str(e)}")
        st.text(traceback.format_exc())

if __name__ == "__main__":
    main()
