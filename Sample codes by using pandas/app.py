import streamlit as st

# Must be the first Streamlit command
st.set_page_config(page_title="ServiceNow KB Search", layout="wide")

import pysnow
import pandas as pd
import numpy as np
import re
import html
import spacy
from sentence_transformers import SentenceTransformer, util

# Initialize spaCy and Sentence-BERT model
@st.cache_resource
def load_nlp_models():
    nlp = spacy.load("en_core_web_sm")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return nlp, model

nlp, model = load_nlp_models()

# Custom synonym mapping
SYNONYM_MAP = {
    r'\bie\b': 'internet explorer',
    r'\bwin\b': 'windows',
    r'\bwin8\b': 'windows 8',
    r'\bwin10\b': 'windows 10',
    r'\bver\b': 'version'
}

def format_instance_url(url):
    """Format ServiceNow instance URL to get the instance name"""
    url = url.replace('http://', '').replace('https://', '')
    url = url.replace('.service-now.com', '')
    return url.rstrip('/')

def expand_synonyms(text):
    """Replace abbreviations with full terms using regex"""
    for pattern, replacement in SYNONYM_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def preprocess_text(text):
    """Advanced text preprocessing with spaCy"""
    if not text:
        return ""
    
    # Clean text
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s.-]', ' ', text)  # Remove special chars
    text = expand_synonyms(text.lower())
    
    # NLP processing
    doc = nlp(text)
    tokens = [
        token.lemma_ 
        for token in doc 
        if not token.is_stop 
        and not token.is_punct
        and len(token.lemma_) > 2
    ]
    
    return " ".join(tokens)

def clean_html_content(content):
    """Clean and format HTML content for better display"""
    if not content:
        return ""
        
    # Convert common HTML elements to Markdown
    content = html.unescape(content)
    content = re.sub(r'</?p>', '\n\n', content)
    content = re.sub(r'</?strong>', '**', content)
    content = re.sub(r'</?em>', '*', content)
    content = re.sub(r'<li>', '\n• ', content)
    content = re.sub(r'</li>', '', content)
    content = re.sub(r'</?ol>', '\n', content)
    content = re.sub(r'</?ul>', '\n', content)
    content = re.sub(r'<br\s*/?>', '\n', content)
    content = re.sub(r'<[^>]+>', '', content)  # Remove remaining HTML tags
    
    # Clean up extra whitespace
    content = re.sub(r'\n\s*\n', '\n\n', content)
    content = content.strip()
    
    return content

def initialize_snow_client(instance_url, username, password):
    """Initialize ServiceNow client with validation"""
    try:
        instance = format_instance_url(instance_url)
        if not re.match(r'^[a-zA-Z0-9-]+$', instance):
            raise ValueError("Invalid instance name format")
            
        client = pysnow.Client(
            instance=instance,
            user=username,
            password=password
        )
        
        # Test connection
        test_resource = client.resource(api_path='/table/sys_user')
        test_resource.get(query={'sysparm_limit': 1})
        return client
        
    except Exception as e:
        st.error(f"Connection failed: {str(e)}")
        return None

def get_kb_articles(client):
    """Fetch and preprocess KB articles with enhanced fields"""
    try:
        kb_table = client.resource(api_path='/table/kb_knowledge')
        query_params = {
            'sysparm_query': 'workflow_state=published',
            'sysparm_fields': 'sys_id,number,text,short_description,keywords',
            'sysparm_limit': 1000
        }
        
        response = kb_table.get(**query_params)
        articles = []
        
        for record in response.all():
            full_text = f"{record.get('short_description', '')} {record.get('text', '')} {record.get('keywords', '')}"
            articles.append({
                'sys_id': record.get('sys_id'),
                'number': record.get('number'),
                'text': preprocess_text(full_text),
                'original_text': record.get('text', ''),
                'short_description': record.get('short_description', ''),
                'keywords': record.get('keywords', '')
            })
            
        kb_dataframe = pd.DataFrame(articles)
        print(kb_dataframe)
        return kb_dataframe
        
    except Exception as e:
        st.error(f"Error fetching articles: {str(e)}")
        return pd.DataFrame()

def get_similar_articles(query, kb_df, threshold=0.4):
    """Find similar articles using Sentence-BERT embeddings"""
    if kb_df.empty:
        return None
    
    # Preprocess query
    clean_query = preprocess_text(query)
    query_embedding = model.encode([clean_query])
    
    # Prepare article embeddings
    article_texts = kb_df['text'].tolist()
    article_embeddings = model.encode(article_texts)
    
    # Calculate similarities
    similarities = util.cos_sim(query_embedding, article_embeddings)[0]
    kb_df['similarity'] = similarities.numpy()
    
    # Dynamic threshold adjustment
    dynamic_threshold = max(threshold, similarities.mean() + 0.1)
    filtered_df = kb_df[kb_df['similarity'] > dynamic_threshold]
    
    if filtered_df.empty:
        st.info("No exact matches found. Here are the closest results:")
        return kb_df.nlargest(5, 'similarity')
    
    return filtered_df.sort_values('similarity', ascending=False)

def display_results(results_df):
    """Enhanced interactive results display with better formatting"""
    for _, row in results_df.iterrows():
        # Create a clean title for the expander
        title = f"{row['number']}: {row['short_description']}"
        score = f"Relevance Score: {row['similarity']:.2f}"
        
        with st.expander(f"{title} ({score})"):
            # Create sections using columns for better organization
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Main content section
                st.markdown("### Article Content")
                
                # Clean and format the content
                content = clean_html_content(row['original_text'])
                
                # Display formatted content with max height
                st.markdown(content)
            
            with col2:
                # Metadata and matching information
                st.markdown("### Article Metadata")
                
                # Display keywords if they exist
                if row['keywords']:
                    st.markdown("**Keywords:**")
                    keywords = row['keywords'].split(',')
                    for keyword in keywords:
                        st.markdown(f"• {keyword.strip()}")
                
                # Show matching terms
                st.markdown("**Matching Terms:**")
                query_words = set(preprocess_text(st.session_state.query).split())
                article_words = set(row['text'].split())
                matching_terms = query_words.intersection(article_words)
                
                if matching_terms:
                    for term in matching_terms:
                        st.markdown(f"• {term}")
                else:
                    st.markdown("*No exact term matches*")
                
            # Add a visual separator between articles
            st.markdown("---")

def main():
    st.title("ServiceNow Knowledge Base Search")
    
    # Application description
    st.markdown("""
    This application provides intelligent search capabilities for ServiceNow Knowledge Base articles.
    Connect to your ServiceNow instance and search through KB articles using natural language.
    """)
    
    # Sidebar configuration
    st.sidebar.header("ServiceNow Configuration")
    
    # Connection settings
    with st.sidebar.expander("Connection Settings", expanded=True):
        instance_url = st.text_input("Instance URL")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
    
    # Session state management
    if 'kb_loaded' not in st.session_state:
        st.session_state.kb_loaded = False
    
    # Connect and load KB
    if st.sidebar.button("Connect & Load KB"):
        with st.spinner("Connecting to ServiceNow..."):
            client = initialize_snow_client(instance_url, username, password)
            if client:
                with st.spinner("Loading Knowledge Base..."):
                    kb_df = get_kb_articles(client)
                    if not kb_df.empty:
                        st.session_state.kb_df = kb_df
                        st.session_state.kb_loaded = True
                        st.success(f"Successfully loaded {len(kb_df)} KB articles")
    
    # Search interface
    if st.session_state.get('kb_loaded', False):
        st.header("Search Knowledge Base")
        
        # Search input
        query = st.text_input(
            "Enter your search query",
            key="query",
            help="Enter keywords or phrases to search for relevant articles"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            search_button = st.button("Search KB")
        
        # Execute search
        if search_button and query:
            with st.spinner("Searching for relevant articles..."):
                results = get_similar_articles(query, st.session_state.kb_df)
                
                if results is not None:
                    st.subheader(f"Found {len(results)} Relevant Articles")
                    display_results(results)
                else:
                    st.warning("No matching articles found. Try broadening your search terms.")

if __name__ == "__main__":
    main()
