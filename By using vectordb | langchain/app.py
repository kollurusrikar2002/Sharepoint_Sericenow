import streamlit as st
import pysnow
import re
import html
from typing import List, Dict
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Set Streamlit page configuration
st.set_page_config(page_title="ServiceNow Incident KB Assistant", layout="wide")

# Initialize embedding function
@st.cache_resource
def get_embeddings():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Custom synonym mapping
SYNONYM_MAP = {
    r'\bie\b': 'internet explorer',
    r'\bwin\b': 'windows',
    r'\bwin8\b': 'windows 8',
    r'\bwin10\b': 'windows 10',
    r'\bver\b': 'version'
}

def format_instance_url(url: str) -> str:
    """Format ServiceNow instance URL to get the instance name"""
    url = url.replace('http://', '').replace('https://', '')
    url = url.replace('.service-now.com', '')
    return url.rstrip('/')

def expand_synonyms(text: str) -> str:
    """Replace abbreviations with full terms using regex"""
    for pattern, replacement in SYNONYM_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def preprocess_text(text: str) -> str:
    """Preprocess text for better search results"""
    if not text:
        return ""
    
    # Clean text
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s.-]', ' ', text)  # Remove special chars
    text = expand_synonyms(text.lower())
    return text.strip()

def clean_html_content(content: str) -> str:
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
    content = re.sub(r'<[^>]+>', '', content)
    
    return content.strip()

def initialize_snow_client(instance_url: str, username: str, password: str):
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

def create_kb_documents(client) -> List[Document]:
    """Fetch KB articles and convert them to LangChain documents"""
    try:
        kb_table = client.resource(api_path='/table/kb_knowledge')
        query_params = {
            'sysparm_query': 'workflow_state=published',
            'sysparm_fields': 'sys_id,number,text,short_description,keywords',
            'sysparm_limit': 1000
        }
        
        response = kb_table.get(**query_params)
        documents = []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        for record in response.all():
            # Combine all text fields
            full_text = f"{record.get('short_description', '')} {record.get('text', '')}"
            processed_text = preprocess_text(full_text)
            
            # Create metadata
            metadata = {
                'sys_id': record.get('sys_id'),
                'number': record.get('number'),
                'short_description': record.get('short_description', ''),
                'keywords': record.get('keywords', ''),
                'original_text': record.get('text', '')
            }
            
            # Split text into chunks if needed
            chunks = text_splitter.create_documents(
                texts=[processed_text],
                metadatas=[metadata]
            )
            documents.extend(chunks)
        
        return documents
        
    except Exception as e:
        st.error(f"Error fetching articles: {str(e)}")
        return []

def initialize_vector_store(documents: List[Document]):
    """Initialize Chroma vector store with documents"""
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./kb_store"
    )
    return vector_store

def search_articles(vector_store, query: str, n_results: int = 5) -> List[Dict]:
    """Search articles using similarity search"""
    try:
        # Preprocess query
        processed_query = preprocess_text(query)
        
        # Perform similarity search
        docs = vector_store.similarity_search_with_relevance_scores(
            query=processed_query,
            k=n_results
        )
        
        # Format results
        results = []
        seen_numbers = set()
        
        for doc, score in docs:
            # Avoid duplicate articles
            if doc.metadata['number'] in seen_numbers:
                continue
                
            seen_numbers.add(doc.metadata['number'])
            
            result = {
                'number': doc.metadata['number'],
                'short_description': doc.metadata['short_description'],
                'original_text': doc.metadata['original_text'],
                'keywords': doc.metadata['keywords'],
                'similarity': score
            }
            results.append(result)
        
        return results
        
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def display_results(results: List[Dict]):
    """Enhanced interactive results display"""
    for result in results:
        # Create a clean title for the expander
        title = f"{result['number']}: {result['short_description']}"
        score = f"Relevance Score: {result['similarity']:.2f}"
        
        with st.expander(f"{title} ({score})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### Article Content")
                content = clean_html_content(result['original_text'])
                st.markdown(content)
            
            with col2:
                st.markdown("### Article Metadata")
                
                if result['keywords']:
                    st.markdown("**Keywords:**")
                    keywords = result['keywords'].split(',')
                    for keyword in keywords:
                        st.markdown(f"• {keyword.strip()}")
            
            st.markdown("---")

def get_incident_details(client, incident_number: str) -> Dict:
    """Fetch incident details from ServiceNow"""
    try:
        incident_table = client.resource(api_path='/table/incident')
        query = {'number': incident_number}
        response = incident_table.get(query=query)
        incident = response.one()
        return incident
    except Exception as e:
        st.error(f"Error fetching incident: {str(e)}")
        return None

def create_search_query_from_incident(incident: Dict) -> str:
    """Create search query text from incident details"""
    fields = [
        incident.get('short_description', ''),
        incident.get('description', ''),
        incident.get('comments', ''),
        incident.get('work_notes', '')
    ]
    return ' '.join([f for f in fields if f])

def post_work_note(client, incident_sys_id: str, article_details: Dict):
    """Post a work note to an incident with full KB article content"""
    try:
        incident_table = client.resource(api_path='/table/incident')
        
        # Format the work note with full article content
        note_content = f"""
KB Article Reference:
Number: {article_details['number']}
Title: {article_details['short_description']}
Relevance Score: {article_details['similarity']:.2f}
Keywords: {article_details['keywords']}

Article Content:
{clean_html_content(article_details['original_text'])}

(Added by KB Assistant)
"""
        
        update_payload = {
            "work_notes": note_content
        }
        
        response = incident_table.update(
            query={'sys_id': incident_sys_id},
            payload=update_payload
        )
        return response
    except Exception as e:
        st.error(f"Error posting work note: {str(e)}")
        return None

def display_incident_details(incident: Dict):
    """Display incident details in the UI"""
    st.subheader("Current Incident Details")
    st.write(f"Number: {incident.get('number')}")
    st.write(f"Short Description: {incident.get('short_description')}")
    
    with st.expander("Full Description"):
        st.write(incident.get('description', 'No description available'))

def main():
    st.title("ServiceNow Incident KB Assistant")
    
    st.markdown("""
    This application helps support agents by:
    1. Finding relevant KB articles based on incident details
    2. Posting the complete KB article content to the incident's work notes
    """)
    
    # Sidebar configuration
    st.sidebar.header("ServiceNow Configuration")
    
    with st.sidebar.expander("Connection Settings", expanded=True):
        instance_url = st.text_input("Instance URL")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
    
    # Session state management
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'current_incident' not in st.session_state:
        st.session_state.current_incident = None
    
    # Connect and load KB
    if st.sidebar.button("Connect & Load KB"):
        with st.spinner("Connecting to ServiceNow..."):
            client = initialize_snow_client(instance_url, username, password)
            if client:
                with st.spinner("Loading Knowledge Base..."):
                    documents = create_kb_documents(client)
                    if documents:
                        vector_store = initialize_vector_store(documents)
                        st.session_state.vector_store = vector_store
                        st.success(f"Successfully loaded {len(documents)} KB articles")
    
    # Incident search interface
    if st.session_state.vector_store:
        st.header("Incident KB Assistance")
        
        incident_number = st.text_input(
            "Enter Incident Number",
            key="incident_number",
            help="e.g. INC0012345"
        )
        
        if st.button("Analyze Incident"):
            with st.spinner("Fetching incident details..."):
                client = initialize_snow_client(instance_url, username, password)
                if client:
                    incident = get_incident_details(client, incident_number)
                    if incident:
                        st.session_state.current_incident = incident
                        
                        # Display incident details
                        display_incident_details(incident)
                        
                        # Create search query from incident details
                        search_text = create_search_query_from_incident(incident)
                        results = search_articles(st.session_state.vector_store, search_text)
                        
                        if results:
                            st.subheader(f"Found {len(results)} Relevant Articles")
                            display_results(results)
                            
                            # Store top result for work note posting
                            st.session_state.top_article = results[0]
                        else:
                            st.warning("No relevant articles found for this incident.")

        # Post to work notes section
        if 'top_article' in st.session_state and st.session_state.current_incident:
            st.markdown("---")
            st.subheader("Update Incident Work Notes")
            
            article = st.session_state.top_article
            
            # Preview the content that will be posted
            with st.expander("Preview Work Note Content"):
                st.markdown(f"**KB Article**: {article['number']}")
                st.markdown(f"**Title**: {article['short_description']}")
                st.markdown(f"**Relevance Score**: {article['similarity']:.2f}")
                st.markdown(f"**Keywords**: {article['keywords']}")
                st.markdown("**Article Content**:")
                st.markdown(clean_html_content(article['original_text']))
            
            if st.button("Post to Work Notes"):
                with st.spinner("Updating incident..."):
                    client = initialize_snow_client(instance_url, username, password)
                    if client:
                        response = post_work_note(
                            client,
                            st.session_state.current_incident['sys_id'],
                            article
                        )
                        if response:
                            st.success("Successfully updated work notes with full article content!")
                            del st.session_state.top_article

if __name__ == "__main__":
    main()
