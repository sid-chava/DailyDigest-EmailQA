import streamlit as st
import os
from dotenv import load_dotenv
from email_fetcher import EmailFetcher
from rag_pipeline import create_email_pipeline, AgentState, answer_question_with_rag

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Email Digest - AI Powered",
    page_icon="📧",
    layout="wide"
)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

def process_emails(days: int):
    """Process emails and update the UI with streaming updates."""
    # Initialize email fetcher
    fetcher = EmailFetcher(os.getenv('GMAIL_CREDENTIALS_PATH', 'credentials.json'))
    
    # Create progress placeholder
    progress_placeholder = st.empty()
    
    with progress_placeholder.container():
        st.write("📥 Fetching emails...")
        emails = fetcher.fetch_recent_emails(days)
        
        st.write("🤖 Processing emails...")
        # Create and run the pipeline
        pipeline = create_email_pipeline()
        
        # Initialize state
        state = {
            "messages": [],
            "emails": emails,
            "categories": {},
            "market_summary": "",
            "exec_summary": ""
        }
        
        # Run the pipeline
        result = pipeline.invoke(state)
        
        # Store results in session state
        st.session_state.processed_data = result
        
        st.write("✅ Processing complete!")
    
    # Clear progress messages
    progress_placeholder.empty()

def display_results():
    """Display the processed email digest."""
    data = st.session_state.processed_data
    
    # Executive Summary
    st.header("📋 Executive Summary")
    st.write(data["exec_summary"])
    
    # Market Summary
    st.header("📈 Market Snapshot")
    st.write(data["market_summary"])
    
    # Email Categories
    st.header("📬 Email Categories")
    
    # Create tabs for each category
    tabs = st.tabs([
        "🚨 Action Required",
        "⚡ High-Priority Read",
        "💹 Market News",
        "📰 Other Newsletters"
    ])
    
    for tab, (category, emails) in zip(tabs, data["categories"].items()):
        with tab:
            if not emails:
                st.info(f"No emails in {category}")
            else:
                for email in emails:
                    with st.expander(f"{email['subject']} - From: {email['sender']}"):
                        st.write(email['snippet'])
                        st.caption(f"Received: {email['date']}")

def main():
    st.title("📧 Smart Email Digest")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        days = st.slider("Days of emails to process", 1, 7, 1)
        if st.button("Process Emails"):
            process_emails(days)
    
    # Display results if available
    if st.session_state.processed_data:
        display_results()
    else:
        st.info("👈 Use the sidebar to process your emails")

# Use processed emails if available, else show a prompt
if st.session_state.get("processed_data") and st.session_state.processed_data.get("emails"):
    emails = st.session_state.processed_data["emails"]
    state = AgentState(
        messages=[],
        emails=emails,
        categories=st.session_state.processed_data.get("categories", {}),
        market_summary=st.session_state.processed_data.get("market_summary", ""),
        exec_summary=st.session_state.processed_data.get("exec_summary", "")
    )
else:
    st.info("Process your emails first to enable Q&A.")
    emails = []
    state = AgentState(
        messages=[],
        emails=emails,
        categories={},
        market_summary="",
        exec_summary=""
    )

st.title("Email Q&A RAG Chat")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question about your emails:")

if st.button("Ask") and user_input:
    with st.spinner("Thinking..."):
        answer = answer_question_with_rag(state, user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Assistant", answer))

for speaker, msg in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {msg}")

if __name__ == "__main__":
    main() 