from typing import Annotated, List, Dict, TypedDict, Sequence, Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from openai import RateLimitError
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
import time
import tiktoken

# Define the state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    emails: List[Dict]
    categories: Dict
    market_summary: str
    exec_summary: str

# Helper to format and truncate email for LLM
MAX_BODY_LEN = 1000

def format_email(email):
    subject = email.get('subject', '')
    sender = email.get('sender', '')
    body = email.get('body', '')
    snippet = email.get('snippet', '')
    # Truncate body and snippet to MAX_BODY_LEN
    if body:
        body = body[:MAX_BODY_LEN]
    if snippet:
        snippet = snippet[:MAX_BODY_LEN]
    if snippet:
        return f"Subject: {subject}\nFrom: {sender}\nSnippet: {snippet}\nBody: {body}\n"
    elif body:
        return f"Subject: {subject}\nFrom: {sender}\nBody: {body}\n"
    else:
        return f"Subject: {subject}\nFrom: {sender}\n"

# Node functions
def categorize_emails(state: AgentState) -> AgentState:
    """Categorize emails into different buckets, batching to avoid token limits."""
    import math
    emails = state["emails"]
    
    categorization_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert email categorizer. Categorize each email into one of these categories:
        - Action Required: Emails that need immediate action or response (security, alerts, scheduling, etc.)
        - High-Priority Read: Important information that should be read soon (notices, important updates, etc.)
        - Market News: News about markets, stocks, or financial information
        - Other newsletters: Regular newsletters and updates (tech news, quantum computing, job boards, etc., promotional emails)
        
        For each email, provide the category and a brief one-line summary."""),
        ("human", "Here are the emails to categorize: {emails}")
    ])
    
    def run_chain(input):
        llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
        chain = categorization_prompt | llm
        return chain.invoke(input)
    
    # Convert emails to string format for the prompt
    email_texts = [format_email(e) for e in emails]
    
    # Batch emails to avoid token limit
    batch_size = 3
    num_batches = math.ceil(len(email_texts) / batch_size)
    batch_results = []
    for i in range(num_batches):
        batch = email_texts[i * batch_size : (i + 1) * batch_size]
        result = run_chain({"emails": "\n\n".join(batch)})
        batch_results.append(result.content)
    
    # Combine all batch results into one string for parsing
    all_results = "\n".join(batch_results)
    
    # Parse the categorization results and update the state
    categories = {
        "Action Required": [],
        "High-Priority Read": [],
        "Market News": [],
        "Other newsletters": []
    }
    # TODO: Parse all_results to assign emails to categories properly.
    # For now, fallback to subject-based heuristic as before.
    for email in emails:
        if "urgent" in email["subject"].lower():
            categories["Action Required"].append(email)
        elif "market" in email["subject"].lower():
            categories["Market News"].append(email)
        elif "newsletter" in email["subject"].lower():
            categories["Other newsletters"].append(email)
        else:
            categories["High-Priority Read"].append(email)
    
    state["categories"] = categories
    return state

def generate_market_summary(state: AgentState) -> AgentState:
    """Generate market summary from market news emails in batches to avoid token limits."""
    import math
    market_emails = state["categories"].get("Market News", [])
    
    if not market_emails:
        state["market_summary"] = "No market news to summarize."
        return state
    
    batch_size = 5
    num_batches = math.ceil(len(market_emails) / batch_size)
    batch_summaries = []
    
    market_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a financial analyst. Create a brief market snapshot based on the market-related emails.\nFocus on key trends, important market movements, and actionable insights.\nKeep it concise and highlight the most important points."""),
        ("human", "Here are the market-related emails: {emails}")
    ])
    
    def run_chain(input):
        try:
            llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
            chain = market_prompt | llm
            return chain.invoke(input)
        except RateLimitError:
            try:
                llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
                chain = market_prompt | llm
                return chain.invoke(input)
            except RateLimitError:
                llm = ChatOpenAI(model="gpt-4.1", temperature=0)
                chain = market_prompt | llm
                return chain.invoke(input)
    
    # Summarize each batch
    for i in range(num_batches):
        batch = market_emails[i * batch_size : (i + 1) * batch_size]
        batch_text = "\n\n".join(format_email(e) for e in batch)
        result = run_chain({"emails": batch_text})
        batch_summaries.append(result.content)
    
    # Final summary of all batch summaries
    if len(batch_summaries) == 1:
        state["market_summary"] = batch_summaries[0]
    else:
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a financial analyst. Combine the following batch market summaries into a single, concise market snapshot. Only highlight the most important points."),
            ("human", "Batch summaries:\n{summaries}")
        ])
        def run_final_chain(input):
            try:
                llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
                chain = final_prompt | llm
                return chain.invoke(input)
            except RateLimitError:
                try:
                    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
                    chain = final_prompt | llm
                    return chain.invoke(input)
                except RateLimitError:
                    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
                    chain = final_prompt | llm
                    return chain.invoke(input)
        result = run_final_chain({"summaries": "\n\n".join(batch_summaries)})
        state["market_summary"] = result.content
    return state

def generate_exec_summary(state: AgentState) -> AgentState:
    """Generate executive summary of urgent matters, batching high-priority emails to avoid token limits."""
    import math
    urgent_emails = state["categories"].get("Action Required", [])
    high_priority = state["categories"].get("High-Priority Read", [])
    
    # Batch high-priority emails
    batch_size = 5
    num_batches = math.ceil(len(high_priority) / batch_size) if high_priority else 0
    high_priority_summaries = []
    
    batch_prompt = ChatPromptTemplate.from_messages([
        ("system", """Summarize the following high-priority emails. Focus on the most important information and keep it concise."""),
        ("human", "High Priority emails: {priority}")
    ])
    def run_batch_chain(input):
        try:
            llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
            chain = batch_prompt | llm
            return chain.invoke(input)
        except RateLimitError:
            try:
                llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
                chain = batch_prompt | llm
                return chain.invoke(input)
            except RateLimitError:
                llm = ChatOpenAI(model="gpt-4.1", temperature=0)
                chain = batch_prompt | llm
                return chain.invoke(input)
    if high_priority:
        for i in range(num_batches):
            batch = high_priority[i * batch_size : (i + 1) * batch_size]
            batch_text = "\n".join(format_email(e) for e in batch)
            result = run_batch_chain({"priority": batch_text})
            high_priority_summaries.append(result.content)
    
    # Combine all high-priority summaries
    if len(high_priority_summaries) > 1:
        combine_prompt = ChatPromptTemplate.from_messages([
            ("system", "Combine the following high-priority email summaries into a single concise summary, highlighting only the most critical points."),
            ("human", "Batch summaries:\n{summaries}")
        ])
        def run_combine_chain(input):
            try:
                llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
                chain = combine_prompt | llm
                return chain.invoke(input)
            except RateLimitError:
                try:
                    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
                    chain = combine_prompt | llm
                    return chain.invoke(input)
                except RateLimitError:
                    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
                    chain = combine_prompt | llm
                    return chain.invoke(input)
        combined_high_priority = run_combine_chain({"summaries": "\n\n".join(high_priority_summaries)}).content
    elif len(high_priority_summaries) == 1:
        combined_high_priority = high_priority_summaries[0]
    else:
        combined_high_priority = ""
    
    # Now create the final executive summary with unbatched urgent emails and batched high-priority summary
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", """Create a brief executive summary of urgent matters from the emails.\nFocus on action items and high-priority information that needs immediate attention.\nBe concise and highlight only the most critical points."""),
        ("human", """Action Required emails: {urgent}\nHigh Priority summary: {priority}""")
    ])
    def run_final_chain(input):
        try:
            llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
            chain = summary_prompt | llm
            return chain.invoke(input)
        except RateLimitError:
            try:
                llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
                chain = summary_prompt | llm
                return chain.invoke(input)
            except RateLimitError:
                llm = ChatOpenAI(model="gpt-4.1", temperature=0)
                chain = summary_prompt | llm
                return chain.invoke(input)
    result = run_final_chain({
        "urgent": "\n".join(format_email(e) for e in urgent_emails),
        "priority": combined_high_priority
    })
    
    state["exec_summary"] = result.content
    return state

def create_email_pipeline():
    """Create and return the email processing pipeline."""
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes with distinct names from state keys
    workflow.add_node("categorize_node", categorize_emails)
    workflow.add_node("market_summary_node", generate_market_summary)
    workflow.add_node("exec_summary_node", generate_exec_summary)
    
    # Add edges
    workflow.add_edge(START, "categorize_node")
    workflow.add_edge("categorize_node", "market_summary_node")
    workflow.add_edge("market_summary_node", "exec_summary_node")
    workflow.add_edge("exec_summary_node", END)
    
    # Compile the graph
    return workflow.compile()

# Q&A RAG function using ChromaDB

def answer_question_with_rag(state: AgentState, question: str, chroma_collection_name: str = "emails") -> str:
    # Prepare documents for embedding
    emails = state["emails"]
    docs = []
    metadatas = []
    for idx, email in enumerate(emails):
        content = email.get("body", "")
        if not content:
            content = email.get("subject", "")
        docs.append(content)
        metadatas.append({"index": idx, "subject": email.get("subject", ""), "sender": email.get("sender", "")})

    # Set up ChromaDB (persist in a temp dir or memory)
    persist_dir = os.path.join(".chroma", chroma_collection_name)
    os.makedirs(persist_dir, exist_ok=True)
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    vectordb = Chroma(
        collection_name=chroma_collection_name,
        embedding_function=embedding,
        persist_directory=persist_dir
    )
    # Add documents if collection is empty
    if vectordb._collection.count() == 0:
        encoding = tiktoken.encoding_for_model("text-embedding-3-large")
        MAX_TOKENS = 600_000
        batch_docs = []
        batch_metas = []
        batch_tokens = 0
        sleep_time = 10
        for doc, meta in zip(docs, metadatas):
            doc_tokens = len(encoding.encode(doc))
            if batch_tokens + doc_tokens > MAX_TOKENS and batch_docs:
                while True:
                    try:
                        vectordb.add_texts(batch_docs, metadatas=batch_metas)
                        time.sleep(sleep_time)
                        break
                    except RateLimitError:
                        sleep_time += 5
                        time.sleep(sleep_time)
                batch_docs, batch_metas, batch_tokens = [], [], 0
            batch_docs.append(doc)
            batch_metas.append(meta)
            batch_tokens += doc_tokens
        if batch_docs:
            while True:
                try:
                    vectordb.add_texts(batch_docs, metadatas=batch_metas)
                    time.sleep(sleep_time)
                    break
                except RateLimitError:
                    sleep_time += 5
                    time.sleep(sleep_time)

    # Embed and search
    results = vectordb.similarity_search(question, k=5)
    # Each result is a Document with .page_content and .metadata
    context_emails = []
    for doc in results:
        idx = doc.metadata.get("index")
        if idx is not None and idx < len(emails):
            context_emails.append(emails[idx])

    context = "\n\n".join(format_email(e) for e in context_emails)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert assistant. Use the provided emails to answer the user's question as accurately as possible."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    def run_chain(input):
        try:
            llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
            chain = prompt | llm
            return chain.invoke(input)
        except RateLimitError:
            try:
                llm = ChatOpenAI(model="gpt-4.1", temperature=0)
                chain = prompt | llm
                return chain.invoke(input)
            except RateLimitError:
                try:
                    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
                    chain = prompt | llm
                    return chain.invoke(input)
                except RateLimitError:
                    try:
                        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                        chain = prompt | llm
                        return chain.invoke(input)
                    except RateLimitError:
                        llm = ChatOpenAI(model="gpt-4o", temperature=0)
                        chain = prompt | llm
                        return chain.invoke(input)
    result = run_chain({"context": context, "question": question})
    return result.content