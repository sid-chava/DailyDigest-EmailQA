# 📧 Smart Email Digest (RAG + Streamlit)

A modern, agentic email digest and Q&A app using Retrieval-Augmented Generation (RAG) with Langgraph, ChromaDB, and OpenAI. Process your Gmail inbox, get executive and market summaries, and ask questions about your emails—all in a beautiful Streamlit UI.

---

## Features
- **Automated Email Categorization**: Action Required, High-Priority, Market News, Newsletters
- **Executive & Market Summaries**: LLM-powered, concise, and actionable
- **Q&A Chat**: Ask questions about your emails using RAG (ChromaDB + OpenAI)
- **Streamlit UI**: Easy to use, interactive, and beautiful

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd EmailDigest
```

### 2. Install dependencies
We recommend using a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the root directory with the following:
```
OPENAI_API_KEY=your-openai-api-key
GMAIL_CREDENTIALS_PATH=credentials.json  # Path to your Gmail OAuth credentials
```

- **OpenAI API Key**: Get one at https://platform.openai.com/account/api-keys
- **Gmail Credentials**: Follow [Google's guide](https://developers.google.com/gmail/api/quickstart/python) to create OAuth credentials and download `credentials.json`.

### 4. (Optional) Set up Google API credentials
- Place your `credentials.json` in the project root or specify the path in `.env` as `GMAIL_CREDENTIALS_PATH`.

---

## Running the App

```bash
streamlit run src/app.py
```

- Use the sidebar to process your emails.
- View summaries and categories.
- Use the Q&A chat to ask questions about your inbox!

---

## Notes
- **ChromaDB** will store embeddings in a `.chroma/` directory by default.
- **Token/Rate Limits**: The app batches and/or truncates data to avoid OpenAI rate limits, but very large inboxes may still require patience.
- **Supported Models**: Uses OpenAI's GPT-4.1 and text-embedding-3-large by default, with fallbacks.

---

## Contributing
Pull requests and issues are welcome!

---

## License
MIT  