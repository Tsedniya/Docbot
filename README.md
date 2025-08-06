# AI Chatbot

A lightweight AI assistant that can chat generally or answer questions about uploaded documents using LangChain, TF-IDF embeddings, and Google's Gemini model.

## Features

- 💬 **General Chat**: Chat with the AI about anything
- 📄 **Optional Document Upload**: Support for PDF and TXT files
- 🧠 **Smart Retrieval**: Uses lightweight TF-IDF embeddings to find relevant document chunks
- 💾 **Memory**: Maintains conversation context
- 💾 **Session Management**: Save and load chat sessions
- 📚 **Source Citations**: View the source documents for each answer
- 🔄 **Flexible Mode**: Switch between general chat and document-based Q&A

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirement.txt
   ```

2. **Set up Google Gemini API**:
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a `.env` file with:
   ```
   GOOGLE_API_KEY=your_gemini_api_key
   LANGCHAIN_API_KEY=your_langsmith_api_key  # Optional
   ```

## Usage

1. **Run the application**:
   ```bash
   streamlit run document_chatbot.py
   ```

2. **Start chatting immediately**:
   - You can start chatting right away without uploading any documents
   - Ask general questions or have conversations

3. **Optional: Upload a document**:
   - Use the sidebar to upload a PDF or TXT file
   - Click "Add Document" to process it
   - Ask questions about your document

4. **Switch modes**:
   - Clear documents to return to general chat mode
   - Upload new documents to switch to document Q&A mode

## How it Works

1. **General Chat Mode**: 
   - Works as a standard AI chatbot without document context
   - Maintains conversation memory for context

2. **Document Mode**:
   - Documents are split into chunks and converted to TF-IDF embeddings
   - Chunks are stored in memory with their embeddings
   - When you ask a question:
     - The system finds the most relevant document chunks using cosine similarity
     - Uses the LLM to generate an answer based on those chunks
     - Maintains conversation context for follow-up questions

## File Structure

- `document_chatbot.py` - Main application
- `requirement.txt` - Python dependencies
- `chat_sessions/` - Directory for saved chat sessions (created automatically)

## Customization

You can easily customize the chatbot by modifying:

- **Embedding Method**: Modify the `LightweightEmbeddings` class to use different vectorization techniques
- **LLM**: Replace `ChatGoogleGenerativeAI` with other LangChain LLMs
- **Chunk Size**: Adjust `chunk_size` and `chunk_overlap` in the text splitter
- **Retrieval**: Modify the number of chunks retrieved (`k` parameter)
- **TF-IDF Parameters**: Adjust `max_features`, `ngram_range`, etc. in the TfidfVectorizer

## Troubleshooting

- **API Key not found**: Make sure `GOOGLE_API_KEY` is set in your `.env` file
- **Gemini API errors**: Check your API key and ensure you have access to Gemini
- **Memory issues**: Reduce chunk size or adjust TF-IDF parameters
- **Performance**: TF-IDF embeddings are faster but may be less accurate than neural embeddings 
