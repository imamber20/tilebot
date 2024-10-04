import os
import csv
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables from the .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI chat model
chat_model = ChatOpenAI(
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-3.5-turbo"
)

# Preprocess query for typo correction
def preprocess_query(query):
    corrections = {
        "Graphite Black": "Granite Black",
        # Add more corrections as needed
    }
    return corrections.get(query, query)

# Load and prepare data from the CSV file
def create_vector_store(file_path):
    loader = CSVLoader(file_path)
    documents = loader.load()

    # Combine all fields into a single text field
    for doc in documents:
        content = ' '.join([f"{key}: {value}" for key, value in doc.metadata.items()])
        doc.page_content = content
        doc.metadata = {}  # Clear metadata to avoid redundancy

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(texts, embeddings)

    return vector_store

# Create the vector store from your CSV file (assuming 'tiles_data.csv' is in the repository)
vector_store = create_vector_store("tiles_data.csv")

# Custom prompt template (if needed)
custom_prompt = """
You are a knowledgeable assistant for a tile store. Use the following context to answer the user's question.

Context:
{context}

Question:
{question}

If the answer is not in the context, try to provide a helpful response based on your knowledge.
"""

# Custom memory class to handle errors when saving context
class CustomConversationBufferMemory(ConversationBufferMemory):
    def save_context(self, inputs, outputs):
        try:
            input_str, output_str = self._get_input_output(inputs, outputs)
            super().save_context(inputs, outputs)
        except ValueError as e:
            st.error(f"Memory error: {str(e)}")

# Create ConversationalRetrievalChain with custom memory
memory = CustomConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    retriever=vector_store.as_retriever(search_kwargs={"k": 10}),
    memory=memory,
    verbose=True,
    return_source_documents=True
)

# Function to handle conversation with the chatbot
def chat_bot(user_input, history):
    chain_input = {"question": user_input, "chat_history": history}
    
    # Debugging: Show the chain input structure
    st.write(f"Chain input: {chain_input}")
    
    try:
        response = qa_chain(chain_input)
        answer = response['answer']
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        return "Sorry, something went wrong."

    # Provide additional options if requested
    if "more options" in user_input.lower():
        source_docs = response.get('source_documents', [])
        additional_info = "\nHere are more options:\n"
        for doc in source_docs:
            additional_info += f"- {doc.page_content}\n"
        answer += additional_info

    return answer

# Streamlit UI setup
st.title("Tile Store Chatbot with Voice and Text Input")

# Initialize session state for chat history if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Text input from the user
user_input = st.text_input("Type your question here:")

# When the user submits a query
if st.button("Submit"):
    if user_input:
        processed_query = preprocess_query(user_input)
        response = chat_bot(processed_query, st.session_state.chat_history)
        st.session_state.chat_history.append((processed_query, response))

        # Display the conversation history
        for i, (query, resp) in enumerate(st.session_state.chat_history):
            st.write(f"**User:** {query}")
            st.write(f"**Assistant:** {resp}")
    else:
        st.warning("Please enter a query.")

# Optionally clear chat history
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.success("Chat history cleared.")
