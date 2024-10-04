import os
import csv
import gradio as gr
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

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

# Load and prepare data
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

# Create vector store
vector_store = create_vector_store("tiles_data.csv")

# Custom prompt template
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a knowledgeable assistant for a tile store. Use the following context to answer the user's question.

Context:
{context}

Question:
{question}

If the answer is not in the context, try to provide a helpful response based on your knowledge.
"""
)

# Create ConversationalRetrievalChain with custom prompt
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    retriever=vector_store.as_retriever(search_kwargs={"k": 10}),
    memory=memory,
    verbose=True,
    combine_docs_chain_kwargs={'prompt': custom_prompt},
    return_source_documents=True
)

def transcribe_audio(audio_file):
    # Implement audio transcription using a service of your choice
    # For now, we'll return a placeholder message
    return "Audio transcription placeholder. Implement actual transcription here."

def chat_bot(user_input, history):
    chain_input = {"question": user_input, "chat_history": history}
    response = qa_chain(chain_input)
    answer = response['answer']

    # Provide additional options if requested
    if "more options" in user_input.lower():
        source_docs = response.get('source_documents', [])
        additional_info = "\nHere are more options:\n"
        for doc in source_docs:
            additional_info += f"- {doc.page_content}\n"
        answer += additional_info

    return answer

def process_input(input_type, text_input, audio_input, history):
    if input_type == "text":
        user_input = preprocess_query(text_input)
    elif input_type == "audio" and audio_input is not None:
        user_input = transcribe_audio(audio_input)
        if isinstance(user_input, str) and user_input.startswith("Error:"):
            return f"An error occurred during transcription: {user_input}", history
    else:
        return "No valid input detected. Please try again.", history

    response = chat_bot(user_input, history)
    history.append((user_input, response))
    return history, history

with gr.Blocks() as demo:
    gr.Markdown("# Tile Store Chatbot with Voice Input")
    chatbot = gr.Chatbot()
    state = gr.State([])

    with gr.Row():
        text_input = gr.Textbox(placeholder="Type your message here...")
        audio_input = gr.Audio(type="filepath")
    with gr.Row():
        submit_button = gr.Button("Submit")

    submit_button.click(
        lambda text, audio, hist: process_input("text" if text else "audio", text, audio, hist),
        inputs=[text_input, audio_input, state],
        outputs=[chatbot, state]
    )

if __name__ == "__main__":
    demo.launch(share=True)
