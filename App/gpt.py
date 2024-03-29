import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.core import VectorStoreIndex, ServiceContext, Document
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
import pyttsx3
from llama_index.core.prompts.base import ChatPromptTemplate
import speech_recognition as sr
from gtts import gTTS
import io
import openai

openai.openai_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Chat with your Documents, powered by LlamaIndex", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Chat with your Documents")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Documents...."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        llm = OpenAI(model="gpt-3.5-turbo", temperature="0.2", systemprompt="""Use the books in data file is source for the answer.Generate a valid 
                     and relevant answer to a query related to 
                     construction problems, ensure the answer is based strictly on the content of 
                     the book and not influenced by other sources. Do not hallucinate. The answer should 
                     be informative and fact-based. """)
        service_content = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
        index = VectorStoreIndex.from_documents(docs, service_context=service_content)
        return index

index = load_data()
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

def query_chatbot(query_engine, user_question):
    response = query_engine.query(user_question)
    return response.response

def display_history():
    st.title("Chat History")
    if "messages" in st.session_state:
        for message in st.session_state.messages:
            role = message['role'].capitalize()
            content = message['content']
            if role == 'User' and 'response' in message:
                st.write(f"{role}: {content}\n\nAssistant: {message['response']}")
            else:
                st.write(f"{role}: {content}")
            if "audio_bytes" in message:
                st.audio(message["audio_bytes"], format='audio/ogg', start_time=0)

def back_to_chat():
    st.session_state.show_chat_history = False

def initialize_chatbot(data_dir="./data", model="gpt-3.5-turbo", temperature=0.3):
    openai.openai_key = st.secrets["OPENAI_API_KEY"]
    documents = SimpleDirectoryReader(data_dir).load_data()
    llm = OpenAI(model=model, temperature=temperature)

    additional_questions_prompt_str = (
        "Given the context below, generate additional questions related to the user's query:\n"
        "Context:\n"
        "User Query: {query_str}\n"
        "Chatbot Response: \n"
    )

    new_context_prompt_str = (
        "We have the opportunity to generate additional questions based on new context.\n"
        "New Context:\n"
        "User Query: {query_str}\n"
        "Chatbot Response: \n"
        "Given the new context, generate additional questions related to the user's query."
        "If the context isn't useful, generate additional questions based on the original context.\n"
    )

    chat_text_qa_msgs = [
        (
            "system",
            """Generate three additional questions that facilitate deeper exploration of the main topic 
            discussed in the user's query and the chatbot's response. The questions should be relevant and
              insightful, encouraging further discussion and exploration of the topic. Keep the questions concise 
              and focused on different aspects of the main topic to provide a comprehensive understanding.""",
        ),
        ("user", additional_questions_prompt_str),
    ]
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

    # Refine Prompt
    chat_refine_msgs = [
        (
            "system",
            """Based on the user's question '{prompt}' and the chatbot's response '{response}', please 
            generate only three additional questions related to the main topic. The questions should be 
            insightful and encourage further exploration of the main topic, providing a more comprehensive 
            understanding of the subject matter.""",
        ),
        ("user", new_context_prompt_str),
    ]
    refine_template = ChatPromptTemplate.from_messages(chat_refine_msgs)
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(
        text_qa_template=text_qa_template,
        refine_template=refine_template,
        llm=llm,
    )

    return query_engine

def main():
    recognizer = sr.Recognizer()

    # Initialize show_chat_history if it doesn't exist
    if "show_chat_history" not in st.session_state:
        st.session_state.show_chat_history = False

    st.sidebar.image(r"C:\Users\Dell\Music\App\logo.png", width=100)

    if st.sidebar.button("Chat"):
        st.session_state.show_chat_history = False

    if st.sidebar.button("History"):
        st.session_state.show_chat_history = True

    # Add the Reference button without any functionality
    #st.sidebar.button("Reference")

    if st.session_state.show_chat_history:
        display_history()  # This will display the chat history if the "History" button is clicked
    else:
        prompt = ""  # Define prompt variable
        if st.button("Speak", key="speak_button", help="Click to speak your question"):  # Button to trigger voice input
            with sr.Microphone() as source:
                st.write("Listening...")
                audio = recognizer.listen(source)

            try:
                prompt = recognizer.recognize_google(audio)
                st.session_state.messages.append({"role": "user", "content": prompt})
            except sr.UnknownValueError:
                st.write("Sorry, I could not understand your command.")
            

        if prompt := st.chat_input("Your question"):
            st.session_state.messages.append({"role": "user", "content": prompt})

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "audio_bytes" in message:
                    st.audio(message["audio_bytes"], format='audio/ogg', start_time=0)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_engine.chat(prompt)
                    st.write(response.response)
                    message = {"role": "assistant", "content": response.response}

                    # Add a button to play the audio
                    tts = gTTS(response.response)
                    audio_bytes_io = io.BytesIO()
                    tts.write_to_fp(audio_bytes_io)
                    audio_bytes_io.seek(0)
                    audio_bytes = audio_bytes_io.read()
                    st.audio(audio_bytes, format='audio/ogg', start_time=0)

                    # Save the audio bytes in the session state
                    message["audio_bytes"] = audio_bytes
                    st.session_state.messages.append(message)

                    user_question = prompt
                    additional_questions = query_chatbot(initialize_chatbot(), user_question)
                    st.write("Additional Questions:")
                    st.write(additional_questions)
                    message = {"role": "assistant", "content": additional_questions}
                    st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
