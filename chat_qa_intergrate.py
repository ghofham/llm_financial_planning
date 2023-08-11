import streamlit as st
from langchain.chains import ConversationChain, LLMChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import openai,os
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI

### THIS PART I did it to assign the key without having to deal with API_)
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']
#########################

"""
AIVA: a conversational AI model built with the `langchain` and `llms` libraries.
"""


# Set Streamlit page configuration
# st.set_page_config(page_title="AIVA layout="wide")
# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []


# Define function to get user input
def get_text():
    """
    Get the user input text.

    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input(
        "You: ",
        st.session_state["input"],
        key="input",
        placeholder="Your AI assistant here! Ask me anything ...",
        label_visibility="hidden",
    )
    return input_text


#################    LOAD_QA           ####################
## This is the function that would red the pdf and try to anser the question from the DOC first
########################################################################
chat_history =[]
def load_qa(file, chain_type='stuff', k=1):
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""


    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)
    #QA_CHAIN_PROMPT = PromptTemplate(input_variables=["chat_history","context", "question"],template=template,)
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=MODEL, temperature=0),
        #memory =   memory ,     ## ADDING THE MEMORY FOR SOME REASON DOES NOT WORK
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
        #chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT}
    )
    return qa
#---------------------------------------------
# Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    if "entity_memory" not in st.session_state:
        st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=K)
    st.session_state.entity_memory.entity_store = {}
    st.session_state.entity_memory.buffer.clear()


# Set up sidebar with various options
with st.sidebar.expander("🛠️ ", expanded=False):
    # Option to preview memory store
    if st.checkbox("Preview memory store"):
        with st.expander("Memory-Store", expanded=False):
            st.session_state.entity_memory.store
    # Option to preview memory buffer
    if st.checkbox("Preview memory buffer"):
        with st.expander("Bufffer-Store", expanded=False):
            st.session_state.entity_memory.buffer
    MODEL = st.selectbox(
        label="Model",
        options=[
            "gpt-3.5-turbo"
            # "text-davinci-003",
            # "text-davinci-002",
            # "code-davinci-002",
        ],
    )
    K = st.number_input(
        " (#)Summary of prompts to consider", min_value=3, max_value=1000
    )

# Set up the Streamlit app layout
st.title("AIVA")
st.subheader(" Powered by 🦜 LangChain + OpenAI + Streamlit")

# Ask the user to enter their OpenAI API key
API_O = st.sidebar.text_input("API-KEY", type="password")

# Session state storage would be ideal
if API_O:
    # Create an OpenAI instance
    prompt = """You are acting as an investment manager who helps people prepare for retirement or help them with investment needs.
Your goal is to ask the right questions so that you can have the information you need to give concrete advice.
 You should adhere to the following rules:
- start by greetings.
- don't give high level, generic and pass-par-tout advise. If you need to ask more questions inorder to give
personalized and concrete advice, please do so by asking the questions one by one, and keeping track of the responses.
- Don't ask more than ONE question at a time
Once you have all the information you need, give advice in as short as possible paragraph, maximum 5 lines at a time.
- Think of this as an in person conversation you are having, not as text generation. You should refrain from offering
long responses that would bore your audience.
- No response should be more than 200 words long.
- only ask one question in each response.
For example, instead of saying the answer depends on a bunch of factors, ask questions that would give you enough
information to give a concrete response.
You need to have a friendly and accessible tone and at the end of the conversation ask if the answers are clear and
whether the user needs explanation of the terms in your response.
Use bullet points if you have to make a list, only if necessary."""

    llm = OpenAI(temperature=0, openai_api_key=API_O, model_name=MODEL, verbose=False)

    # Create a ConversationEntityMemory object if not already created
    if "entity_memory" not in st.session_state:
        st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=K)

    # Create the ConversationChain object with the specified configuration
    Conversation = ConversationChain(
        llm=llm,
        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        memory=st.session_state.entity_memory,
    )
    Conversation.run(prompt)

else:
    st.sidebar.warning(
        "API key required to try this app.The API key is not stored in any form."
    )
    # st.stop()

# Add a button to start a new chat
st.sidebar.button("New Chat", on_click=new_chat, type="primary")

#### INIT the QA and read the 529.pdf as the only DOCUMENT
qa = load_qa('529.pdf')


# Get the user input
user_input = get_text()

# Generate the output using the ConversationChain object and the user input, and add the input/output to the session
if user_input:
    #### FIST we call the QA retriever to see if the answwer is in the DOC
    result = qa({"question":user_input,  "chat_history": chat_history})
    print('BBBBBBB',result)
    output = result["answer"]
    
    ### ONLY IF THE RESPONSE is unknown, call chatGPT. Curretnly dont know a better way to see if the response is unknown
    if "I don't know" in output or "there is no information"  in output :
        print('CCCC','cALLING CHATGPT')
        output = Conversation.run(input= user_input)
    #output = Conversation.run(input= user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

# Allow to download as well
download_str = []
# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        st.success(st.session_state["generated"][i], icon="🤖")
        st.info(st.session_state["past"][i], icon="🧐")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])

    # Can throw error - requires fix
    download_str = "\n".join(download_str)
    if download_str:
        st.download_button("Download", download_str)

# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
    with st.sidebar.expander(label=f"Conversation-Session:{i}"):
        st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session

