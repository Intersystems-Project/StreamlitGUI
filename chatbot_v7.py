# RAG version

from langchain_community.llms import cohere
import os 
import streamlit as st
import plotly.express as px
import pandas as pd
from langchain.prompts.prompt import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_cohere import ChatCohere, CohereEmbeddings
#from pretty_notification_box import notification_box
import cohere
import openai
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_iris import IRISVector
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import sqlalchemy as db
import re

username = 'yoj'
password = '28272522Ab'
hostname = os.getenv('IRIS_HOSTNAME', 'localhost')
port = '1972'
namespace = 'SILOAM_DATASET'
CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"

def text_loader(llm_choice, api_key):
    loader = TextLoader("factsheet_example.txt", encoding='utf-8')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=5)
    docs = text_splitter.split_documents(documents)
    if llm_choice == 'Cohere':
        os.environ["COHERE_API_KEY"] = api_key
        embeddings = CohereEmbeddings(model="embed-english-v2.0")
    elif llm_choice == 'Google Gemini':
        os.environ["GOOGLE_API_KEY"] = api_key
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    COLLECTION_NAME = "factsheet_test"

    engine = db.create_engine(CONNECTION_STRING)
    connection = engine.connect()
    metadata = db.MetaData()
    metadata.reflect(bind=engine)
    if COLLECTION_NAME in metadata.tables:
        db_rag = IRISVector(
            embedding_function=embeddings, 
            connection_string=CONNECTION_STRING, 
            collection_name=COLLECTION_NAME
        )
    else:
        db_rag = IRISVector.from_documents(
            embedding=embeddings,
            documents=docs,
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING,
        )
    connection.close()
    retriever = db_rag.as_retriever()
    return retriever
# query: find the disease names and the number of occurrences by each disease, among patients below 30 who have normal cholesterol level
# plot bar: find the disease names and the number of occurrences by each disease, among patients below 30 who have normal cholesterol level
# plot pie: find the disease names and the number of occurrences by each disease, among patients below 30 who have normal cholesterol level

def initialize_llm(llm_choice, api_key):
    try:
        if not api_key:
            raise ValueError("API key is missing.")
        
        if llm_choice == 'Cohere':
            llm = ChatCohere(model="command", temperature=0, cohere_api_key=api_key)
            #llm.invoke("test llm")
        elif llm_choice == 'Google Gemini':
            llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
            #llm.invoke("test llm")
        elif llm_choice == 'OpenAI':
            openai.api_key = api_key
            response = openai.Model.list()
            llm = openai.ChatCompletion()
        else:
            llm = None
        return llm
    except Exception as e:
        raise Exception(f"API Key error: {e}")
    
def execute_chain(query, api_key, llm_choice):

    llm = initialize_llm(llm_choice, api_key)

    _DEFAULT_TEMPLATE = """Answer should ONLY be the result interpreted in natural language. Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
    Follow STRICTLY to the instructions here.
    The table name is DiseaseProfile.

    Do NOT use "DESC" in the SQLQuery.

    The SQL QUERY should NOT contain any backticks (`).
    an example of a correctly generated sql query response is: 
    SELECT Disease, Count(*) AS Number FROM DiseaseProfile GROUP By Disease

    Do NOT end the query with "DESC LIMIT".
    Do NOT end the query with backticks (```), 
    Do NOT enclose the query with backticks (```)
    for example, a wrong response:
     ```SELECT * FROM Test.DiseaseProfile ``` or SELECT * FROM Test.DiseaseProfile; ```
     an example of the right response is:
     SELECT * FROM DiseaseProfile

    Do NOT use the word "Count" as a column name, for example, instead of:
     SELECT Disease, Count(*) AS Count FROM DiseaseProfile GROUP By Disease
     Write it as:
     SELECT Disease, Count(*) AS Number FROM DiseaseProfile GROUP By Disease

    Use the following format:

    Question: "Question here"
    SQLQuery: "SQL Query to run"
    SQLResult: "Result of the SQLQuery"
    Answer: "Final answer in complete English sentences here. This should be answering the query"

    The SQL query should NOT end with semi-colon (;) 
    
    
    Question: {input}"""

    PROMPT = PromptTemplate(
        input_variables=["input", "dialect"], template=_DEFAULT_TEMPLATE
    )

    db_sql = SQLDatabase.from_uri(CONNECTION_STRING) 
    db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db_sql,  return_intermediate_steps=True, prompt=PROMPT, verbose=False) 
    #return_direct=True, return_intermediate_steps=True, 
    retriever = text_loader(llm_choice, api_key)

    template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | db_chain
    )
    return retrieval_chain.invoke(query)

def get_original_result(query, api_key, llm_choice):
    output = execute_chain(query, api_key, llm_choice)
    nl_response = output['result']
    for step in output["intermediate_steps"]:
        if isinstance(step, dict) and "sql_cmd" in step:
            sql_query = step["sql_cmd"].split("SQLQuery: ")[-1].split("SQLResult:")[0].strip()
    engine = db.create_engine(CONNECTION_STRING)
    connection = engine.connect()
    result = connection.exec_driver_sql(sql_query)
    connection.close()
    return nl_response, sql_query, result

def extract_sql_result(query, api_key, llm_choice):
    _, _, result = get_original_result(query, api_key, llm_choice)
    list_result = list(result)
    return list_result

def generate_response(query, api_key, llm_choice):
    nl_response, sql_query, result = get_original_result(query, api_key, llm_choice)
    result_string = str(result.fetchall())
    # # need 2 whitespapces before \n for it to show: https://discuss.streamlit.io/t/st-write-cant-recognise-n-n-in-the-response-but-when-copied-and-used-in-the-prints-with-new-line/52995
    return nl_response + "  \n\nSQL Query: " + "  \n" + sql_query + "  \n\nResult: " + "  \n" + result_string

def generate_visualization(data, visualization_type):
    df = pd.DataFrame(data)

    # more visualisation types can be added if necessary
    if visualization_type == "bar":
        fig = px.bar(df, x=df.columns[0], y=df.columns[1], title="Bar Chart")
    
    elif visualization_type == "hist" :
        fig = px.histogram(df, x=df.columns[1], title="Histogram")
    
    elif visualization_type == "box":
        fig = px.box(df, y=df.columns[1], title="Box Plot")
    
    elif visualization_type == "pie":
        fig = px.pie(df, names=df.columns[0], values=df.columns[1], title="Pie Chart")
    return fig

def sql_visualisation(query, api_key, llm_choice):
    result = extract_sql_result(query, api_key, llm_choice)
    if "bar" in query.lower():
        fig = generate_visualization(result, "bar")
    elif "pie" in query.lower():
        fig = generate_visualization(result, "pie")
    elif "box" in query.lower():
        fig = generate_visualization(result, "box")
    elif ("hist" or "histogram") in query.lower():
        fig = generate_visualization(result, "hist")   
    
    return fig

# streamlit
def main():

    # This is to prevent the notiffrom appearing every time a question is asked 
    if "reminder_shown" not in st.session_state:
        st.session_state["reminder_shown"] = False

    if not st.session_state["reminder_shown"]:
        @st.experimental_dialog("📢IMPORTANT")
        def reminder():
            st.write('Before proceeding, please key in your API Key for the LLM API you wish to use in the side panel')
        reminder()
        st.session_state["reminder_shown"] = True

    def submit():
        if not st.session_state.api_key:
            st.session_state.api_key = st.session_state.widget
            st.session_state.widget = ""
            st.sidebar.info("API Key submitted")
        try:
            initialize_llm(llm_choice, st.session_state.api_key) 
        except Exception as e:
                st.sidebar.error(str(e), icon="🚨")

    st.title('👩‍⚕🧬Siloam🩺💉')

    with st.sidebar:
        st.title('APIs')

        if "api_key" not in st.session_state:
                st.session_state.api_key = ""
        
        llm_choice = st.selectbox('Choose LLM', ['Cohere', 'Google Gemini', 'OpenAI'])
        if llm_choice == 'Cohere':
            st.text_input('Cohere API Key', type='password', key="widget", on_change=submit)
            api_key = st.session_state.api_key

        elif llm_choice == 'Google Gemini':
            st.text_input('Google Gemini API Key', type='password', key="widget", on_change=submit)
            api_key = st.session_state.api_key

        elif llm_choice == 'OpenAI':
            st.text_input('OpenAI API Key', type='password', key="widget", on_change=submit)
            api_key = st.session_state.api_key

    def gpt_response(message):
        co = cohere.Client(api_key)
        response = co.generate(
            prompt=message,
        )
        return response.generations[0].text.strip()
    
    user_question =  st.chat_input("Ask a Question")

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", 
                                      "content": """
                                                 How may I help you? \n
                                                 To get sql query: start your question with "query"! 
                                                 (i.e. query: find the disease names and the number of occurrences by each disease, 
                                                 among patients below 30 who have normal cholesterol level) \n
                                                 To visualise and plot graphs: start your question with "plot"! 
                                                 (i.e. plot bar / pie: find the disease names and the number of occurrences by each disease, 
                                                 among patients below 30 who have normal cholesterol level)
                                                 """}]
    if "messages" in st.session_state.keys():
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    if user_question is not None:
        st.session_state.messages.append({
            "role":"user",
            "content":user_question
        })

        with st.chat_message("user"):
            st.write(user_question)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Loading"):
                if user_question.startswith("plot"):
                    ai_response = sql_visualisation(user_question, api_key, llm_choice)
                    st.plotly_chart(ai_response)
                elif user_question.startswith("query"):
                    ai_response = generate_response(user_question, api_key, llm_choice)
                    st.write(ai_response)
                elif llm_choice == 'Cohere' and not user_question.startswith("query") and not user_question.startswith("plot"):
                    ai_response = gpt_response(user_question)
                    st.write(ai_response)

        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        
if __name__ == '__main__':
    main()

