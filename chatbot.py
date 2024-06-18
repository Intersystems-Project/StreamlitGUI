from langchain.llms import cohere
import os 
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
import plotly.express as px
import pandas as pd
from langchain.prompts.prompt import PromptTemplate

from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from pandasai import SmartDataframe

from langchain_cohere import ChatCohere
import sqlalchemy as db
import cohere

# for reading directly from csv file in the same dir
#current_dir = os.path.dirname(__file__)
#csv_file_path = os.path.join(current_dir, 'Disease_symptom_and_patient_profile_dataset1.csv')
#DiseaseProfile = pd.read_csv(csv_file_path)

def execute_chain(query, api_key):
    username = 'yoj'
    password = '28272522Ab'
    hostname = os.getenv('IRIS_HOSTNAME', 'localhost')
    port = '1972'
    namespace = 'TEST03_VISUALISATION'
    CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"

    engine = db.create_engine("iris://yoj:28272522Ab@localhost:1972/TEST03_VISUALISATION")
    connection = engine.connect()
    DiseaseProfile = pd.read_sql_table('DiseaseProfile', connection, schema="SQLUser")

    #cohere_api_key = "mD1SGFkiLf0RlAzUBJGn7uIPbdK95sCp2ys1eGWL"
    llm = ChatCohere(model="command",temperature=0, cohere_api_key=api_key)
    smart_df = SmartDataframe(DiseaseProfile, name="DiseaseProfile", description="Dataset used to generate SQL query", config={"llm": llm})

    _DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.

    The only table available is DiseaseProfile.

    The columns are Disease VARCHAR(512), Fever VARCHAR(25), Cough VARCHAR(25), Fatigue VARCHAR(25), Difficulty_Breathing VARCHAR(25), Age INT, Gender VARCHAR(25), Blood_Pressure VARCHAR(25), Cholesterol_Level VARCHAR(25), Outcome_Variable VARCHAR(25).
    Fever, Cough, Fatigue and DifficultyBreathing are potential symptoms which the patients are experiencing. 

    Columns and Usage:

    Disease: The name of the disease or medical condition.
    Fever: Indicates whether the patient has a fever (Yes/No).
    Cough: Indicates whether the patient has a cough (Yes/No).
    Fatigue: Indicates whether the patient experiences fatigue (Yes/No).
    Difficulty_Breathing: Indicates whether the patient has difficulty breathing (Yes/No).
    Age: The age of the patient in years.
    Gender: The gender of the patient (Male/Female).
    Blood_Pressure: The blood pressure level of the patient (Normal/High).
    Cholesterol_Level: The cholesterol level of the patient (Normal/High).
    Outcome_Variable: The outcome variable indicating the result of the diagnosis or assessment for the specific disease (Positive/Negative).

    Use the following format:

    Question: "Question here"
    SQLQuery: "SQL Query to run"
    SQLResult: "Result of the SQLQuery"
    Answer: "Final answer here"

    The SQL query should NOT end with semi-colon
    Question: {input}"""

    PROMPT = PromptTemplate(
        input_variables=["input", "dialect"], template=_DEFAULT_TEMPLATE
    )

    db_sql = SQLDatabase.from_uri(CONNECTION_STRING) 

    db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db_sql, return_direct=True, return_intermediate_steps=True, prompt=PROMPT, verbose=True) 
    return db_chain(query)

def extract_sql_result(query, api_key):
    output = execute_chain(query, api_key)
    result = output.get('result', None)
    list_result = eval(result)
    return list_result

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

def sql_visualisation(query, api_key):
    result = extract_sql_result(query, api_key)
    if "bar" in query.lower():
        fig = generate_visualization(result, "bar")
    elif "pie" in query.lower():
        fig = generate_visualization(result, "pie")
    elif "box" in query.lower():
        fig = generate_visualization(result, "box")
    elif ("hist" or "histgram") in query.lower():
        fig = generate_visualization(result, "hist")   
    
    return fig

# streamlit
def main():
    def submit():
            st.session_state.api_key = st.session_state.widget
            st.session_state.widget = ""
            st.sidebar.info("API Key submitted")

    st.title('👩‍⚕🧬Siloam🩺💉')
    with st.sidebar:
        st.title('APIs')
        llm_choice = st.selectbox('Choose LLM', ['Cohere', 'OpenAI'])
        if llm_choice == 'Cohere':
            st.text_input('Cohere API Key', type='password', key="widget", on_change=submit)
            api_key = st.session_state.api_key
            if "api_key" not in st.session_state:
                st.session_state.api_key = ""
            #llm = ChatCohere(model="command", temperature=0, cohere_api_key=cohere_api_key)
        elif llm_choice == 'OpenAI':
            st.text_input('OpenAI API Key', type='password', key="widget", on_change=submit)
            api_key = st.session_state.api_key
            if "api_key" not in st.session_state:
                st.session_state.api_key = ""

    def generate_response(message):
        #prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        co = cohere.Client(api_key)
        response = co.generate(
            prompt=message,
        )
        return response.generations[0].text.strip()

    if llm_choice == 'Cohere':
        #llm = cohere.Client(cohere_api_key)
        llm = ChatCohere(model="command", temperature=0, cohere_api_key=api_key)
    elif llm_choice == 'OpenAI':
        import openai
        openai.api_key = api_key

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", 
                                      "content": """ 
                                                 How may I help you? \n
                                                 To visualise and plot graphs: start your question with "plot"! """}]
    if "messages" in st.session_state.keys():
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    prompt = st.chat_input("Say something")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        # Generate and display the response
        if "plot" in prompt.lower():

            fig = sql_visualisation(prompt, api_key) 
            st.session_state.messages.append({"role": "assistant", "content": fig}) 
            with st.chat_message("assistant"):
                st.plotly_chart(fig)
        
        else:
            response = generate_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)
        
if __name__ == '__main__':
    main()

