from langchain_community.llms import cohere
import os 
from openai import OpenAI
import streamlit as st
import plotly.express as px
import pandas as pd
from langchain.prompts.prompt import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_cohere import ChatCohere
from google.auth.exceptions import GoogleAuthError
import openai
import cohere


# for reading directly from csv file in the same dir
#current_dir = os.path.dirname(__file__)
#csv_file_path = os.path.join(current_dir, 'Disease_symptom_and_patient_profile_dataset1.csv')
#DiseaseProfile = pd.read_csv(csv_file_path)

# query: find the disease names and the number of occurrences by each disease, among patients below 30 who have normal cholesterol level
# plot bar: find the disease names and the number of occurrences by each disease, among patients below 30 who have normal cholesterol level
# plot pie: find the disease names and the number of occurrences by each disease, among patients below 30 who have normal cholesterol level
def initialize_llm(llm_choice, api_key):
    try:
        if not api_key:
            raise ValueError("API key is missing.")
        
        if llm_choice == 'Cohere':
            co = cohere.Client(api_key=api_key)
            response = co.generate(model='command', prompt="test")
            llm = ChatCohere(model="command", temperature=0, cohere_api_key=api_key)
        elif llm_choice == 'Google Gemini':
            client = ChatGoogleGenerativeAI(api_key=api_key)
            response = client.chat(model="gemini-pro", message="test")
            llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
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
    username = 'yoj'
    password = '28272522Ab'
    hostname = os.getenv('IRIS_HOSTNAME', 'localhost')
    port = '1972'
    namespace = 'TEST03_VISUALISATION'
    CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"

    llm = initialize_llm(llm_choice, api_key)


    _DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.

    Do NOT use backticks, for example, instead of:
     ```sql SELECT * FROM Test.DiseaseProfile ``` or SELECT * FROM Test.DiseaseProfile; ```
     Write it as:
     SELECT * FROM DiseaseProfile

    Do NOT use the word "Count" as a column name, for example, instead of:
     SELECT Disease, Count(*) as Count FROM DiseaseProfile
     Write it as:
     SELECT Disease, Count(*) as AnyOtherWord FROM DiseaseProfile

    The only table available is DiseaseProfile.

    The columns are Disease VARCHAR(512), Fever VARCHAR(25), Cough VARCHAR(25), Fatigue VARCHAR(25), Difficulty_Breathing VARCHAR(25), Age INT, Gender VARCHAR(25), Blood_Pressure VARCHAR(25), Cholesterol_Level VARCHAR(25), Outcome_Variable VARCHAR(25).
    Fever, Cough, Fatigue and Difficulty_Breathing are potential symptoms which the patients are experiencing. 

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

def extract_sql_result(query, api_key, llm_choice):
    output = execute_chain(query, api_key, llm_choice)
    result = output.get('result', None)
    list_result = eval(result)
    return list_result


def generate_response(query, api_key, llm_choice):
    #prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    output = execute_chain(query, api_key, llm_choice)
    for step in output["intermediate_steps"]:
        if isinstance(step, dict) and "sql_cmd" in step:
            sql_query = step["sql_cmd"].split("SQLQuery: ")[-1].split("SQLResult:")[0].strip()
    result = output["intermediate_steps"][-1]  
    # need 2 whitespapces before \n for it to show: https://discuss.streamlit.io/t/st-write-cant-recognise-n-n-in-the-response-but-when-copied-and-used-in-the-prints-with-new-line/52995
    return "SQL Query: " + "  \n" + sql_query + "  \n\nResult: " + "  \n" + result

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
    def submit():
        try:
            print(f"API Key in session state: {st.session_state.api_key}")
            if not st.session_state.api_key:
                st.session_state.api_key = st.session_state.widget
                st.session_state.widget = ""
                st.sidebar.info("API Key submitted")

            # Validate the API key by attempting to initialize the LLM
            api_key = st.session_state.api_key
            print(f"API Key being passed to initialize_llm: {api_key}")
            initialize_llm(llm_choice, api_key) 
        except Exception as e:
                st.sidebar.info(str(e), icon="🚨")

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

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", 
                                      "content": """ 
                                                [BEFORE proceeding, please key in your API Key for the Large Language Model you wish to use in the side panel] \n
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
    
    original_prompt = st.chat_input("Say something")

    if original_prompt is not None:
        prompt = original_prompt.lower()
    else:
        prompt = original_prompt

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        # Generate and display the response
        if prompt.startswith("plot"):
            fig = sql_visualisation(prompt, api_key, llm_choice) 
            st.session_state.messages.append({"role": "assistant", "content": fig}) 
            with st.chat_message("assistant"):
                st.plotly_chart(fig)
        
        elif prompt.startswith("query"):
            response = generate_response(prompt, api_key, llm_choice)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)
        else:
            response = gpt_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)
        
if __name__ == '__main__':
    main()

