import os
import streamlit as st
from langchain_cohere import ChatCohere
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate
import sqlalchemy as db
import pandas as pd

import plotly.express as px

# Find the disease names and each of the disease's number of occurrences among patients below 30 who have normal cholesterol level
# Plot bar/pie: Find the disease names and each of the disease's number of occurrences among patients below 30 who have normal cholesterol level
# Plot pie: Find the male to female ratio

def execute_db_chain(query, api_key):
    #Prompttemplate
    _DEFAULT_TEMPLATE = """

    Given an input question, first create a syntactically correct {dialect} query to run not including sql ```, then look at the results of the query and return the answer.

    "COUNT" is a reserve word, DO NOT use it as a name for SQL query; For example, instead of:
    SELECT COUNT(attrb) as COUNT 
    Write it as:
    SELECT COUNT(attrb) as AnyOtherName 


    Do NOT use backticks, for example, instead of:
     ```sql SELECT * FROM DiseaseProfile ``` or SELECT * FROM DiseaseProfile; ```
     Write it as:
     SELECT * FROM DiseaseProfile

     Do NOT use "DESC LIMIT 10;" for the SQL queries.

    The only table available is DiseaseProfile.

    The columns are Disease VARCHAR(512), Fever VARCHAR(25), Cough VARCHAR(25), Fatigue VARCHAR(25), DifficultyBreathing VARCHAR(25), Age INT, Gender VARCHAR(25), BloodPressure VARCHAR(25), CholesterolLevel VARCHAR(25), OutcomeVariable VARCHAR(25).
    Fever, Cough, Fatigue and DifficultyBreathing are potential symptoms which the patients are experiencing. 

    Columns and Usage:

    Disease: The name of the disease or medical condition.
    Fever: Indicates whether the patient has a fever (Yes/No).
    Cough: Indicates whether the patient has a cough (Yes/No).
    Fatigue: Indicates whether the patient experiences fatigue (Yes/No).
    Difficulty Breathing: Indicates whether the patient has difficulty breathing (Yes/No).
    Age: The age of the patient in years.
    Gender: The gender of the patient (Male/Female).
    Blood Pressure: The blood pressure level of the patient (Normal/High).
    Cholesterol Level: The cholesterol level of the patient (Normal/High).
    Outcome Variable: The outcome variable indicating the result of the diagnosis or assessment for the specific disease (Positive/Negative).

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

    username = 'superuser'
    password = 'sys'
    hostname = os.getenv('IRIS_HOSTNAME', 'localhost')
    port = '1972'
    namespace = 'TEST'
    CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"

    #LLM
    # llm=ChatCohere(model="command", temperature=0, cohere_api_key = st.secrets["COHERE_API_KEY"])
    llm=ChatCohere(model="command", temperature=0, cohere_api_key = api_key)

    db_sql = SQLDatabase.from_uri(CONNECTION_STRING) 

    db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db_sql, return_direct=True, return_intermediate_steps=True, prompt=PROMPT, verbose=True) 
    return db_chain.invoke(query)

def extract_sql_result(query, api_key):
    output = execute_db_chain(query, api_key)
    result = output.get('result', None)
    list_result = eval(result)
    return list_result

def get_visualisation(data, visualization_type):
    df = pd.DataFrame(data)
    
    if visualization_type == "bar":
        fig = px.bar(df, x=df.columns[0], y=df.columns[1], title="Bar Chart")
    
    elif visualization_type == "hist":
        fig = px.histogram(df, x=df.columns[1], title="Histogram")
    
    elif visualization_type == "box":
        fig = px.box(df, y=df.columns[1], title="Box Plot")
    
    elif visualization_type == "pie":
        fig = px.pie(df, names=df.columns[0], values=df.columns[1], title="Pie Chart")
    return fig

def sql_visualisation(query, visualization_type, api_key):
    result = extract_sql_result(query, api_key)
    fig = get_visualisation(result, visualization_type)
    return fig

def get_response(query, api_key):
    output = execute_db_chain(query, api_key)
    for step in output["intermediate_steps"]:
        if isinstance(step, dict) and "sql_cmd" in step:
            sql_query = step["sql_cmd"].split("SQLQuery: ")[-1].split("SQLResult:")[0].strip()
    result = output["intermediate_steps"][-1]  
    # need 2 whitespapces before \n for it to show: https://discuss.streamlit.io/t/st-write-cant-recognise-n-n-in-the-response-but-when-copied-and-used-in-the-prints-with-new-line/52995
    return "SQL Query: " + "  \n" + sql_query + "  \n\nResult: " + "  \n" + result
    # return {"SQL Query": sql_query, "Result": result}

def main():
    # st.set_page_config(page_title="Chatbot🐨")
    st.title("SQLbot🐨")

    selected_model = st.sidebar.radio("Select Language Model", ("Cohere", "Others"))

    if selected_model == "Cohere":      
        
        def clear_text():
            st.session_state["api"] = ""
            st.sidebar.info("API Key submitted")

        api_key = st.sidebar.text_input("Enter API Key:", type="password", key="api", on_change=clear_text)
    else:
        st.sidebar.write("Select a model to see the API key textbox")

    user_question =  st.chat_input("Ask a Question")

    if "messages" not in st.session_state.keys():
        st.session_state["messages"] = [{"role": "assistant",
                                         "content": """
                                                    Hello there, how can I help you?  \n
                                                    [STILL TESTING] To plot a bar graph: Start your question with the keyword "Plot bar:"  \n
                                                    [STILL TESTING] To plot a pie chart: Start your question with the keyword "Plot pie:" 
                                                    """
                                        }]

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
                if user_question.startswith("Plot bar"):
                    visualization_type = "bar"  # Default visualization type FOR NOW
                    output = sql_visualisation(user_question, visualization_type, api_key)
                    ai_response = output
                    st.plotly_chart(ai_response)

                elif user_question.startswith("Plot box"):
                    visualization_type = "box" 
                    output = sql_visualisation(user_question, visualization_type, api_key)
                    ai_response = output
                    st.plotly_chart(ai_response)

                elif user_question.startswith("Plot pie"):
                    visualization_type = "pie" 
                    output = sql_visualisation(user_question, visualization_type, api_key)
                    ai_response = output
                    st.plotly_chart(ai_response)

                else:
                    output = get_response(user_question, api_key)
                    ai_response = output
                    st.success(ai_response)

        new_ai_message = {"role":"assistant","content": ai_response}
        st.session_state.messages.append(new_ai_message)


if __name__ == '__main__':
    main()