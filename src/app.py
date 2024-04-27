from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama

import streamlit as st

def connect_to_database(user: str, password: str,host: str, port: str, database: str):
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    #return SQLDatabase.from_uri(db_uri)
    return SQLDatabase.from_uri("sqlite:///Chinook.db") #Only for the moment

def get_sql_chain(db):
    template = """You are a SQL expert. Given an input question, create a syntactically correct SQL query to run.
    Return only the query sql without other words. Your output must go directly into input to a db to do the query
    Based on the table schema below, write a SQL query that would answer the user's question. Take che conversation history into account.
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, count(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT3;
    Question: name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:

    Question: {question}
    SQL Query:"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = Ollama(model="duckdb-nsql")

    def get_schema(_):
        schema = db.get_context()
        return schema

    return (
            RunnablePassthrough.assign(schema=get_schema)  # assegna il valore corretto a schema che visualizzeremo nel prompt
            | prompt
            | llm
            | StrOutputParser()
    )

def get_response(user_query:str, db:SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)

    template = '''
    You are a data analyst. You are interacting with a user who is asking you questions about che company's database.
    Based on the table schema below, question, SQL query and SQL response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}'''

    prompt_response = ChatPromptTemplate.from_template(template)

    def run_query(query):
        print("La query che viene usata è: ", query)
        result = db.run(query)
        print("Il risultato della query è: ", result)
        return result

    llm3 = Ollama(model="llama3")
    full_chain = (
    RunnablePassthrough.assign(query=sql_chain)
    .assign(schema=lambda _:db.get_table_info())
    .assign(response=lambda vars: run_query(vars["query"]),
    )
    | prompt_response
    | llm3
)
    return full_chain.invoke({
        "question" : user_query,
        "chat_history" : chat_history,
    })

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content = "Hello! I'm a SQL assistant. Ask me anything about your database")
    ]

st.set_page_config(page_title= "Chat with my Database", page_icon=":speech_balloon:") #layout="wide"
st.title("Chat with my Database")

with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using Database. Connect to the database and start chatting.")
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="5432", key="Port")
    st.text_input("Database", value="Chinook", key="Database")
    st.text_input("User", value="postgres", key="User")
    st.text_input("Password", type= "password", value="admin", key="Password")
    if st.button("Start"):
        with st.spinner(text="Connecting to the database..."):
            db = connect_to_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.database = db #da capire
            st.success("Connected to the database.")

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_question = st.chat_input("Type a message...")
if user_question is not None and user_question.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_question))

    with st.chat_message("Human"):
        st.markdown(user_question)

    with (st.chat_message("AI")):
        response = get_response(user_question, st.session_state.database, st.session_state.chat_history) #da capire
        st.markdown(response)
    st.session_state.chat_history.append(AIMessage(content=response))
    #with st.spinner("Queryng database..."):

