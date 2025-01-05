import streamlit as st
import pandas as pd
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai import SmartDatalake
from pandasai.llm import BambooLLM
from pandasai import Agent
from pandasai.responses.streamlit_response import StreamlitResponse
import os
from langchain_openai import ChatOpenAI
import openai
import builtins 
from PIL import Image

load_dotenv(dotenv_path='.env')

data = {}

def main():
    im = Image.open("S-satelix.ico")
    st.set_page_config(page_title="Satelix", 
                       page_icon=im,
                       layout="wide")

    st.title("Satelix Data Insights ✨")

    password = st.sidebar.text_input("Entrez le mot de passe", type="password")

    if password == st.secrets["pwd"]:

        with st.sidebar:
            st.image("Satelix1.png", width=250)
            st.title("⚙️ Configuration")
            st.text("📝 Data")
            file_upload = st.file_uploader("Uploader votre fichier",
                                        accept_multiple_files=False,
                                        type=['csv', 'xls', 'xlsx'])

            st.markdown("⚠️ :green[*Assurez-vous que la 1ère ligne du fichier \
                                    contient les noms des colonnes.*]")

            llm_type = st.selectbox(
                "🤖 Choix du LLM",
                ('gemini-1.5-flash', 'gpt-4o-mini'), index=0)


        if file_upload is not None:
            try:
                data = extract_dataframes(file_upload)
                df = st.selectbox("Données téléchargées !",
                                tuple(data.keys()), index=0)
                st.dataframe(data[df])

                llm = get_LLM(llm_type, os.getenv('GOOGLE_API_KEY'))

                if llm:
                    analyst = get_agent(data, llm)
                    chat_window(analyst)
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.warning("Assurez-vous que vos données ne contiennent pas des formats complexes de dates")
        else:
            st.warning("⬅️ Téléchargez dans un 1er temps vos données (.csv ou .xlsx) !")

def get_LLM(llm_type,user_api_key):
    try:
        # elif llm_type =='gemini-1.5-flash':
        if llm_type =='gemini-1.5-flash':
            genai.configure(api_key= st.secrets["OPENAI_API_KEY"])
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", 
                                         temperature=0.3,
                                         google_api_key = st.secrets["GOOGLE_API_KEY"])
        elif llm_type == 'gpt-4o-mini':
            openai.api_key = st.secrets["OPENAI_API_KEY"]
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3,
                api_key= st.secrets["OPENAI_API_KEY"]
            )
        return llm
    except Exception as e:
        st.error("No/Incorrect API key provided! Please Provide/Verify your API key")

def chat_window(analyst):
    with st.chat_message("assistant"):
        st.text("Explorer vos données avec Satelix LLM Insights ?🧐")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if 'question' in message:
                st.markdown(message["question"])
            elif 'response' in message:
                st.write(message['response'])
            elif 'error' in message:
                st.text(message['error'])

    user_question = st.chat_input("Que souhaitez-vous savoir ? ")
    
    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role":"user","question":user_question})
       
        try:
            with st.spinner("Analyzing..."):
                response = analyst.chat(user_question)
                st.write(response)
                st.session_state.messages.append({"role":"assistant","response":response})
        
        except Exception as e:
            error_message = f" Error: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({"role":"assistant","error":error_message})
            
    def clear_chat_history():
        st.session_state.messages = []

    st.sidebar.text("Cliquez pour effacer l'historique du Chat !")
    st.sidebar.button("CLEAR 🗑️",on_click=clear_chat_history)
        
def get_agent(data, llm):
    configs = {
        "llm": llm,
        "verbose": True,
        "response_parser": StreamlitResponse,
        "custom_whitelisted_dependencies": [
            "pandas", "numpy", "matplotlib", "seaborn", "plotly.express",
            "sklearn", "scipy", "datetime", "collections", "json", "re", "os", 
            "builtins", "io"
        ],
        "error_handler": {
            "handle_errors": True,
            "max_retries": 3
        }
    }
    
    agent = Agent(list(data.values()), config=configs)
    return agent

def extract_dataframes(raw_file):
    try:
        dfs = {}
        if raw_file.name.split('.')[1].lower() == 'csv':
            csv_name = raw_file.name.split('.')[0]
            df = pd.read_csv(raw_file, parse_dates=True)
            for col in df.select_dtypes(include=['datetime64']).columns:
                df[col] = df[col].astype(str)
            dfs[csv_name] = df

        elif raw_file.name.split('.')[1].lower() in ['xlsx', 'xls']:
            xls = pd.ExcelFile(raw_file)
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(raw_file, sheet_name=sheet_name, parse_dates=True)
                for col in df.select_dtypes(include=['datetime64']).columns:
                    df[col] = df[col].astype(str)
                dfs[sheet_name] = df

        return dfs
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        raise

if __name__ == "__main__":
    main()