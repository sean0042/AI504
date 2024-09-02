import streamlit as st
from home_page import home_page
from main_page import main_page
import os

os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "tmp"

def main():
    if "student_id" not in st.session_state:
        home_page()
    else:
        main_page()

if __name__ == "__main__":
    main()