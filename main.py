import streamlit as st

from first_page import *
from second_page import *
import os

os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "AI504"

def main():
    if "student_id" not in st.session_state:
        first_page()
    else:
        second_page()

if __name__ == "__main__":
    main()