import streamlit as st

def first_page():
    st.image("./asset/kaist.png", width=300)
    st.title("KAIST AI504 Virtual TA")

    st.markdown("""
    ### About this Virtual TA
    This Virtual Teaching Assistant is powered by the GPT-4 API. It provides answers based on AI 504 course materials, syllabus, and class transcriptions.
    
    **Important Notice:**
    - This tool is exclusively for the AI504 course. **Do not** use it for any other purposes.
    - There is a rate limit on GPT-4 usage. Please be mindful of your usage to ensure that all students have an equal opportunity to benefit from this tool.
    - **Student IDs found to be using this tool for purposes other than for the AI 504 course, or with abnormally high usage, may have their access revoked.**
    - Conversations with the Virtual TA will be stored and can be used for research purposes. However, your student ID will be thoroughly anonymized. **Do not** include any identifying information in your conversations.
    - Since the model may hallucinate, for matters directly related to grades (e.g., project submission deadlines), be sure to check the relevant documents directly or contact the TA.
    - By using this Virtual TA, you agree to these terms and conditions.
    """)

    # Contact Info
    st.markdown("""
    **Contact Info:**
    - If you have any questions or need assistance, please contact: sean0042@kaist.ac.kr
    - This program was developed by [KAIST Edlab](http://mp2893.com).
    """)

    # Agreement Checkbox
    agreement = st.checkbox("I agree to the terms and conditions stated above.")
    
    student_id = st.text_input("Submit your Student ID to get started!")

    if st.button("Submit"):
        if agreement:
            if student_id in st.secrets["student_ids"]:
                st.session_state["student_id"] = student_id
                st.rerun()
            else:
                st.error("Invalid Student ID. Please try again.")
        else:
            st.error("You must agree to the terms and conditions before proceeding.")


