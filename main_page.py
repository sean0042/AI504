import streamlit as st
from chains import get_vector_store, get_retreiver_chain, get_conversational_rag
from langchain_core.messages import ChatMessage,HumanMessage,AIMessage
from langchain_core.tracers.context import collect_runs
from langsmith import Client
from langsmith import traceable
from streamlit_feedback import streamlit_feedback
client = Client()

def main_page():
    st.header("AI504 Chatbot")

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Go to Home", key="home_page"):

            st.session_state.pop("student_id", None)
            st.session_state.pop("chat_history", None)

            st.rerun()

    with col2:
        if st.button("Refresh", key="refresh"):
            st.session_state.pop("chat_history", None)

            st.rerun()


    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []   
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vector_store()

    for message in st.session_state.chat_history:
        if isinstance(message,AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        else:
            with st.chat_message("Human"):
                st.write(message.content)


    def get_response(user_input):
        history_retriever_chain = get_retreiver_chain(st.session_state.vector_store)
        conversation_rag_chain = get_conversational_rag(history_retriever_chain)
        response = conversation_rag_chain.invoke({
            "chat_history":st.session_state.chat_history,
            "input":user_input,
            "student_id" : "20218179"
        })
        return response["answer"]

    if user_input := st.chat_input("Type your message here..."):
        st.chat_message("Human").write(f"{user_input}")
        
        with collect_runs() as cb:
            with st.spinner("Thinking..."):
                response = get_response(user_input)
                st.chat_message("AI").write(f"{response}")

                st.session_state.chat_history.append(HumanMessage(content=user_input))
                st.session_state.chat_history.append(AIMessage(content=response))
            st.session_state.run_id = cb.traced_runs[0].id

    feedback_option = "thumbs"
    if st.session_state.get("run_id"):
        run_id = st.session_state.run_id
        feedback = streamlit_feedback(
            feedback_type="thumbs",
            optional_text_label="[Optional] Please provide an explanation",
            key=f"feedback_{run_id}",
        )

        # Define score mappings for both "thumbs" and "faces" feedback systems
        score_mappings = {
            "thumbs": {"👍": 1, "👎": 0},
            "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
        }

        # Get the score mapping based on the selected feedback option
        scores = score_mappings[feedback_option]

        if feedback:
            # Get the score from the selected feedback option's score mapping
            score = scores.get(feedback["score"])

            if score is not None:
                # Formulate feedback type string incorporating the feedback option
                # and score value
                feedback_type_str = f"{feedback_option} {feedback['score']}"

                # Record the feedback with the formulated feedback type string
                # and optional comment
                feedback_record = client.create_feedback(
                    run_id,
                    feedback_type_str,
                    score=score,
                    comment=feedback.get("text"),
                )
                st.session_state.feedback = {
                    "feedback_id": str(feedback_record.id),
                    "score": score,
                }
            else:
                st.warning("Invalid feedback score.")