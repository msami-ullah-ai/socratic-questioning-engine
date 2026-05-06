import streamlit as st
from src.pipeline import SocraticPipeline

pipeline = SocraticPipeline()

st.title("Socratic Questioning Engine")

q = st.text_area("Question")
ca = st.text_area("Correct Answer")

if "history" not in st.session_state:
    st.session_state.history = []

sa = st.text_input("Student Answer")

if st.button("Submit"):
    result = pipeline.run_turn(q, ca, sa, st.session_state.history)
    st.session_state.history.append((sa, result["socratic_question"]))

    st.write("Tutor:", result["socratic_question"])

    st.sidebar.write(result)