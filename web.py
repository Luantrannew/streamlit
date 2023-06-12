import streamlit as st
import pandas as pd
import altair


st.set_page_config(
    page_title = "RECOMMEND FILM SYSTEM",
    page_icon=":tada",
    layout = "wide"
)

st.title("**:blue[FILM RECOMMENDATION SYSTEM]**")
with st.form("thông tin"):
    options =["Title"]
    name = st.selectbox('**:red[Typing the film titles]**', options=options)