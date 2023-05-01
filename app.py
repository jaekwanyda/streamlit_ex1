import streamlit as st

import io
import os

import pandas as pd
from inference import csv_inference


# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")
st.title("STS Score for csv file")

uploaded_file = st.file_uploader("Choose an csv file", type=["csv"])


if uploaded_file:
    test_file = pd.read_csv(uploaded_file, encoding='utf-8')
    columns=test_file.columns
    if 'sentence_1' not in columns or 'sentence_2' not in columns or 'label' not in columns:
        st.warning('The CSV file is not in the correct format.')
        
    predicts_csv=csv_inference(test_file)
    st.download_button(
    "Press to Download",
    predicts_csv,
    "file.csv",
    "text/csv",
    )