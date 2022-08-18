import streamlit as st
import pandas
from utils import create_cosine_sim, get_recommendations

# Set page config
st.set_page_config(
    page_title="Netflix Movies & Series Recommendation",
    page_icon="🎬",
    layout="wide",
)

# hide streamlit default header and footer
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """

st.markdown(hide_menu_style, unsafe_allow_html=True)

# Remove space above the title
padding_top = 0

st.markdown(
    f"""
    <style>
        .reportview-container .main .block-container{{
            padding-top: {padding_top}rem;
        }}
    </style>""",
    unsafe_allow_html=True,
)

# import Data
df = pandas.read_csv(
    "data/clean_data.csv",
    usecols=["title", "description", "genres", "imdb_score", "imdb_votes", "overview"],
)

# Create cosine similarity matrix
cosine_sim, indices = create_cosine_sim(df)

# Set Title
st.title("Netflix Movies & Series Recommendation")

# Movie or Show Title
title = st.selectbox(
    "Enter any movie or show name which you have watched",
    df.sort_values("title")["title"].unique(),
)

# No. of Recommendations
k = st.radio("No. of Recommendations", [5, 10, 15, 20])

clicked = st.button("Get Recommendations")

if clicked:
    st.success("Here are your recommendations for {}".format(title))
    st.table(get_recommendations(df, title, indices, cosine_sim, k))
