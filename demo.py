import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from PIL import Image
#import plotly.figure_factory as ff
import time
import visualization

##### image
image = Image.open('ff.jpg')

st.title('DataTalks')
st.write("Here powerful and interactive graphics will help you to precisely analyse :sunglasses:")
st.image(image, caption=None, width=20, use_column_width=True, clamp=False, channels='RGB', format='JPEG')


st.selectbox("Which visualization?",
                        ("pie-chart","scatter chart","line charts"))


st.title('line chart')


chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

st.title('Scatter chart')
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)






#pages = ["Home", "nalytics"]
#tags = ["Awesome", "Social"]

#page = st.sidebar.radio("Navigate", options=pages)
#st.title(page)
#if page == "Resources":
#    selection = st.multiselect("Select tag", tags)
#    st.write(selection)


################# navigations ##########################

page = st.sidebar.selectbox("Choose a page", ["Homepage", "Visualization", "Gallery"])

if page == "Homepage":
    st.header("This is your data explorer.")
    st.write("Please select a page on the left.")
    
elif page == "Visualization":
    visualization.get_visualization_page()
elif page == "Gallery":
    st.header("This is your Gallery.")
    st.write("Please select a page on the left.")
    st.write(df)    
st.title("survey results :sunglasses:")


st.sidebar.title("Movie Rating Survey :smile:")


Movie = st.sidebar.multiselect("Which do you like the most?",
                            ("Avengers","The Golden Compass","Harry Potter"))


director = st.sidebar.multiselect("Who is you fav director?",
                            ("Rv","Av","MP"))


zonors = st.sidebar.multiselect("which topic you love",
                            ("Action","Horror","Thriller"))


st.write("{} is your favourite type of movie".format(', '.join(Movie)))
st.write("{} is your favourite type of director".format(', '.join(director)))




slider_ph = st.empty()
info_ph = st.empty()

value = st.sidebar.slider("Ratings", 0, 5, 1, 1)








    





  