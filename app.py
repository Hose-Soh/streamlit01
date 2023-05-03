import json
from datetime import datetime, timedelta, date

import ee
import geemap.colormaps as cm
import geemap.foliumap as geemap
import streamlit as st

# ______ GEE Authenthication ______

# _____ STREAMLIT _______

# Secrets
json_data = st.secrets["json_data"]
service_account = st.secrets["service_account"]

# Preparing values
json_object = json.loads(json_data, strict=False)
json_object = json.dumps(json_object)

# Authorising the app
credentials = ee.ServiceAccountCredentials(service_account, key_data=json_object)
ee.Initialize(credentials)

# _______________________ LAYOUT CONFIGURATION __________________________

st.set_page_config(page_title='SST monitor', layout="wide")

# shape the map
st.markdown(
        f"""
<style>
    .appview-container .main .block-container{{

        padding-top: {3}rem;
        padding-right: {2}rem;
        padding-left: {0}rem;
        padding-bottom: {0}rem;
    }}


</style>
""",
        unsafe_allow_html=True,
    )

#_________________________Importing Libraries______________________

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import folium
import pprint
import branca.colormap as cm

#__________________________Input Parameters________________________


# Define the date range slider
i_date = st.date_input("Initial date of interest (inclusive)", min_value=datetime.strptime('1992-10-02', '%Y-%m-%d'), max_value = datetime.now())
f_date = st.date_input("Final date of interest (exclusive)", min_value=datetime.strptime('1992-10-02', '%Y-%m-%d'), max_value = datetime.now())

# Take input from user for lon and lat
lon = st.number_input("Enter the longitude", value=5.145041)
lat = st.number_input("Enter the latitude", value=45.772439)
poi = ee.Geometry.Point(lon, lat)

# A nominal scale in meters of the projection to work in [in meters].
scale = 1000