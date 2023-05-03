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


## ___________________ SIDEBAR PARAMETERS ___________________________

st.sidebar.info('### ***Welcome***\n###### ***Sea Surface Temperature (SST) monitor*** 🧐🌊🌡️')

form = st.sidebar.form('Ocean data')

with form:

    # depths in slider
    depth_options = [0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 125, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000]

    depth = st.select_slider('Depth (m)', options = depth_options, value = 0, help = 'Depth in ocean to fetch temperature layer' )

    # dates of sst
    ocean_date = st.date_input('Date', min_value=datetime.strptime('1992-10-02', '%Y-%m-%d'), max_value = datetime.now(), help = 'Selected date of temperature to be displayed')

    # conditions to get the available layer 2 days before today
    if str(ocean_date) == str(date.today()):
        ocean_date = ocean_date - timedelta(2)

    if str(ocean_date) == str(date.today() - timedelta(1)):
        ocean_date = ocean_date - timedelta(1)

    # visualization threshold
    min, max = st.slider('Min and Max (°C)', 0, 40, value=(10, 32), help='Threshold of visualization in Celsius')

    # button to update visualization
    update_depth = st.form_submit_button('Update')


# __________________ MAP INSTANCE _________________

# add a map instance
Map = geemap.Map(zoom=3, center=(-10, -55))

Map.add_basemap('Esri.OceanBasemap') # "HYBRID"

# get the layer with current date
sst_thumb = ee.ImageCollection('HYCOM/sea_temp_salinity').filterDate(str(ocean_date)) #('2022-01-10', '2022-01-15')

# get fist date just in case, and select the depth, and transform the values
image = sst_thumb.limit(1, 'system:time_start', False).first().select(f'water_temp_{depth}').multiply(0.001).add(20)

vis_param = {'min': min,
             'max': max, 'alpha': 0.4, 
             'palette': cm.palettes.jet,
             }

# add image
Map.addLayer(image, vis_param)

# add color bar with depth and date info
Map.add_colorbar(vis_param, label = f'Sea Surface Temperature (C°) at {depth} depth on {ocean_date}', layer_name = f"SST at {depth} depth", discrete=False)

# _______ DISPLAY ON STREAMLIT _______
Map.to_streamlit(height=600,  responsive=True, scrolling=False)