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


#_______________________________________________________Determination of Soil Texture and Properties____________________________________________


# Soil depths [in cm] where we have data.
olm_depths = [0, 10, 30, 60, 100, 200]

# Names of bands associated with reference depths.
olm_bands = ["b" + str(sd) for sd in olm_depths]

def get_soil_prop(param):
    """
    This function returns soil properties image
    param (str): must be one of:
        "sand"     - Sand fraction
        "clay"     - Clay fraction
        "orgc"     - Organic Carbon fraction
    """
    if param == "sand":  # Sand fraction [%w]
        snippet = "OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02"
        # Define the scale factor in accordance with the dataset description.
        scale_factor = 1 * 0.01

    elif param == "clay":  # Clay fraction [%w]
        snippet = "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02"
        # Define the scale factor in accordance with the dataset description.
        scale_factor = 1 * 0.01

    elif param == "orgc":  # Organic Carbon fraction [g/kg]
        snippet = "OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02"
        # Define the scale factor in accordance with the dataset description.
        scale_factor = 5 * 0.001  # to get kg/kg
    else:
        st.write("error")
        return

    # Apply the scale factor to the ee.Image.
    dataset = ee.Image(snippet).multiply(scale_factor)

    return dataset

#________________________________________Visualization for Soil Content___________________________________________


# Define the Earth Engine image collection
dataset = ee.ImageCollection("OpenLandMap/SOL/SOL_PHIHOX_MEDR")

# Define the location of interest
lat, lon = 46.2044, 6.1432
poi = ee.Geometry.Point(lon, lat)
scale = 30

# Define the soil properties of interest
olm_bands = ["BDRICM", "CLYPPT", "ORCDRC", "PHIHOX"]
olm_depths = [5, 15, 30, 60, 100]

# Get soil property images
sand = dataset.select("BDRICM").first()
clay = dataset.select("CLYPPT").first()
orgc = dataset.select("ORCDRC").first()
#ph = dataset.select("PHIHOX").first()


def add_ee_layer(self, ee_image_object, vis_params, name):
    """Adds a method for displaying Earth Engine image tiles to folium map."""
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict["tile_fetcher"].url_format,
        attr="Map Data &copy; <a href='https://earthengine.google.com/'>Google Earth Engine</a>",
        name=name,
        overlay=True,
        control=True,
    ).add_to(self)


# Add Earth Engine drawing method to folium.
folium.Map.add_ee_layer = add_ee_layer

# Create a folium map centered on the location of interest
my_map = folium.Map(location=[lat, lon], zoom_start=3)

# Set visualization parameters.
vis_params = {
    "bands": ["BDRICM"],
    "min": 0.01,
    "max": 1,
    "opacity": 1,
    "palette": ["white", "#464646"],
}

# Add the sand content data to the map object.
my_map.add_ee_layer(sand, vis_params, "Sand Content")

# Add a marker at the location of interest.
folium.Marker([lat, lon], popup="point of interest").add_to(my_map)

# Add a layer control panel to the map.
my_map.add_child(folium.LayerControl())

# Display the map.
st.write(my_map._repr_html_(), unsafe_allow_html=True)


def local_profile(dataset, poi, buffer):
    # Get properties at the location of interest and transfer to client-side.
    prop = dataset.sample(poi, buffer).select(olm_bands).getInfo()

    # Selection of the features/properties of interest.
    profile = prop["features"][0]["properties"]

    # Re-shaping of the dict.
    profile = {key: round(val, 3) for key, val in profile.items()}

    return profile


# Apply the function to get the sand profile.
profile_sand = local_profile(sand, poi, scale)

# Print the result.
st.write("Sand content profile at the location of interest:\n", profile_sand)

# Clay and organic content profiles.
profile_clay = local_profile(clay, poi, scale)
profile_orgc = local_profile(orgc, poi, scale)

# Data visualization in the form of a bar plot.
# Create the plot
fig, ax = plt.subplots(figsize=(15, 6))
ax.axes.get_yaxis().set_visible(False)

# Definition of label locations.
x = np.arange(len(olm_bands))

# Definition of the bar width.
width = 0.25

# Bar plot representing the sand content profile.
rect1 = ax.bar(
    x - width,
    [round(100 * profile_sand[b], 2) for b in olm_bands],
    width,
    label="Sand",
    color="#ecebbd",
)

# Bar plot representing the clay content profile.
rect2 = ax.bar(
    x,
    [round(100 * profile_clay[b], 2) for b in olm_bands],
    width,
    label="Clay",
    color="#6f6c5d",
)

# Bar plot representing the organic carbon content profile.
rect3 = ax.bar(
    x + width,
    [round(100 * profile_orgc[b], 2) for b in olm_bands],
    width,
    label="Organic Carbon",
    color="black",
    alpha=0.75,
)

# Definition of a function to attach a label to each bar.
def autolabel_soil_prop(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            "{}".format(height) + "%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset.
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

# Application of the function to each barplot.
autolabel_soil_prop(rect1)
autolabel_soil_prop(rect2)
autolabel_soil_prop(rect3)

# Title of the plot.
ax.set_title("Properties of the soil at different depths (mass content)", fontsize=14)

# Properties of x/y labels and ticks.
ax.set_xticks(x)
x_labels = [str(d) + " cm" for d in olm_depths]
ax.set_xticklabels(x_labels, rotation=45, fontsize=10)

ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# Shrink current axis's height by 10% on the bottom.
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

# Add a legend below current axis.
ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3
)

# Display the plot using Streamlit.
st.pyplot(fig)