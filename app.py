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

st.set_page_config(page_title='Soil Data Exploration')

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
f_date = st.date_input("Final date of interest (exclusive)", max_value = datetime.now())

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


# Get soil property images
sand = get_soil_prop("sand")
clay = get_soil_prop("clay")
orgc = get_soil_prop("orgc")
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


# Create a GEE map centered on the location of interest
my_map = geemap.Map(center=[lat, lon], zoom=3)

# Set visualization parameters.
vis_params = {
    
    "min": 0.01,
    "max": 1,
    "opacity": 1,
    
}

# Add the sand content data to the map object.
my_map.addLayer(sand, vis_params, "Sand Content")

# Add a marker at the location of interest.
# Add a marker at the location of interest.
folium.Marker([lat, lon], popup="point of interest").add_to(my_map)
# Add a layer control panel to the map.
my_map.add_child(folium.LayerControl())


# Add a layer control panel to the map.
my_map.addLayerControl()

# Display the map.
my_map.to_streamlit(height=600,  responsive=True, scrolling=False)


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

#___________________________________________________Calculate Hydraulic Properties_____________________________________________________________


#Conversion of organic carbon content into organic matter content.
orgm = orgc.multiply(1.724)

#Organic matter content profile.
profile_orgm = local_profile(orgm, poi, scale)

#Print the organic matter content profile.
st.write("Organic Matter content profile at the location of interest: ", profile_orgm)
##st.write(.getInfo())

#Initialization of two constant images for wilting point and field capacity.
wilting_point = ee.Image(0)
field_capacity = ee.Image(0)

#Calculation for each standard depth using a loop.
for key in olm_bands:

    # Getting sand, clay and organic matter at the appropriate depth.
    si = sand.select(key)
    ci = clay.select(key)
    oi = orgm.select(key)

    # Calculation of the wilting point.
    # The theta_1500t parameter is needed for the given depth.
    theta_1500ti = (
        ee.Image(0)
        .expression(
            "-0.024 * S + 0.487 * C + 0.006 * OM + 0.005 * (S * OM)\
        - 0.013 * (C * OM) + 0.068 * (S * C) + 0.031",
            {
                "S": si,
                "C": ci,
                "OM": oi,
            },
        )
        .rename("T1500ti")
    )

    # Final expression for the wilting point.
    wpi = theta_1500ti.expression(
        "T1500ti + ( 0.14 * T1500ti - 0.002)", {"T1500ti": theta_1500ti}
    ).rename("wpi")

    # Add as a new band of the global wilting point ee.Image.
    # Do not forget to cast the type with float().
    wilting_point = wilting_point.addBands(wpi.rename(key).float())

    # Same process for the calculation of the field capacity.
    # The parameter theta_33t is needed for the given depth.
    theta_33ti = (
        ee.Image(0)
        .expression(
            "-0.251 * S + 0.195 * C + 0.011 * OM +\
        0.006 * (S * OM) - 0.027 * (C * OM)+\
        0.452 * (S * C) + 0.299",
            {
                "S": si,
                "C": ci,
                "OM": oi,
            },
        )
        .rename("T33ti")
    )

    # Final expression for the field capacity of the soil.
    # Final expression for the field capacity of the soil.
    fci = theta_33ti.expression(
        "T33ti + (1.283 * T33ti * T33ti - 0.374 * T33ti - 0.015)",
        {"T33ti": theta_33ti.select("T33ti")},
    )

    # Add a new band of the global field capacity ee.Image.
    field_capacity = field_capacity.addBands(fci.rename(key).float())

profile_wp = local_profile(wilting_point, poi, scale)
profile_fc = local_profile(field_capacity, poi, scale)

st.write("Wilting point profile: ", profile_wp)
st.write("Field capacity profile:", profile_fc)

#_________________________________________Hydraulic Properties Visualization__________________________________________________


fig, ax = plt.subplots(figsize=(15, 6))
ax.axes.get_yaxis().set_visible(False)

# Definition of the label locations.
x = np.arange(len(olm_bands))

# Width of the bar of the barplot.
width = 0.25

# Barplot associated with the water content at the wilting point.
rect1 = ax.bar(
    x - width / 2,
    [round(profile_wp[b] * 100, 2) for b in olm_bands],
    width,
    label="Water content at wilting point",
    color="red",
    alpha=0.5,
)

# Barplot associated with the water content at the field capacity.
rect2 = ax.bar(
    x + width / 2,
    [round(profile_fc[b] * 100, 2) for b in olm_bands],
    width,
    label="Water content at field capacity",
    color="blue",
    alpha=0.5,
)

# Add Labels on top of bars.
autolabel_soil_prop(rect1)
autolabel_soil_prop(rect2)

# Title of the plot.
ax.set_title("Hydraulic properties of the soil at different depths", fontsize=14)

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

# Put a legend below current axis.
ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2
)

st.pyplot(fig)


#_____________________________________________Getting Meteorological Datasets_____________________________________________

# Import precipitation.
pr = (
    ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
    .select("precipitation")
    .filterDate(i_date, f_date)
)

# Import potential evaporation PET and its quality indicator ET_QC.
pet = (
    ee.ImageCollection("MODIS/006/MOD16A2")
    .select(["PET", "ET_QC"])
    .filterDate(i_date, f_date)
)

# Evaluate local precipitation conditions.
local_pr = pr.getRegion(poi, scale)
pprint.pprint(local_pr)



def ee_array_to_df(arr, list_of_bands):
    """Transforms client-side ee.Image.getRegion array to pandas.DataFrame."""
    df = pd.DataFrame(arr)

    # Rearrange the header.
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    # Convert the data to numeric values.
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors="coerce")

    # Convert the time field into a datetime.
    df["datetime"] = pd.to_datetime(df["time"], unit="ms")

    # Keep the columns of interest.
    df = df[["time", "datetime", *list_of_bands]]

    # The datetime column is defined as index.
    df = df.set_index("datetime")

    return df

pr_df = ee_array_to_df(local_pr, ["precipitation"])
#pr_df.head(10)




# Evaluate local potential evapotranspiration.
local_pet = pet.getRegion(poi, scale)

# Transform the result into a pandas dataframe.
pet_df = ee_array_to_df(local_pet, ["PET", "ET_QC"])
pet_df.head(5)

def sum_resampler(coll, freq, unit, scale_factor, band_name):
    """
    This function aims to resample the time scale of an ee.ImageCollection.
    The function returns an ee.ImageCollection with the averaged sum of the
    band on the selected frequency.

    coll: (ee.ImageCollection) only one band can be handled
    freq: (int) corresponds to the resampling frequence
    unit: (str) corresponds to the resampling time unit.
                must be 'day', 'month' or 'year'
    scale_factor (float): scaling factor used to get our value in the good unit
    band_name (str) name of the output band
    """

    # Define initial and final dates of the collection.
    firstdate = ee.Date(
        coll.sort("system:time_start", True).first().get("system:time_start")
    )

    lastdate = ee.Date(
        coll.sort("system:time_start", False).first().get("system:time_start")
    )

    # Calculate the time difference between both dates.
    # https://developers.google.com/earth-engine/apidocs/ee-date-difference
    diff_dates = lastdate.difference(firstdate, unit)

    # Define a new time index (for output).
    new_index = ee.List.sequence(0, ee.Number(diff_dates), freq)

    # Define the function that will be applied to our new time index.
    def apply_resampling(date_index):
        # Define the starting date to take into account.
        startdate = firstdate.advance(ee.Number(date_index), unit)

        # Define the ending date to take into account according
        # to the desired frequency.
        enddate = firstdate.advance(ee.Number(date_index).add(freq), unit)

        # Calculate the number of days between starting and ending days.
        diff_days = enddate.difference(startdate, "day")

        # Calculate the composite image.
        image = (
            coll.filterDate(startdate, enddate)
            .mean()
            .multiply(diff_days)
            .multiply(scale_factor)
            .rename(band_name)
        )

        # Return the final image with the appropriate time index.
        return image.set("system:time_start", startdate.millis())

    # Map the function to the new time index.
    res = new_index.map(apply_resampling)

    # Transform the result into an ee.ImageCollection.
    res = ee.ImageCollection(res)

    return res

# Apply the resampling function to the precipitation dataset.
pr_m = sum_resampler(pr, 1, "month", 1, "pr")

# Evaluate the result at the location of interest.
pprint.pprint(pr_m.getRegion(poi, scale))

# Apply the resampling function to the PET dataset.
pet_m = sum_resampler(pet.select("PET"), 1, "month", 0.0125, "pet")

# Evaluate the result at the location of interest.
pprint.pprint(pet_m.getRegion(poi, scale))

# Combine precipitation and evapotranspiration.
meteo = pr_m.combine(pet_m)

# Import meteorological data as an array at the location of interest.
meteo_arr = meteo.getRegion(poi, scale)

# Print the result.
pprint.pprint(meteo_arr)

# Transform the array into a pandas dataframe and sort the index.
meteo_df = ee_array_to_df(meteo_arr, ["pr", "pet"]).sort_index()

# Data visualization
fig, ax = plt.subplots(figsize=(15, 6))

# Barplot associated with precipitations.
meteo_df["pr"].plot(kind="bar", ax=ax, label="precipitation")

# Barplot associated with potential evapotranspiration.
meteo_df["pet"].plot(
    kind="bar", ax=ax, label="potential evapotranspiration", color="orange", alpha=0.5
)

# Add a legend.
ax.legend()

# Add some x/y-labels properties.
ax.set_ylabel("Intensity [mm]")
ax.set_xlabel(None)

# Define the date format and shape of x-labels.
x_labels = meteo_df.index.strftime("%m-%Y")
ax.set_xticklabels(x_labels, rotation=90, fontsize=10)

st.pyplot(fig)

#############################################################
#############################
##########################
########################
##################
##############


#________________________________________________Implementation of the TM procedure_________________________________________________

zr = ee.Image(0.5)
p = ee.Image(0.5)

def olm_prop_mean(olm_image, band_output_name):
    """
    This function calculates an averaged value of
    soil properties between reference depths.
    """
    mean_image = olm_image.expression(
        "(b0 + b10 + b30 + b60 + b100 + b200) / 6",
        {
            "b0": olm_image.select("b0"),
            "b10": olm_image.select("b10"),
            "b30": olm_image.select("b30"),
            "b60": olm_image.select("b60"),
            "b100": olm_image.select("b100"),
            "b200": olm_image.select("b200"),
        },
    ).rename(band_output_name)

    return mean_image


# Apply the function to field capacity and wilting point.
fcm = olm_prop_mean(field_capacity, "fc_mean")
wpm = olm_prop_mean(wilting_point, "wp_mean")

# Calculate the theoretical available water.
taw = (
    (fcm.select("fc_mean").subtract(wpm.select("wp_mean"))).multiply(1000).multiply(zr)
)

# Calculate the stored water at the field capacity.
stfc = taw.multiply(p)

# Define the initial time (time0) according to the start of the collection.
time0 = meteo.first().get("system:time_start")

# Initialize all bands describing the hydric state of the soil.
# Do not forget to cast the type of the data with a .float().
# Initial recharge.
initial_rech = ee.Image(0).set("system:time_start", time0).select([0], ["rech"]).float()

# Initialization of APWL.
initial_apwl = ee.Image(0).set("system:time_start", time0).select([0], ["apwl"]).float()

# Initialization of ST.
initial_st = stfc.set("system:time_start", time0).select([0], ["st"]).float()

# Initialization of precipitation.
initial_pr = ee.Image(0).set("system:time_start", time0).select([0], ["pr"]).float()

# Initialization of potential evapotranspiration.
initial_pet = ee.Image(0).set("system:time_start", time0).select([0], ["pet"]).float()

initial_image = initial_rech.addBands(
    ee.Image([initial_apwl, initial_st, initial_pr, initial_pet])
)

image_list = ee.List([initial_image])

#_____________________________________________Iteration over an ee.ImageCollection_________________________________________________


def recharge_calculator(image, image_list):
    """
    Contains operations made at each iteration.
    """
    # Determine the date of the current ee.Image of the collection.
    localdate = image.date().millis()

    # Import previous image stored in the list.
    prev_im = ee.Image(ee.List(image_list).get(-1))

    # Import previous APWL and ST.
    prev_apwl = prev_im.select("apwl")
    prev_st = prev_im.select("st")

    # Import current precipitation and evapotranspiration.
    pr_im = image.select("pr")
    pet_im = image.select("pet")

    # Initialize the new bands associated with recharge, apwl and st.
    # DO NOT FORGET TO CAST THE TYPE WITH .float().
    new_rech = (
        ee.Image(0)
        .set("system:time_start", localdate)
        .select([0], ["rech"])
        .float()
    )

    new_apwl = (
        ee.Image(0)
        .set("system:time_start", localdate)
        .select([0], ["apwl"])
        .float()
    )

    new_st = (
        prev_st.set("system:time_start", localdate).select([0], ["st"]).float()
    )

    # Calculate bands depending on the situation using binary layers with
    # logical operations.

    # CASE 1.
    # Define zone1: the area where PET > P.
    zone1 = pet_im.gt(pr_im)

    # Calculation of APWL in zone 1.
    zone1_apwl = prev_apwl.add(pet_im.subtract(pr_im)).rename("apwl")
    # Implementation of zone 1 values for APWL.
    new_apwl = new_apwl.where(zone1, zone1_apwl)

    # Calculate ST in zone 1.
    zone1_st = prev_st.multiply(
        ee.Image.exp(zone1_apwl.divide(stfc).multiply(-1))
    ).rename("st")
    # Implement ST in zone 1.
    new_st = new_st.where(zone1, zone1_st)

    # CASE 2.
    # Define zone2: the area where PET <= P.
    zone2 = pet_im.lte(pr_im)

    # Calculate ST in zone 2.
    zone2_st = prev_st.add(pr_im).subtract(pet_im).rename("st")
    # Implement ST in zone 2.
    new_st = new_st.where(zone2, zone2_st)

    # CASE 2.1.
    # Define zone21: the area where PET <= P and ST >= STfc.
    zone21 = zone2.And(zone2_st.gte(stfc))

    # Calculate recharge in zone 21.
    zone21_re = zone2_st.subtract(stfc).rename("rech")
    # Implement recharge in zone 21.
    new_rech = new_rech.where(zone21, zone21_re)
    # Implement ST in zone 21.
    new_st = new_st.where(zone21, stfc)

    # CASE 2.2.
    # Define zone 22: the area where PET <= P and ST < STfc.
    zone22 = zone2.And(zone2_st.lt(stfc))

    # Calculate APWL in zone 22.
    zone22_apwl = (
        stfc.multiply(-1).multiply(ee.Image.log(zone2_st.divide(stfc))).rename("apwl")
    )
    # Implement APWL in zone 22.
    new_apwl = new_apwl.where(zone22, zone22_apwl)

    # Create a mask around area where recharge can effectively be calculated.
    # Where we have have PET, P, FCm, WPm (except urban areas, etc.).
    mask = pet_im.gte(0).And(pr_im.gte(0)).And(fcm.gte(0)).And(wpm.gte(0))

    # Apply the mask.
    new_rech = new_rech.updateMask(mask)

    # Add all Bands to our ee.Image.
    new_image = new_rech.addBands(ee.Image([new_apwl, new_st, pr_im, pet_im]))

    # Add the new ee.Image to the ee.List.
    return ee.List(image_list).add(new_image)

# Iterate the user-supplied function to the meteo collection.
rech_list = meteo.iterate(recharge_calculator, image_list)

# Remove the initial image from our list.
rech_list = ee.List(rech_list).remove(initial_image)

# Transform the list into an ee.ImageCollection.
rech_coll = ee.ImageCollection(rech_list)

arr = rech_coll.getRegion(poi, scale).getInfo()
rdf = ee_array_to_df(arr, ["pr", "pet", "apwl", "st", "rech"]).sort_index()
rdf.head(12)

# Data visualization in the form of barplots.
fig, ax = plt.subplots(figsize=(15, 6))

# Barplot associated with precipitation.
rdf["pr"].plot(kind="bar", ax=ax, label="precipitation", alpha=0.5)

# Barplot associated with potential evapotranspiration.
rdf["pet"].plot(
    kind="bar", ax=ax, label="potential evapotranspiration", color="orange", alpha=0.2
)

# Barplot associated with groundwater recharge
rdf["rech"].plot(kind="bar", ax=ax, label="recharge", color="green", alpha=1)

# Add a legend.
ax.legend()

# Define x/y-labels properties.
ax.set_ylabel("Intensity [mm]")
ax.set_xlabel(None)

# Define the date format and shape of x-labels.
x_labels = rdf.index.strftime("%m-%Y")
ax.set_xticklabels(x_labels, rotation=90, fontsize=10)

st.pyplot(fig)

# Resample the pandas dataframe on a yearly basis making the sum by year.
rdfy = rdf.resample("Y").sum()

# Calculate the mean value.
mean_recharge = rdfy["rech"].mean()

# Print the result.
st.write(
    "The mean annual recharge at our point of interest is", int(mean_recharge), "mm/an"
)



