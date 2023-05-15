import ee
import geemap
import streamlit as st


# Authenticate Earth Engine with Streamlit
def authenticate_earth_engine():
    ee.Authenticate()
    ee.Initialize()

# Create a map widget using geemap
def create_map():
    Map = geemap.Map()
    Map.setCenter(-99.957, 46.8947, 10)
    Map.addLayerControl()
    return Map

# Add a layer to the map
def add_layer_to_map(Map, image, vis_params, name):
    Map.addLayer(image, vis_params, name)

# Define the region of interest (ROI) on the map
def define_roi(Map):
    region = Map.user_roi
    Map.centerObject(region)

# Main function to run the Streamlit app
def main():
    # Authenticate Earth Engine
    authenticate_earth_engine()

    # Create a Streamlit app and add a title
    st.title("Soil Moisture Map")

    # Create a map widget
    Map = create_map()

    # Define the region of interest (ROI)
    define_roi(Map)

    # Extract the soil moisture band from the image collection
    dataset = ee.ImageCollection("NASA_USDA/HSL/SMAP10KM_soil_moisture")
    image = dataset.filterDate("2015-08-01", "2022-09-01").first()
    soilMoisture = image.select("ssm")

    # Apply a color palette to the band and visualize it on the map
    soilMoistureVis = {"min": 0.0, "max": 28.0, "palette": ["0300ff", "418504", "efff07", "efff07", "ff0303"]}
    add_layer_to_map(Map, soilMoisture, soilMoistureVis, "Soil Moisture")

    # Display the map
    Map

if __name__ == "__main__":
    main()
