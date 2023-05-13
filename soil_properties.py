import ee
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def convert_orgc_to_orgm(org_c):
    ''' 
        Converts organic carbon content into organic matter content.
    '''
    return org_c.multiply(1.724)

def get_soil_prop(soil_type):
    """
    This function returns soil properties image
    param (str): must be one of:
        "sand"     - Sand fraction
        "clay"     - Clay fraction
        "orgc"     - Organic Carbon fraction
    """
    if soil_type == "sand":  # Sand fraction [%w]
        snippet = "OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02"
        # Define the scale factor in accordance with the dataset description.
        scale_factor = 1 * 0.01

    elif soil_type == "clay":  # Clay fraction [%w]
        snippet = "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02"
        # Define the scale factor in accordance with the dataset description.
        scale_factor = 1 * 0.01

    elif soil_type == "orgc":  # Organic Carbon fraction [g/kg]
        snippet = "OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02"
        # Define the scale factor in accordance with the dataset description.
        scale_factor = 5 * 0.001  # to get kg/kg
    else:
        logger.error(f"The soil property '{soil_type} was not recognised")
        return None

    # Apply the scale factor to the ee.Image.
    dataset = ee.Image(snippet).multiply(scale_factor)

    return dataset

def get_local_soil_profile_at_poi(dataset, roi, buffer, olm_bands, file_name):
    # # Soil depths [in cm] where we have data.
    # olm_depths = [0, 10, 30, 60, 100, 200]

    # # Names of bands associated with reference depths.
    # olm_bands = ["b" + str(sd) for sd in olm_depths]
    # Get properties at the location of interest and transfer to client-side.
    prop = dataset.sample(roi, buffer).select(olm_bands).getInfo()
    #print(prop)
    
    # Initialize an empty list to store dictionaries for each ID
    data_dicts = []

    # Iterate over each feature and extract 'id' and 'properties'
    for feature in prop['features']:
        id_value = feature['id']
        properties = feature['properties']
    
        # Create a dictionary for each ID
        data_dict = {'id': id_value}
        data_dict.update(properties)
    
        # Append the dictionary to the list
        data_dicts.append(data_dict)

    # Create the DataFrame from the list of dictionaries
    df = pd.DataFrame(data_dicts)

    # Reorder the columns
    column_order = ['id', 'b0', 'b10', 'b30', 'b60', 'b100', 'b200']
    df = df[column_order]

    #print(df)
    df.to_csv(file_name, index=False)
    
    # Calculate the average of 'b0', 'b10', 'b30', 'b60', 'b100', 'b200'
    averages = df[['b0', 'b10', 'b30', 'b60', 'b100', 'b200']].mean()

    # Create a dictionary with 'b0', 'b10', 'b30', 'b60', 'b100', 'b200' as keys and their average as values
    average_dict = averages.to_dict()
    #print(average_dict)
    
    # Selection of the features/properties of interest.
    #profile = prop["features"][0]["properties"]
    #print("profile", profile)
    # Re-shaping of the dict.
    profile = {key: round(val, 3) for key, val in average_dict.items()}
    #print("profile", profile)
    return profile