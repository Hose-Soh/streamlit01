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

def get_local_soil_profile_at_poi(dataset, poi, buffer, olm_bands):
    # Get properties at the location of interest and transfer to client-side.
    prop = dataset.sample(poi, buffer).select(olm_bands).getInfo()

    # Selection of the features/properties of interest.
    profile = prop["features"][0]["properties"]

    # Re-shaping of the dict.
    profile = {key: round(val, 3) for key, val in profile.items()}

    return profile



def get_local_soil_profile_at_roi(dataset, roi, buffer, olm_bands, max_points = 5000):
    # Get properties at the location of interest and transfer to client-side.
    prop = dataset.sample(roi, buffer, numPixels=max_points).select(olm_bands).getInfo()

    # Selection of the features/properties of interest.
    profile = {key: [] for key in olm_bands}
    for idx in range(len(prop["features"])):
        for key, val in prop["features"][idx]["properties"].items():
            profile[key].append(round(val, 3))

    return profile


def get_band_mean_for_roi(roi_profile):
    logger.info(roi_profile.keys())
    mean_vals = {key: np.mean(values) for key, values in roi_profile.items()}
    return mean_vals
    

