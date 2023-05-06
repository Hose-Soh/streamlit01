import ee
import ee_utils


def get_precipitation_data_for_dates(start_date, end_date):
    return (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .select("precipitation")
        .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        )


def get_potential_evaporation_for_dates(start_date, end_date):
    # Import potential evaporation PET and its quality indicator ET_QC.
    return (
        ee.ImageCollection("MODIS/006/MOD16A2")
        .select(["PET", "ET_QC"])
        .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        )

def get_meteorological_for_poi(poi, scale, start_date, end_date):
    pr = get_precipitation_data_for_dates(start_date, end_date)
    pet = get_potential_evaporation_for_dates(start_date, end_date)

    local_pr = pr.getRegion(poi, scale).getInfo()

    pr_df = ee_utils.ee_array_to_df(local_pr, ["precipitation"])


    # Evaluate local potential evapotranspiration.
    local_pet = pet.getRegion(poi, scale).getInfo()

    # Transform the result into a pandas dataframe.
    pet_df = ee_utils.ee_array_to_df(local_pet, ["PET", "ET_QC"])

    # Apply the resampling function to the precipitation dataset.
    pr_m = ee_utils.sum_resampler(pr, 1, "month", 1, "pr")

    # Evaluate the result at the location of interest.
    #pprint.pprint(pr_m.getRegion(poi, scale).getInfo()[:5])

    # Apply the resampling function to the PET dataset.
    pet_m = ee_utils.sum_resampler(pet.select("PET"), 1, "month", 0.0125, "pet")

    # Evaluate the result at the location of interest.
    #pprint.pprint(pet_m.getRegion(poi, scale).getInfo()[:5])

    # Combine precipitation and evapotranspiration.
    meteo = pr_m.combine(pet_m)

    return meteo


def get_meteorological_df_for_poi(meteo, poi, scale):
    # Import meteorological data as an array at the location of interest.
    meteo_arr = meteo.getRegion(poi, scale).getInfo()

    # Print the result.
    #pprint.pprint(meteo_arr[:5])

    # Transform the array into a pandas dataframe and sort the index.
    meteo_df = ee_utils.ee_array_to_df(meteo_arr, ["pr", "pet"]).sort_index()

    return meteo_df