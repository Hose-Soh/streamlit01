import ee
import ee_utils
'''
Functions related to the calculation of Soild Water Recharge (SWR)
'''

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

def calculate_available_water(fcm, wpm, zr):
    # Calculate the theoretical available water.
    taw = (
        (fcm.select("fc_mean").subtract(wpm.select("wp_mean"))).multiply(1000).multiply(zr)
    )
    return taw

def calculate_stored_water_at_fc(taw, p):
    stfc = taw.multiply(p)
    return stfc



def get_soil_hydric_bands(stfc, time0):
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
    return initial_image, ee.List([initial_image])



def compute_recharge(meteo_data, image_list, stfc, fcm, wpm):
    
    def calculate_recharge(image, image_list):
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
    
    return meteo_data.iterate(calculate_recharge, image_list)



def get_recharge_at_poi_df(meteo, poi, scale, stfc, fcm, wpm, time0):
    initial_image, image_list = get_soil_hydric_bands(stfc, time0)
    # Iterate the user-supplied function to the meteo collection.
    rech_list = compute_recharge(meteo, image_list, stfc, fcm, wpm)

    # Remove the initial image from our list.
    rech_list = ee.List(rech_list).remove(initial_image)

    # Transform the list into an ee.ImageCollection.
    rech_coll = ee.ImageCollection(rech_list)

    arr = rech_coll.getRegion(poi, scale).getInfo()
    rdf = ee_utils.ee_array_to_df(arr, ["pr", "pet", "apwl", "st", "rech"]).sort_index()

    return rdf
