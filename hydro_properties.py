import ee
import logging

logger = logging.getLogger(__name__)


def compute_hyrdo_properties(sand, clay, orgm, olm_bands):
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

    return field_capacity, wilting_point