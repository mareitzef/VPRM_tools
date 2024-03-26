import pandas as pd
import numpy as np


def create_dataframe(index_input):
    if isinstance(index_input, (list, np.ndarray)):
        # If input is a list or array, create a DataFrame with the specified index
        return pd.DataFrame(index=index_input)
    else:
        raise ValueError("Unsupported input type. Please provide an list, or array.")


def VPRM_for_timeseries(
    Tmin,
    Tmax,
    Topt,
    PAR0,
    alpha,
    beta,
    lambd,
    T2M,
    LSWI,
    LSWI_min,
    LSWI_max,
    EVI,
    PAR,
    VEGTYP,
    VEGFRA,
):

    # ! VPRM vegetation classes:
    # !1-Evergreen c
    # !2-Deciduous
    # !3-Mixed forest
    # !4-Shrubland
    # !5-Savanna
    # !6-Cropland
    # !7-Grassland
    # !8-Others

    EVI_min = min(EVI)
    EVI_max = min(EVI)

    Tscale = []
    for i in range(len(T2M)):
        a1 = T2M[i] - Tmin
        a2 = T2M[i] - Tmax
        a3 = T2M[i] - Topt
        if a1 < 0 or a2 > 0:
            Tscale.append(0)
        else:
            Tscale.append(max(0, a1 * a2 / (a1 * a2 - a3**2)))

    # Tscale = [
    #     (((t - Tmin) * (t - Tmax)) / ((t - Tmin) * (t - Tmax) - (t - Topt) ** 2))
    #     for t in T2M
    # ]

    Wscale = []
    for i in range(len(LSWI)):
        if (
            VEGTYP == 4 or VEGTYP == 7
        ):  # Vegetation types 4 (shrubland) and 7 (grassland)
            if LSWI[i] < 1e-7:
                Wscale.append(0.0)
            else:
                Wscale.append((LSWI[i] - LSWI_min) / (LSWI_max - LSWI_min))
        else:
            Wscale.append((1.0 + LSWI[i]) / (1.0 + LSWI_max))

    Pscale = []
    for i in range(len(LSWI)):
        if VEGTYP == 1:  # Vegetation type 1 (Evergreen)
            Pscale.append(1.0)
        elif (
            VEGTYP == 5 or VEGTYP == 7
        ):  # Vegetation types 5 (Savanna) and 7 (grassland)
            Pscale.append((1.0 + LSWI[i]) / 2.0)
        else:
            evithresh = EVI_min + 0.55 * (EVI_max - EVI_min)
            if EVI[i] >= evithresh:
                Pscale.append(1.0)
            else:
                Pscale.append((1.0 + LSWI[i]) / 2.0)

    GPP = []
    for i in range(len(Tscale)):
        GPP_value = (
            (lambd * Tscale[i] * Pscale[i] * Wscale[i] * EVI[i] * VEGFRA)
            / (1 + (PAR[i] / PAR0))
            * PAR[i]
        )
        GPP.append(max(0, GPP_value))  # set negative values to 0

    Reco = [max(0, alpha * T2 + beta) for T2 in T2M]

    return GPP, Reco


def VPRM_new_for_timeseries(
    Tmin,
    Tmax,
    Topt,
    PAR0,
    beta,
    lambd,
    T_crit,
    T_mult,
    alpha1,
    alpha2,
    gamma,
    theta1,
    theta2,
    theta3,
    T2M,
    LSWI,
    LSWI_min,
    LSWI_max,
    EVI,
    PAR,
    VEGTYP,
    VEGFRA,
):

    EVI_min = min(EVI)
    EVI_max = max(EVI)

    Tscale = []
    for i in range(len(T2M)):
        a1 = T2M[i] - Tmin
        a2 = T2M[i] - Tmax
        a3 = T2M[i] - Topt
        if a1 < 0 or a2 > 0:
            Tscale.append(0)
        else:
            Tscale.append(max(0, a1 * a2 / (a1 * a2 - a3**2)))

    Wscale = []
    for i in range(len(LSWI)):
        if (
            VEGTYP == 4 or VEGTYP == 7
        ):  # Vegetation types 4 (shrubland) and 7 (grassland)
            if LSWI[i] < 1e-7:
                Wscale.append(0.0)
            else:
                Wscale.append((LSWI[i] - LSWI_min) / (LSWI_max - LSWI_min))
        else:
            Wscale.append((1.0 + LSWI[i]) / (1.0 + LSWI_max))

    Pscale = []
    for i in range(len(LSWI)):
        if VEGTYP == 1:  # Vegetation type 1 (Evergreen)
            Pscale.append(1.0)
        elif (
            VEGTYP == 5 or VEGTYP == 7
        ):  # Vegetation types 5 (Savanna) and 7 (grassland)
            Pscale.append((1.0 + LSWI[i]) / 2.0)
        else:
            evithresh = EVI_min + 0.55 * (EVI_max - EVI_min)
            if EVI[i] >= evithresh:
                Pscale.append(1.0)
            else:
                Pscale.append((1.0 + LSWI[i]) / 2.0)

    GPP = []
    for i in range(len(Tscale)):
        GPP_value = (
            (lambd * Tscale[i] * Pscale[i] * Wscale[i] * EVI[i] * VEGFRA)
            / (1 + (PAR[i] / PAR0))
            * PAR[i]
        )
        GPP.append(max(0, GPP_value))  # set negative values to 0

    Reco = []
    for i in range(len(Tscale)):
        if T2M[i] < T_crit:
            T_ds = T_crit - T_mult * (T_crit - T2M[i])
        else:
            T_ds = T2M[i]

        Wscale2 = (LSWI[i] - LSWI_min) / (LSWI_max - LSWI_min)
        Reco_t = (
            beta
            + alpha1 * T_ds
            + alpha2 * T_ds**2
            + gamma * EVI[i]
            + theta1 * Wscale2
            + theta2 * Wscale2 * T_ds
            + theta3 * Wscale2 * T_ds
        )

        Reco.append(max(0, Reco_t))

    return GPP, Reco
