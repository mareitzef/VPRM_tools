import math

# def calculate_tscale(t, t_min, t_max, t_opt):
#     if t_min <= t <= t_max:
#         value = ((t - t_min) * (t - t_max)) / (
#             (t - t_min) * (t - t_max) - (t - t_opt) ** 2
#         )
#     else:
#         value = 0
#     return value if ((t - t_min) * (t - t_max) - (t - t_opt) ** 2 != 0) else 0


def calculate_tscale(t, t_min, t_max, t_opt):
    if not math.isfinite(t):
        return 0

    if t_min <= t <= t_max:
        value = ((t - t_min) * (t - t_max)) / (
            (t - t_min) * (t - t_max) - (t - t_opt) ** 2
        )
    else:
        value = 0

    return value if ((t - t_min) * (t - t_max) - (t - t_opt) ** 2 != 0) else 0


def calculate_wscale(lswi, vegtyp, lswi_min, lswi_max):
    wscale = []
    for l in lswi:
        if vegtyp in [4, 7]:  # Vegetation types 4 (shrubland) and 7 (grassland)
            if l < 1e-7:
                wscale.append(0.0)
            else:
                wscale.append((l - lswi_min) / (lswi_max - lswi_min))
        else:
            wscale.append((1.0 + l) / (1.0 + lswi_max))
    return wscale


def calculate_pscale(evi, lswi, vegtyp, evi_min, evi_max):
    pscale = []
    evithresh = evi_min + 0.55 * (evi_max - evi_min)
    for e, l in zip(evi, lswi):
        if vegtyp == 1:  # Evergreen
            pscale.append(1.0)
        elif vegtyp in [5, 7]:  # Savanna and Grassland
            pscale.append((1.0 + l) / 2.0)
        else:
            if e >= evithresh:
                pscale.append(1.0)
            else:
                pscale.append((1.0 + l) / 2.0)
    return pscale


def calculate_GPP(lambd, Tscale, Pscale, Wscale, EVI, VEGFRA, PAR, PAR0):
    GPP = []
    for i in range(len(Tscale)):
        GPP_value = (
            (lambd * Tscale[i] * Pscale[i] * Wscale[i] * EVI[i] * VEGFRA)
            / (1 + (PAR[i] / PAR0))
        ) * PAR[i]
        GPP.append(max(0, GPP_value))  # set negative values to 0
    return GPP


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

    Tscale = [calculate_tscale(t, Tmin, Tmax, Topt) for t in T2M]

    Wscale = calculate_wscale(LSWI, VEGTYP, LSWI_min, LSWI_max)

    Pscale = calculate_pscale(EVI, LSWI, VEGTYP, min(EVI), max(EVI))

    GPP = calculate_GPP(lambd, Tscale, Pscale, Wscale, EVI, VEGFRA, PAR, PAR0)

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

    Tscale = [calculate_tscale(t, Tmin, Tmax, Topt) for t in T2M]

    Wscale = calculate_wscale(LSWI, VEGTYP, LSWI_min, LSWI_max)

    Pscale = calculate_pscale(EVI, LSWI, VEGTYP, min(EVI), max(EVI))

    GPP = calculate_GPP(lambd, Tscale, Pscale, Wscale, EVI, VEGFRA, PAR, PAR0)

    Reco = (
        []
    )  # only Reco is different in VPRM_new, but as VPRM is optimized against NEE it also changed GPP parameters
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
