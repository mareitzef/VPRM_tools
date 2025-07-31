import pandas as pd
import numpy as np
from pyrealm.pmodel import PModel, PModelEnvironment, SubdailyPModel, AcclimationModel


def pModel_subdaily(
    df_site_and_modis: pd.DataFrame,
    days_memory: float,
    window_center_i: int,
    half_width_i: int,
):
    gC_to_mumol = 0.0833  # 1 µg C m⁻² s⁻¹ → 0.0833 µmol CO₂ m⁻² s⁻¹

    # Rename Fpar column if needed
    df_site_and_modis.columns = df_site_and_modis.columns.str.replace(
        r"Fpar_500m_\d{4}-\d{2}-\d{2}", "Fpar_500m", regex=True
    )

    # Extract and sanitize inputs
    temp_subdaily = df_site_and_modis.loc[:, "TA_F"].copy()
    temp_subdaily[temp_subdaily < -25] = np.nan

    vpd_subdaily = df_site_and_modis.loc[:, "VPD_F"] * 100
    vpd_subdaily[vpd_subdaily < 0] = 0

    ppfd_subdaily = df_site_and_modis.loc[:, "PPFD_IN"].copy()
    ppfd_subdaily[ppfd_subdaily < 0] = 0

    co2_subdaily = df_site_and_modis.loc[:, "CO2_F_MDS"]
    patm_subdaily = df_site_and_modis.loc[:, "PA_F"] * 1000
    fpar_subdaily = df_site_and_modis.loc[:, "Fpar_500m"] * 100
    datetime_subdaily = pd.to_datetime(df_site_and_modis["TIMESTAMP_START"]).to_numpy()

    # # Setup acclimation model
    # acclim_model = AcclimationModel(
    #     datetime_subdaily, allow_holdover=True, alpha=1 / days_memory
    # )
    # acclim_model.set_window(
    #     window_center=np.timedelta64(window_center_i, "h"),
    #     half_width=np.timedelta64(half_width_i, "m"),
    # )

    # Create the acclimation model - merging acclimation functions into a common class
    acclim_model = AcclimationModel(
        datetime_subdaily, alpha=1 / 15, allow_holdover=True
    )
    acclim_model.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(1, "h"),
    )

    # Create the PModelEnvironment, including FAPAR and PPFD
    pm_env = PModelEnvironment(
        tc=temp_subdaily.values,
        vpd=vpd_subdaily.values,
        co2=co2_subdaily.values,
        patm=patm_subdaily.values,
        ppfd=ppfd_subdaily.values,
        fapar=fpar_subdaily.values,
    )

    # Fit the subdaily model - which now accepts all of the alternative method
    # arguments  used by the PModel class.
    subdaily_model = SubdailyPModel(
        env=pm_env,
        acclim_model=acclim_model,
        method_kphio="fixed",
        method_optchi="prentice14",
        reference_kphio=1 / 8,  # Again, this is the default.
    )
    pmodel_subdaily_gpp = subdaily_model.gpp * gC_to_mumol

    return pmodel_subdaily_gpp


# # From Pyrealm 1.0.0
# import pandas as pd
# import numpy as np
# from pyrealm.pmodel import PModel, PModelEnvironment, SubdailyScaler, SubdailyPModel
# from pyrealm.pmodel.functions import calc_ftemp_arrh, calc_ftemp_kphio
# from pyrealm.pmodel.subdaily import memory_effect
# from pyrealm.pmodel.optimal_chi import OptimalChiPrentice14


# def pModel_subdaily(
#     df_site_and_modis: pd.DataFrame,
#     days_memory: float,
#     window_center_i: int,
#     half_width_i: int,
# ):

#     # Extract the key half hourly timestep variables as numpy arrays
#     temp_subdaily = df_site_and_modis.loc[
#         :, "TA_F"
#     ]  # daily temperature, unit degree (°C)
#     df_site_and_modis.loc[df_site_and_modis["TA_F"] < -25, "TA_F"] = np.nan

#     ppfd_subdaily = df_site_and_modis.loc[:, "PPFD_IN"]
#     vpd_subdaily = (
#         df_site_and_modis.loc[:, "VPD_F"] * 100
#     )  # vpd, unit: hPa converted to Pa
#     co2_subdaily = df_site_and_modis.loc[:, "CO2_F_MDS"]  # CO2, unit ppm
#     patm_subdaily = (
#         df_site_and_modis.loc[:, "PA_F"] * 1000
#     )  # site pressure, unit kPa converted to Pa
#     # rename column starting with 'Fpar_500m_(some Date)' to 'Fpar_500m'
#     df_site_and_modis.columns = df_site_and_modis.columns.str.replace(
#         r"Fpar_500m_\d{4}-\d{2}-\d{2}", "Fpar_500m", regex=True
#     )
#     fpar_subdaily = (
#         df_site_and_modis.loc[:, "Fpar_500m"] * 100
#     )  # 100 to convert to percent
#     vpd_subdaily.values[vpd_subdaily.values < 0] = 0
#     ppfd_subdaily.values[ppfd_subdaily.values < 0] = 0


#     # Calculate the photosynthetic environment
#     subdaily_env = PModelEnvironment(
#         tc=temp_subdaily.values,
#         vpd=vpd_subdaily.values,
#         co2=co2_subdaily.values,
#         patm=patm_subdaily.values,
#     )

#     # Create the fast slow scaler
#     datetime_subdaily = pd.to_datetime(df_site_and_modis["TIMESTAMP_START"]).to_numpy()
#     fsscaler = SubdailyScaler(datetime_subdaily)

#     # Set the acclimation window as the values within a one hour window centred on noon
#     fsscaler.set_window(
#         window_center=np.timedelta64(window_center_i, "h"),
#         half_width=np.timedelta64(half_width_i, "m"),
#     )

#     # Fit the P Model with fast and slow responses
#     pmodel_subdaily = SubdailyPModel(
#         env=subdaily_env,
#         fs_scaler=fsscaler,
#         allow_holdover=True,
#         ppfd=ppfd_subdaily.values,
#         fapar=fpar_subdaily.values,
#     )

#     temp_acclim = fsscaler.get_daily_means(temp_subdaily.values)
#     co2_acclim = fsscaler.get_daily_means(co2_subdaily.values)
#     vpd_acclim = fsscaler.get_daily_means(vpd_subdaily.values)
#     patm_acclim = fsscaler.get_daily_means(patm_subdaily.values)
#     ppfd_acclim = fsscaler.get_daily_means(ppfd_subdaily.values)
#     fapar_acclim = fsscaler.get_daily_means(fpar_subdaily.values)

#     # Fit the P Model to the acclimation conditions
#     daily_acclim_env = PModelEnvironment(
#         tc=temp_acclim, vpd=vpd_acclim, co2=co2_acclim, patm=patm_acclim
#     )

#     pmodel_acclim = PModel(daily_acclim_env, kphio=1 / 8)
#     pmodel_acclim.estimate_productivity(fapar=fapar_acclim, ppfd=ppfd_acclim)
#     # pmodel_acclim.summarize()

#     ha_vcmax25 = 65330
#     ha_jmax25 = 43900
#     tk_acclim = temp_acclim + pmodel_subdaily.env.core_const.k_CtoK
#     vcmax25_acclim = pmodel_acclim.vcmax * (1 / calc_ftemp_arrh(tk_acclim, ha_vcmax25))
#     jmax25_acclim = pmodel_acclim.jmax * (1 / calc_ftemp_arrh(tk_acclim, ha_jmax25))
#     # Calculation of memory effect in xi, vcmax25 and jmax25
#     xi_real = memory_effect(pmodel_acclim.optchi.xi, alpha=1 / days_memory)
#     vcmax25_real = memory_effect(
#         vcmax25_acclim, alpha=1 / days_memory, allow_holdover=True
#     )
#     jmax25_real = memory_effect(
#         jmax25_acclim, alpha=1 / days_memory, allow_holdover=True
#     )
#     tk_subdaily = subdaily_env.tc + pmodel_subdaily.env.core_const.k_CtoK

#     # Fill the realised jmax and vcmax from subdaily to daily
#     vcmax25_subdaily = fsscaler.fill_daily_to_subdaily(vcmax25_real)
#     jmax25_subdaily = fsscaler.fill_daily_to_subdaily(jmax25_real)

#     # Adjust to actual temperature at subdaily timescale
#     vcmax_subdaily = vcmax25_subdaily * calc_ftemp_arrh(tk=tk_subdaily, ha=ha_vcmax25)
#     jmax_subdaily = jmax25_subdaily * calc_ftemp_arrh(tk=tk_subdaily, ha=ha_jmax25)

#     # Interpolate xi to subdaily scale
#     xi_subdaily = fsscaler.fill_daily_to_subdaily(xi_real)

#     # Calculate the optimal chi, imposing the realised xi values
#     subdaily_chi = OptimalChiPrentice14(env=subdaily_env)
#     subdaily_chi.estimate_chi(xi_values=xi_subdaily)

#     # Calculate Ac
#     Ac_subdaily = (
#         vcmax_subdaily
#         * (subdaily_chi.ci - subdaily_env.gammastar)
#         / (subdaily_chi.ci + subdaily_env.kmm)
#     )

#     # Calculate J and Aj
#     phi = (1 / 8) * calc_ftemp_kphio(tc=temp_subdaily)
#     iabs = fpar_subdaily * ppfd_subdaily

#     J_subdaily = (4 * phi * iabs) / np.sqrt(1 + ((4 * phi * iabs) / jmax_subdaily) ** 2)

#     Aj_subdaily = (
#         (J_subdaily / 4)
#         * (subdaily_chi.ci - subdaily_env.gammastar)
#         / (subdaily_chi.ci + 2 * subdaily_env.gammastar)
#     )

#     # Calculate GPP and convert from micromols to micrograms
#     GPP_subdaily = (
#         np.minimum(Ac_subdaily, Aj_subdaily)
#         * pmodel_subdaily.env.core_const.k_c_molmass
#     )

#     gC_to_mumol = 0.0833  # 1 µg C m⁻² s⁻¹ × (1 µmol C / 12.01 µg C) × (1 µmol CO₂ / 1 µmol C) = 0.0833 µmol CO₂ m⁻² s⁻¹
#     GPP_subdaily *= gC_to_mumol
#     # print(
#     #     f"GPPmean {np.nanmean(GPP_subdaily)} at {days_memory} days_mem at {window_center_i}h {half_width_i}m"
#     # )

#     return GPP_subdaily  # ,datetime_subdaily
