import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
import xarray as xr
from pyrealm.pmodel import PModel, PModelEnvironment
from pyrealm.splash.splash import SplashModel
from pyrealm.core.calendar import Calendar
import pyrealm.pmodel
from pyrealm.core.pressure import calc_patm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
import cftime


def pModel_subdaily(df_site_and_modis):

    # Extract the key half hourly timestep variables as numpy arrays
    site_temp = df_site_and_modis.loc[:, "TA_F"]  # daily temperature, unit degree (°C)
    site_temp[site_temp < -25] = np.nan
    # site_swin = df_site_and_modis.loc[:, 'SW_IN_F_MDS']  # shortwave radiation, unit W/m2
    site_ppfd = df_site_and_modis.loc[
        :, "PPFD_IN"
    ]  # Shortwave radiation (W/m²) × 0.50 -> PAR (W/m²) × 4.6 -> PPFD (umol/m²/s)
    site_vpd = df_site_and_modis.loc[:, "VPD_F"] * 100  # vpd, unit: hPa converted to Pa
    site_co2 = df_site_and_modis.loc[:, "CO2_F_MDS"]  # CO2, unit ppm
    site_patm = (
        df_site_and_modis.loc[:, "PA_F"] * 1000
    )  # site pressure, unit kPa converted to Pa
    # site_prep = df_site_and_modis.loc[:,'P_F']
    # rename column starting with 'Fpar_500m_(some Date)' to 'Fpar_500m'
    df_site_and_modis.columns = df_site_and_modis.columns.str.replace(
        r"Fpar_500m_\d{4}-\d{2}-\d{2}", "Fpar_500m", regex=True
    )
    site_fapar = (
        df_site_and_modis.loc[:, "Fpar_500m"] * 100
    )  # 100 to convert to percent
    # clip to zero
    site_vpd.values[site_vpd.values < 0] = 0
    site_ppfd.values[site_ppfd.values < 0] = 0

    # Calculate the photosynthetic environment
    subdaily_env = pyrealm.pmodel.PModelEnvironment(
        tc=site_temp.values,
        vpd=site_vpd.values,
        co2=site_co2.values,
        patm=site_patm.values,
    )

    # Create the fast slow scaler
    datetime_subdaily = pd.to_datetime(df_site_and_modis["TIMESTAMP_START"]).to_numpy()
    fsscaler = pyrealm.pmodel.SubdailyScaler(datetime_subdaily)

    # Set the acclimation window as the values within a one hour window centred on noon
    fsscaler.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(30, "m"),
    )

    # Fit the P Model with fast and slow responses
    pmodel_subdaily = pyrealm.pmodel.SubdailyPModel(
        env=subdaily_env,
        fs_scaler=fsscaler,
        allow_holdover=True,
        ppfd=site_ppfd.values,
        fapar=site_fapar.values,
    )

    gC_to_mumol = 0.0833  # 1 µg C m⁻² s⁻¹ × (1 µmol C / 12.01 µg C) × (1 µmol CO₂ / 1 µmol C) = 0.0833 µmol CO₂ m⁻² s⁻¹
    pmodel_subdaily.gpp *= gC_to_mumol

    return pmodel_subdaily.gpp  # ,datetime_subdaily
