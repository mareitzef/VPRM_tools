import pandas as pd
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from sklearn.metrics import r2_score, mean_squared_error
from VPRM import (
    VPRM_old,
    VPRM_old_only_Reco,
    VPRM_new,
    VPRM_new_only_Reco,
    VPRM_new_only_GPP,
)
from plots_for_VPRM import (
    plot_measured_vs_optimized_VPRM,
    plot_site_input,
    plot_measured_vs_modeled,
)
import argparse
import sys


############################## general functions #############################################
def calculate_NSE(observed_values, predicted_values):
    """
    Calculate the Nash-Sutcliffe Efficiency (NSE).

    Parameters:
        observed_values (array-like): Observed values.
        predicted_values (array-like): Predicted values.

    Returns:
        float: Nash-Sutcliffe Efficiency (NSE) value.
    """
    # Calculate mean of observed values
    mean_observed = np.mean(observed_values)

    # Calculate numerator and denominator for NSE
    numerator = np.sum((observed_values - predicted_values) ** 2)
    denominator = np.sum((observed_values - mean_observed) ** 2)

    # Calculate NSE
    NSE = 1 - (numerator / denominator)

    return NSE


def calculate_AIC(n, mse, k):
    """
    Calculate Akaike Information Criterion (AIC).

    Parameters:
    - n: Number of data points.
    - mse: Mean squared error of the model.
    - k: Number of parameters in the model.

    Returns:
    - AIC value.
    """
    AIC = 2 * k - np.log(mse) + 2 * k * (k + 1) / (n - k - 1)
    return AIC


def migliavacca_LinGPP(
    T_ref, T0, E0, k_mm, k2, alpha_p, alpha_lai, max_lai, R_lai0, GPP, P, T_A
):
    GPP_gC_per_day = GPP * 12 * 86400 / 10**6
    R_ref = R_lai0 + alpha_lai * max_lai + k2 * GPP_gC_per_day
    f_T = np.exp(E0 * (1 / (T_ref - T0) - 1 / (T_A - T0)))
    f_P = (alpha_p * k_mm + P * (1 - alpha_p)) / (k_mm + P * (1 - alpha_p))

    reco_LinGPP_gC_per_day = R_ref * f_T * f_P
    reco_LinGPP_mumol_per_sec = reco_LinGPP_gC_per_day * 10**6 / (86400 * 12)

    # conversion factor is identical to R code below
    # library(bigleaf)
    # GPP_umol_CO2_per_m2_per_s <- 10
    # GPP_gC <- umolCO2.to.gC(GPP_umol_CO2_per_m2_per_s)
    # conv_factor <- GPP_umol_CO2_per_m2_per_s / GPP_gC
    # cat("Conversion factor:", conv_factor, "\n")

    return reco_LinGPP_mumol_per_sec


def main():
    """
    This function serves as the main entry point for running the VPRM optimization process.
    It accepts command-line arguments for specifying the base path, folder, maximum iteration,
    optimization method, and VPRM version. If no arguments are provided, default values are used.
    The script reads site information from a CSV file and iterates over each site folder to perform
    VPRM optimization for each site-year combination. It retrieves FLUXNET and MODIS data,
    preprocesses them, and initializes VPRM parameters based on the specified VPRM version.
    Optimization is carried out using differential evolution algorithm. Results are saved
    as Excel files, and plots are generated for visualization and analysis.

    Versions:
    V1 - tests
    V3 - still with bugs
    V4 - final version
    V5 - using CUT instead of VUT method
    V6 - testing NEE_VUT_MEAN for Reco calibration
    V7 - with RECO and GPP FLUXNET tuning and impvoed plots and testing lower boundary of T_opt till -5 and initial T_opt = T_mean - 5
        -> there was a bug in GPP_calc for VPRM new, I used RECO correctly but not GPP, for VPRM_old it was correct
    V8 - drop years with not enough data and removed uncertainty and -5 border again and still using RECO and GPP FLUXNET tuning
    V9 - back to using using NEE not tuning (not RECO and GPP) and keep initial T_opt = T_mean -5 with min of 1°
    V10 - defining min boundary of Topt from analysis of FLUXNET data
    V11 - optimize migliacacca R_eco
    V12 - using get_global_bounds_for_migliavacca to find better solutions
    V12 - using direct RECO and GPP FLUXNET tuning again as in V7/8
    """

    if len(sys.argv) > 1:  # to run all on cluster with 'submit_jobs_tune_VPRM.sh'
        parser = argparse.ArgumentParser(description="Description of your script")
        parser.add_argument("-p", "--base_path", type=str, help="Base path argument")
        parser.add_argument("-f", "--folder", type=str, help="Folder argument")
        parser.add_argument("-i", "--maxiter", type=int, help="Max iteration argument")
        parser.add_argument(
            "-m", "--opt_method", type=str, help="Optimization method argument"
        )
        parser.add_argument(
            "-v",
            "--CO2_parametrization",
            type=str,
            help="chose a model for CO2 parametrization",
        )
        args = parser.parse_args()

        base_path = args.base_path
        maxiter = args.maxiter
        opt_method = args.opt_method
        CO2_parametrization = args.CO2_parametrization
        folder = args.folder
        single_year = False  # only for local testing
        year_to_plot = 2000  # only for local testing
    else:  # to run locally for single cases
        base_path = "/home/madse/Downloads/Fluxnet_Data/"
        maxiter = 1  # (default=100 takes ages)
        opt_method = "diff_evo_V13"  # version of diff evo
        CO2_parametrization = "old"  # "old","new", "migli"
        folder = "FLX_IT-Tor_FLUXNET2015_FULLSET_2008-2014_2-4"
        single_year = True  # True for local testing, default=False
        year_to_plot = 2014

    VEGFRA = 1  # not applied for EC measurements, set to 1
    site_info = pd.read_csv(base_path + "site_info_all_FLUXNET2015.csv")
    # Reco is optomized against NEE at night as it is measured directly
    # in FLUXNET Reco and GPP are seperated by a model

    ###########################################################################################

    ####################################### define  functions #################################

    def objective_function_VPRM_old_Reco(x):
        alpha, beta = x
        Reco_VPRM = VPRM_old_only_Reco(
            alpha,
            beta,
            T2M,
        )

        residuals_Reco = (np.array(Reco_VPRM) - df_year[reco_from_nee]) * df_year[night]
        return np.sum(residuals_Reco**2)

    def objective_function_VPRM_old_GPP(x):
        Topt, PAR0, lambd = x
        GPP_VPRM, Reco_VPRM = VPRM_old(
            Topt,
            PAR0,
            alpha,
            beta,
            lambd,
            Tmin,
            Tmax,
            T2M,
            LSWI,
            LSWI_min,
            LSWI_max,
            EVI,
            PAR,
            VPRM_veg_ID,
            VEGFRA,
        )
        # it is optomized against NEE as it is measured directly, in FLUXNET Reco and GPP are seperated by a model
        residuals_GPP = np.array(GPP_VPRM) - df_year["GPP_calc"]
        del Reco_VPRM
        return np.sum(residuals_GPP**2)

    def objective_function_VPRM_new_Reco(x):
        (
            beta,
            T_crit,
            T_mult,
            alpha1,
            alpha2,
            gamma,
            theta1,
            theta2,
            theta3,
        ) = x
        Reco_VPRM = VPRM_new_only_Reco(
            beta,
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
        )
        residuals_Reco = (np.array(Reco_VPRM) - df_year[reco_from_nee]) * df_year[night]
        return np.sum(residuals_Reco**2)

    def objective_function_VPRM_new_GPP(x):
        (
            Topt,
            PAR0,
            lambd,
        ) = x
        GPP_VPRM = VPRM_new_only_GPP(
            Topt,
            PAR0,
            lambd,
            Tmin,
            Tmax,
            T2M,
            LSWI,
            LSWI_min,
            LSWI_max,
            EVI,
            PAR,
            VPRM_veg_ID,
            VEGFRA,
        )

        residuals_GPP = np.array(GPP_VPRM) - df_year["GPP_calc"]
        return np.sum(residuals_GPP**2)

    def objective_function_migliavacca_LinGPP(params):
        (R_lai0, alpha_lai, k2, E0, alpha_p, k_mm) = params

        reco_LinGPP = migliavacca_LinGPP(
            T_ref,
            T0,
            E0,
            k_mm,
            k2,
            alpha_p,
            alpha_lai,
            max_lai,
            R_lai0,
            df_year["GPP_avrg"],
            df_year["P_avrg"],
            T2M + 273.15,
        )
        # residuals = (reco_LinGPP - df_year[reco_from_nee]) * df_year[night]
        residuals = reco_LinGPP - df_year[reco_from_nee]

        valid_indices = ~np.isnan(residuals)
        abs_error = residuals[valid_indices]

        return np.sum(abs_error**2)

    def get_Topt_per_site(df_site):
        # Clean the data
        df_site["TA_F"] = df_site["TA_F"].replace(-9999.0, np.nan)
        df_site = df_site.dropna(subset=["TA_F"])
        df_site["GPP_DT_VUT_REF"] = df_site["GPP_DT_VUT_REF"].replace(-9999.0, np.nan)
        df_site = df_site.dropna(subset=["GPP_DT_VUT_REF"])

        # Set the values to np.nan during nighttime

        df_daily = (
            df_site.resample("D")
            .agg({"TA_F": "mean", "GPP_DT_VUT_REF": "mean"})
            .dropna()
        )
        df_daily["YEAR"] = df_daily.index.year
        years = df_daily["YEAR"].unique()

        real_Topt = []
        min_Topt = []
        for year in years:
            df_year_Topt = df_daily[df_daily["YEAR"] == year].copy()
            # Group by each degree of temperature and calculate the mean values
            df_year_Topt.loc[:, "TA_F_rounded"] = df_year_Topt["TA_F"].round(1)
            mean_values = df_year_Topt.groupby("TA_F_rounded").mean()
            coeffs = np.polyfit(mean_values.index, mean_values["GPP_DT_VUT_REF"], 2)
            poly_fit = np.polyval(coeffs, mean_values.index)
            max_index = mean_values.index[np.argmax(poly_fit)]
            if max_index < 0:
                max_index = np.nan
            max_value = poly_fit[np.argmax(poly_fit)]

            if max_index < df_year_Topt["TA_F_rounded"].max():
                print(
                    f"Topt={max_index} is real with GPP={max_value.round(2)} for {year} at Site {site_name}"
                )
                real_Topt.append(max_index)
            else:
                print(
                    f"Topt={max_index} is a minimum with GPP={max_value.round(1)} for {year} at Site {site_name}"
                )
                min_Topt.append(max_index)
        if len(real_Topt) > len(min_Topt):
            real_Topt_flag = True
            Topt_min = np.mean(real_Topt)
            Topt_max = np.mean(real_Topt)
        else:
            real_Topt_flag = False
            Topt_min = np.mean(min_Topt)
            Topt_max = 50
        print(f"Topt_min={Topt_min.round(1)} - Topt_max={Topt_max} at Site {site_name}")
        return Topt_min, Topt_max, real_Topt_flag

    ###########################################################################################

    ################################# loop over the sites #####################################

    optimized_params_df_all = pd.DataFrame()

    print(folder)

    site_name = "_".join(folder.split("_")[1:2])
    file_base = "_".join(folder.split("_")[0:4])
    years = "_".join(folder.split("_")[4:6])
    file_path = base_path + folder + "/" + file_base + "_HH_" + years + ".csv"

    # get PFT of site
    i = 0
    for site_i in site_info["site"]:
        if site_name == site_i:
            target_pft = site_info["pft"][i]
            if target_pft == "EBF":
                target_pft = "DBF"  # TODO check OSH
            if target_pft == "OSH":
                target_pft = "SHB"
            latitude = site_info["lat"][i].replace(",", ".")
            longitude = site_info["lon"][i].replace(",", ".")
            elevation = site_info["elev"][i]
        i += 1

    ############################### Read FLUXNET Data ##############################################
    XUT = "VUT"  # VUT or CUT method
    timestamp = "TIMESTAMP_START"
    t_air = "TA_F"  # Air temperature, consolidated from TA_F_MDS and TA_ERA
    gpp = (
        "GPP_DT_" + XUT + "_REF"
    )  # Gross Primary Production, from Daytime partitioning method, reference selected from GPP versions using model efficiency (MEF).
    r_eco = (
        "RECO_DT_" + XUT + "_REF"
    )  # is just used for plotting and R2 out of interest
    nee = (
        "NEE_" + XUT + "_REF"
    )  # Net Ecosystem Exchange, using Variable Ustar Threshold (VUT) for each year, reference selected on the basis of the model efficiency (MEF).
    nee_qc = "NEE_" + XUT + "_REF_QC"  # Quality flag for NEE_VUT_REF
    night = "NIGHT"  # flag for nighttime

    reco_from_nee = "NEE_" + XUT + "_REF_pos"  # only positive values from NEE_VUT_REF
    # test for "RECO_DT_" + XUT + "_REF" # V5 tested XUT vs VUT
    reco_from_nee = r_eco  # for V7/8/13
    # reco_from_nee = "NEE_" + XUT + "_MEAN"  # V6: tested if NEE_VUT_MEAN hat better R2
    sw_in = "SW_IN_F"  # Shortwave radiation, incoming consolidated from SW_IN_F_MDS and SW_IN_ERA (negative values set to zero)
    columns_to_copy = [
        timestamp,
        night,
        t_air,
        gpp,
        r_eco,
        nee,
        sw_in,
        nee_qc,
        "P",
        # reco_from_nee,  # TODO uncomment for using NEE_VUT_MEAN
    ]
    converters = {k: lambda x: float(x) for k in columns_to_copy}
    df_site = pd.read_csv(file_path, usecols=columns_to_copy, converters=converters)
    df_site[timestamp] = pd.to_datetime(df_site[timestamp], format="%Y%m%d%H%M")
    # uncomment to plot single years
    if single_year:
        df_site = df_site[df_site[timestamp].dt.year == year_to_plot]
    df_site.set_index(timestamp, inplace=True)
    modis_path = base_path + folder + "/"

    ##################################### Check data #########################################

    def filter_nan(df_site, var_i):
        df_site.loc[df_site[var_i] == -9999, var_i] = np.nan
        return df_site

    for var_i in columns_to_copy[2:]:
        df_site = filter_nan(df_site, var_i)

    df_site.loc[df_site[t_air] < -40, t_air] = np.nan
    df_site.loc[df_site[sw_in] < 0, sw_in] = np.nan

    # Conversion factors
    PAR_conversion = 0.505  #  global radiation is proportional to PAR (Rg = 0.505*PAR - Mahadevan 2008)

    df_site["PAR"] = df_site[sw_in] / PAR_conversion
    df_site.drop(columns=[sw_in], inplace=True)

    # averages of precipitation and GPP
    # df_site["P_avrg"] = df_site["P"].rolling(window="7D").mean().fillna(0)
    # df_site["GPP_avrg"] = df_site[gpp].rolling(window="1D").mean().fillna(0)

    df_site["P_avrg"] = df_site["P"].rolling(window="7D").mean().interpolate()
    initial_mean = df_site["P"].iloc[: 7 * 48].mean()
    df_site["P_avrg"].iloc[: 7 * 48] = (
        df_site["P_avrg"].iloc[: 7 * 48].fillna(initial_mean)
    )
    final_mean = df_site["P"].iloc[-7 * 48 :].mean()
    df_site["P_avrg"].iloc[-7 * 48 :] = (
        df_site["P_avrg"].iloc[-7 * 48 :].fillna(final_mean)
    )
    df_site["P_avrg"] = df_site["P_avrg"].fillna(0)

    df_site["GPP_avrg"] = df_site[gpp].rolling(window="1D").mean().interpolate()
    initial_mean = df_site[gpp].iloc[:48].mean()
    df_site["GPP_avrg"].iloc[:48] = df_site["GPP_avrg"].iloc[:48].fillna(initial_mean)
    final_mean = df_site[gpp].iloc[-48:].mean()
    df_site["GPP_avrg"].iloc[-48:] = df_site["GPP_avrg"].iloc[-48:].fillna(final_mean)
    df_site["GPP_avrg"] = df_site["GPP_avrg"].fillna(0)

    ##################################### read  MODIS data ##################################

    mod_files = glob(os.path.join(modis_path, "*_MOD*.xlsx"))
    myd_files = glob(os.path.join(modis_path, "*_MYD*.xlsx"))

    # Process MOD and MYD files
    dfs_modis = []
    for file_list in [mod_files, myd_files]:
        for file_path in file_list:
            df = pd.read_excel(file_path)[["calendar_date", "value"]].assign(
                value=lambda x: x["value"] * 0.0001  # TODO: scale Lai diffently
            )  # sclaing factor from user guide: https://lpdaac.usgs.gov/documents/103/MOD13_User_Guide_V6.pdf
            file_parts = file_path.split("/")[-1]
            file_parts2 = file_parts.split("_")[2:6]
            if file_parts2[0] == "250m":
                file_type = "_".join(file_parts2)
            else:
                file_type = "_".join(file_parts2[:-1])
            df.rename(columns={"value": file_type}, inplace=True)
            dfs_modis.append(df)

    # Merge dataframes based on 'calendar_date'
    df_modis = dfs_modis[0]
    for df in dfs_modis[1:]:
        df_modis = pd.merge(
            df_modis, df, on="calendar_date", how="outer", suffixes=("_x", "_y")
        )

    for column in df_modis.columns:
        if column.endswith("_x"):
            base_column = column[:-2]  # Remove suffix '_x'
            if base_column + "_y" in df_modis.columns:
                df_modis[column].fillna(df_modis[base_column + "_y"], inplace=True)
                df_modis.drop(columns=[base_column + "_y"], inplace=True)
                df_modis.rename(columns={column: base_column}, inplace=True)

    df_modis.sort_values(by="calendar_date", inplace=True)
    df_modis.reset_index(drop=True, inplace=True)
    df_modis["calendar_date"] = pd.to_datetime(df_modis["calendar_date"])
    df_modis.set_index("calendar_date", inplace=True)
    # Interpolate the DataFrame linearly to fill missing values and resample to hourly frequency
    df_modis.loc[df_modis["sur_refl_b02"] < 0, "sur_refl_b02"] = np.nan
    df_modis.loc[df_modis["sur_refl_b06"] < 0, "sur_refl_b06"] = np.nan
    df_modis["250m_16_days_EVI"] = df_modis["250m_16_days_EVI"]
    df_modis_intp = df_modis.resample("30T").interpolate(
        method="linear"
    )  # TODO: create 30 min timeseries here
    df_modis_intp.reset_index(inplace=True)
    df_modis_intp.rename(columns={"calendar_date": timestamp}, inplace=True)
    df_modis_intp.set_index(timestamp, inplace=True)

    # Perform inner join to keep only matching dates
    df_site_and_modis = pd.merge(
        df_site, df_modis_intp, left_index=True, right_index=True, how="inner"
    )
    df_site_and_modis.reset_index(inplace=True)

    ############################# prepare input variables  #############################
    # just use fluxnet qualities 0 and 1 - new in V3 - in V11 > 2 again  TODO: test when to use
    df_site_and_modis.loc[df_site_and_modis[nee_qc] > 2, nee] = (
        np.nan
    )  # 0 = measured; 1 = good quality gapfill; 2 = medium; 3 = poor
    # create extra column for daytime NEE
    # TODO uncomment next 3 lines, if using NEE_VUT_REF - make automatic if needed later..
    # df_site_and_modis[reco_from_nee] = df_site_and_modis[nee].copy()
    # # only the respiration of reco_from_nee is used
    # df_site_and_modis.loc[df_site_and_modis[reco_from_nee] < 0, reco_from_nee] = 0

    # calculate LSWI from MODIS Bands 2 and 6
    df_site_and_modis["LSWI"] = (
        df_site_and_modis["sur_refl_b02"] - df_site_and_modis["sur_refl_b06"]
    ) / (df_site_and_modis["sur_refl_b02"] + df_site_and_modis["sur_refl_b06"])
    T2M = df_site_and_modis[t_air]
    EVI = df_site_and_modis["250m_16_days_EVI"]
    PAR = df_site_and_modis["PAR"]
    LSWI = df_site_and_modis["LSWI"]
    df_site_and_modis["P"] = df_site_and_modis["P"].fillna(0)

    # LSWImax is the site-specific multiyear maximum daily LSWI from May to October
    df_may_to_october = df_site_and_modis[
        (df_site_and_modis[timestamp].dt.month >= 5)
        & (df_site_and_modis[timestamp].dt.month <= 10)
    ]
    max_lswi_by_year = df_may_to_october.groupby(df_may_to_october[timestamp].dt.year)[
        "LSWI"
    ].max()
    LSWI_max = max_lswi_by_year.max()
    LSWI_min = max(
        0, df_site_and_modis["LSWI"].min()
    )  # TODO: site-specific minimum LSWI across a full year  (from a multi-year mean)

    variables = [
        nee,
        reco_from_nee,
        gpp,
        r_eco,
        t_air,
        "PAR",
        "LSWI",
        "250m_16_days_EVI",
        "GPP_avrg",
        "P_avrg",
    ]

    # Parameters are set constant for physical reason of no PSN above and below (true for the Alps)
    Tmin = 0
    Tmax = 45

    Topt_min, Topt_max, real_Topt_flag = get_Topt_per_site(df_site)

    #############################  first guess  of parameters #########################
    # adopted from VPRM_table_Europe with values for Wetland from Gourdji 2022
    if CO2_parametrization == "old" or CO2_parametrization == "migli":
        VPRM_table_first_guess = {
            "PFT": ["ENF", "DBF", "MF", "SHB", "WET", "CRO", "GRA"],
            "VPRM_veg_ID": [1, 2, 3, 4, 5, 6, 7],
            "PAR0": [270.2, 271.4, 236.6, 363.0, 579, 690.3, 229.1],
            "lambda": [
                -0.3084,
                -0.1955,
                -0.2856,
                -0.0874,
                -0.0752,
                -0.1350,
                -0.1748,
            ],
            "alpha": [0.1797, 0.1495, 0.2258, 0.0239, 0.111, 0.1699, 0.0881],
            "beta": [0.8800, 0.8233, 0.4321, 0.0000, 0.82, -0.0144, 0.5843],
            "T_opt": [
                Topt_min,
                Topt_min,
                Topt_min,
                Topt_min,
                Topt_min,
                Topt_min,
                Topt_min,
            ],
        }
    ###################### table from Gourdji 2022 for VPRM_new ############################
    elif CO2_parametrization == "new":
        VPRM_table_first_guess = {
            "PFT": ["DBF", "ENF", "MF", "SHB", "GRA", "WET", "CRO", "CRC"],
            "VPRM_veg_ID": [2, 1, 3, 4, 7, 5, 6, 8],
            "T_crit": [-15, 1, 0, 5, 11, 6, 7, -1],
            "T_mult": [0.55, 0.05, 0.05, 0.1, 0.1, 0.1, 0.05, 0],
            "lambda": [
                -0.1023,
                -0.1097,
                -0.1097,
                -0.0996,
                -0.1273,
                -0.1227,
                -0.0732,
                -0.0997,
            ],
            "PAR0": [539, 506, 506, 811, 673, 456, 1019, 1829],
            "beta": [0.12, 0.47, 0.47, 1.53, -6.18, -0.82, -1.20, -0.02],
            "alpha1": [0.065, 0.088, 0.088, 0.004, 0.853, 0.261, 0.234, 0.083],
            "alpha2": [
                0.0024,
                0.0047,
                0.0047,
                0.0049,
                -0.0250,
                -0.0051,
                -0.0060,
                -0.0018,
            ],
            "gamma": [4.61, 1.39, 1.39, 0.09, 5.19, 3.46, 3.85, 4.89],
            "theta1": [
                0.116,
                -0.530,
                -0.530,
                -1.787,
                1.749,
                -0.7,
                0.032,
                0.150,
            ],
            "theta2": [
                -0.0005,
                0.2063,
                0.2063,
                0.4537,
                -0.2829,
                0.0990,
                -0.0429,
                -0.1324,
            ],
            "theta3": [
                0.0009,
                -0.0054,
                -0.0054,
                -0.0138,
                0.0166,
                0.0018,
                0.0090,
                0.0156,
            ],
            "T_opt": [
                Topt_min,
                Topt_min,
                Topt_min,
                Topt_min,
                Topt_min,
                Topt_min,
                Topt_min,
                Topt_min,
            ],  # TODO: define for Europe
        }

    if CO2_parametrization == "migli":
        # Initial values and boundaries for all PFTs
        boundaries = {
            "ENF": {
                "RLAI": (1.02, 0.42),
                "alphaLAI": (0.42, 0.08),
                "k2": (0.478, 0.013),
                "E0(K)": (124.833, 4.656),
                "alpha_p": (0.604, 0.065),
                "K (mm)": (0.222, 0.070),
            },
            "DBF": {
                "RLAI": (1.27, 0.50),
                "alphaLAI": (0.34, 0.10),
                "k2": (0.247, 0.009),
                "E0(K)": (87.655, 4.405),
                "alpha_p": (0.796, 0.031),
                "K (mm)": (0.184, 0.064),
            },
            "GRA": {
                "RLAI": (0.41, 0.71),
                "alphaLAI": (1.14, 0.33),
                "k2": (0.578, 0.062),
                "E0(K)": (101.181, 6.362),
                "alpha_p": (0.670, 0.052),
                "K (mm)": (0.765, 1.589),
            },
            "CRO": {
                "RLAI": (0.25, 0.66),
                "alphaLAI": (0.40, 0.11),
                "k2": (0.244, 0.016),
                "E0(K)": (129.498, 5.618),
                "alpha_p": (0.934, 0.065),
                "K (mm)": (0.035, 3.018),
            },
            "SAV": {
                "RLAI": (0.42, 0.39),
                "alphaLAI": (0.57, 0.17),
                "k2": (0.654, 0.024),
                "E0(K)": (81.537, 7.030),
                "alpha_p": (0.474, 0.018),
                "K (mm)": (0.567, 0.119),
            },
            "SHB": {
                "RLAI": (0.42, 0.39),
                "alphaLAI": (0.57, 0.17),
                "k2": (0.354, 0.021),
                "E0(K)": (156.746, 8.222),
                "alpha_p": (0.850, 0.070),
                "K (mm)": (0.097, 1.304),
            },
            "EBF": {
                "RLAI": (-0.47, 0.50),
                "alphaLAI": (0.82, 0.13),
                "k2": (0.602, 0.044),
                "E0(K)": (52.753, 4.351),
                "alpha_p": (0.593, 0.032),
                "K (mm)": (2.019, 1.052),
            },
            "MF": {
                "RLAI": (0.78, 0.18),
                "alphaLAI": (0.44, 0.04),
                "k2": (0.391, 0.068),
                "E0(K)": (176.542, 8.222),
                "alpha_p": (0.703, 0.083),
                "K (mm)": (2.831, 4.847),
            },
            "WET": {
                "RLAI": (0.78, 0.18),
                "alphaLAI": (0.44, 0.04),
                "k2": (0.398, 0.013),
                "E0(K)": (144.705, 8.762),
                "alpha_p": (0.582, 0.163),
                "K (mm)": (0.054, 0.593),
            },
        }

        def get_bounds_for_migliavacca(boundaries, pft):
            return [
                (
                    boundaries[pft]["RLAI"][0] - 2 * boundaries[pft]["RLAI"][1],
                    boundaries[pft]["RLAI"][0] + 2 * boundaries[pft]["RLAI"][1],
                ),
                (
                    boundaries[pft]["alphaLAI"][0] - 2 * boundaries[pft]["alphaLAI"][1],
                    boundaries[pft]["alphaLAI"][0] + 2 * boundaries[pft]["alphaLAI"][1],
                ),
                (
                    boundaries[pft]["k2"][0] - 2 * boundaries[pft]["k2"][1],
                    boundaries[pft]["k2"][0] + 2 * boundaries[pft]["k2"][1],
                ),
                (
                    boundaries[pft]["E0(K)"][0] - 2 * boundaries[pft]["E0(K)"][1],
                    boundaries[pft]["E0(K)"][0] + 2 * boundaries[pft]["E0(K)"][1],
                ),
                (
                    boundaries[pft]["alpha_p"][0] - 2 * boundaries[pft]["alpha_p"][1],
                    boundaries[pft]["alpha_p"][0] + 2 * boundaries[pft]["alpha_p"][1],
                ),
                (
                    boundaries[pft]["K (mm)"][0] - 2 * boundaries[pft]["K (mm)"][1],
                    boundaries[pft]["K (mm)"][0] + 2 * boundaries[pft]["K (mm)"][1],
                ),
            ]

        def get_global_bounds_for_migliavacca(boundaries):
            global_boundaries = {}

            # Iterate through each parameter to find global min, max, and adjusted boundaries
            for parameter in ["RLAI", "alphaLAI", "k2", "E0(K)", "alpha_p", "K (mm)"]:
                values = []
                stds = []

                # absolute min and max with 3 std
                for pft, params in boundaries.items():
                    value, std = params[parameter]
                    values.append(value)
                    stds.append(std)
                    global_min = min(values)
                    global_max = max(values)
                    max_std = max(stds)
                    if parameter != "RLAI":
                        global_min = max(global_min, 0)

                lower_bound = global_min - 2 * max_std
                upper_bound = global_max + 2 * max_std
                if parameter != "RLAI":
                    lower_bound = max(lower_bound, 0)
                if parameter == "alpha_p":
                    upper_bound = min(upper_bound, 1)
                if parameter == "K (mm)":
                    lower_bound = max(lower_bound, 0.01)
                    upper_bound = min(upper_bound, 10)

                # maximum bounds do not work
                # if parameter != "RLAI":
                #     lower_bound = -5
                #     upper_bound = 10
                # if parameter != "alphaLAI":
                #     lower_bound = 0
                #     upper_bound = 1
                # if parameter != "k2":
                #     lower_bound = 0.1
                #     upper_bound = 2
                # if parameter != "E0(K)":
                #     lower_bound = 20
                #     upper_bound = 300
                # if parameter == "alpha_p":
                #     lower_bound = 0
                #     upper_bound = 1
                # if parameter == "K (mm)":
                #     lower_bound = 0
                #     upper_bound = 100

                global_boundaries[parameter] = (lower_bound, upper_bound)

            return list(global_boundaries.values())

    else:
        print("ERROR - you have to choose a model")

    ###################### select first guess according to PFT ##########################

    # Topt = 20.0  # T_opt was constant before V8
    if CO2_parametrization == "old" or CO2_parametrization == "migli":
        df_VPRM_table_first_guess = pd.DataFrame(VPRM_table_first_guess)
        parameters = df_VPRM_table_first_guess[
            df_VPRM_table_first_guess["PFT"] == target_pft
        ].iloc[0]
        PAR0 = parameters["PAR0"]
        alpha = parameters["alpha"]
        beta = parameters["beta"]
        lambd = -parameters["lambda"]
        VPRM_veg_ID = parameters["VPRM_veg_ID"]
        Topt = parameters["T_opt"]
    elif CO2_parametrization == "new":
        df_VPRM_table_first_guess = pd.DataFrame(VPRM_table_first_guess)
        parameters = df_VPRM_table_first_guess[
            df_VPRM_table_first_guess["PFT"] == target_pft
        ].iloc[0]
        PAR0 = parameters["PAR0"]
        beta = parameters["beta"]
        lambd = -parameters["lambda"]
        VPRM_veg_ID = parameters["VPRM_veg_ID"]
        T_crit = parameters["T_crit"]
        T_mult = parameters["T_mult"]
        alpha1 = parameters["alpha1"]
        alpha2 = parameters["alpha2"]
        gamma = parameters["gamma"]
        theta1 = parameters["theta1"]
        theta2 = parameters["theta2"]
        theta3 = parameters["theta3"]
        Topt = parameters["T_opt"]

    ###################### run VPRM with first Guess for comparison plot #################
    if CO2_parametrization == "old":
        GPP_VPRM, Reco_VPRM = VPRM_old(
            Topt,
            PAR0,
            alpha,
            beta,
            lambd,
            Tmin,
            Tmax,
            T2M,
            LSWI,
            LSWI_min,
            LSWI_max,
            EVI,
            PAR,
            VPRM_veg_ID,
            VEGFRA,
        )
        df_site_and_modis["GPP_first_guess"] = GPP_VPRM
        if (
            folder == "FLX_IT-Tor_FLUXNET2015_FULLSET_2008-2014_2-4"
        ):  # cut values for plotting
            df_site_and_modis.loc[
                df_site_and_modis["GPP_first_guess"] > 30, "GPP_first_guess"
            ] = np.nan
        df_site_and_modis["Reco_first_guess"] = Reco_VPRM
    elif CO2_parametrization == "new":
        GPP_VPRM, Reco_VPRM = VPRM_new(
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
            Tmin,
            Tmax,
            T2M,
            LSWI,
            LSWI_min,
            LSWI_max,
            EVI,
            PAR,
            VPRM_veg_ID,
            VEGFRA,
        )
        df_site_and_modis["GPP_first_guess"] = GPP_VPRM
        if (
            folder == "FLX_IT-Tor_FLUXNET2015_FULLSET_2008-2014_2-4"
        ):  # cut values for plotting
            df_site_and_modis.loc[
                df_site_and_modis["GPP_first_guess"] > 30, "GPP_first_guess"
            ] = np.nan
        df_site_and_modis["Reco_first_guess"] = Reco_VPRM
    elif CO2_parametrization == "migli":
        T_ref = 288.15
        T0 = 227.13
        lai_column = [
            col for col in df_site_and_modis.columns if col.startswith("Lai_")
        ]
        max_lai = (
            df_site_and_modis[lai_column].max()
            * 1000  # TODO: use correct scale 0.1 directly, now its scale from LSWI*1000=0.1
        )
        max_lai = float(max_lai)

        # df_site_and_modis["TIMESTAMP_START"] = pd.to_datetime(
        #     df_site_and_modis["TIMESTAMP_START"]
        # )
        # df_site_and_modis.set_index("TIMESTAMP_START", inplace=True)
        # # gpp_daily_mean_all = (
        # #     df_site_and_modis[gpp]
        # #     .resample("D")
        # #     .mean()
        # #     .reindex(df_site_and_modis.index, method="pad")
        # # )  # Resample and reindex the 'gpp' column with daily frequency
        # # gpp_daily_mean_all.reset_index(drop=True, inplace=True)
        # P_3d_avg_all = (
        #     df_site_and_modis["P"]
        #     .rolling(window="3D")
        #     .mean()
        #     .reindex(df_site_and_modis.index, method="pad")
        # )  # Compute the 30-day rolling average of 'P'
        # P_3d_avg_all.reset_index(drop=True, inplace=True)
        # df_site_and_modis.reset_index(inplace=True)

        df_site_and_modis["GPP_first_guess"] = df_site_and_modis[gpp]
        df_site_and_modis["Reco_first_guess"] = migliavacca_LinGPP(
            T_ref,
            T0,
            boundaries[target_pft]["E0(K)"][0],
            boundaries[target_pft]["K (mm)"][0],
            boundaries[target_pft]["k2"][0],
            boundaries[target_pft]["alpha_p"][0],
            boundaries[target_pft]["alphaLAI"][0],
            max_lai,
            boundaries[target_pft]["RLAI"][0],
            df_site_and_modis["GPP_avrg"],
            df_site_and_modis["P_avrg"],
            T2M + 273.15,
        )

    ###################### optimization for each site year  ####################
    optimized_params_df = pd.DataFrame()

    start_year = df_site_and_modis[timestamp].dt.year.min()
    end_year = df_site_and_modis[timestamp].dt.year.max() + 1

    # for year in range(start_year, start_year + 3):  # TODO: set years here manually
    for year in range(start_year, end_year):
        # Filter data for the current year
        df_year = df_site_and_modis[
            df_site_and_modis[timestamp].dt.year == year
        ].reset_index(drop=True)

        nan_values = df_year.isna()
        nan_sum = nan_values.sum()
        full_year = len(df_year)
        df_year = df_year.dropna()
        df_year.reset_index(drop=True, inplace=True)

        if nan_sum.any():
            print(
                f"WARNING: There are {(nan_sum).max()} NaN values dropped from df_site_and_modis DataFrame"
            )
            percent_nan = nan_sum.max() / full_year * 100
            if percent_nan > 20:
                print(
                    f"WARNING: The year {year} is skipped, as {percent_nan}% values are missing"
                )
                continue

        # Extract relevant columns
        T2M = df_year[t_air].reset_index(drop=True)
        LSWI = df_year["LSWI"].reset_index(drop=True)
        EVI = df_year["250m_16_days_EVI"].reset_index(drop=True)
        PAR = df_year["PAR"].reset_index(drop=True)

        ####################### Set bounds which are valid for all PFTs ################
        if CO2_parametrization == "old":
            bounds_Reco = [
                (0.01, 5),  # Bounds for alpha
                (0.01, 6),  # Bounds for beta
            ]
            bounds_GPP = [
                (Topt_min, Topt_max),  # Bounds for Topt
                (1, 6000),  # Bounds for PAR0
                (0.01, 1),  # Bounds for lambd
            ]
        elif CO2_parametrization == "new":
            bounds_GPP = [
                (Topt_min, Topt_max),  # Bounds for Topt
                (1, 6000),  # Bounds for PAR0
                (0.01, 1),  # Bounds for lambd
            ]
            bounds_Reco = [
                (0.01, 6),  # Bounds for beta
                (-20, 20),  # Bounds for T_crit,
                (0, 1),  # Bounds for T_mult
                (0, 0.8),  # Bounds for alpha1
                (-0.01, 0.01),  # Bounds for alpha2
                (0, 10),  # Bounds for gamma
                (-3, 3),  # Bounds for theta1
                (-1, 1),  # Bounds for theta2
                (-0.1, 0.1),  # Bounds for theta3
            ]

        ################### optimize with 'differential_evolution' ###############
        if CO2_parametrization == "old":
            result = differential_evolution(
                objective_function_VPRM_old_Reco,
                bounds_Reco,
                maxiter=maxiter,  # Number of generations
                disp=True,
            )

            [
                alpha,
                beta,
            ] = result.x

            Reco_optimized_0 = VPRM_old_only_Reco(
                alpha,
                beta,
                T2M,
            )

            # df_year["GPP_calc"] = -(df_year[nee] - Reco_optimized_0)
            # df_year.loc[df_year["GPP_calc"] < 0, "GPP_calc"] = 0
            df_year["GPP_calc"] = df_year[
                gpp
            ]  # TODO make swith here if GPP_calc is needed

            result = differential_evolution(
                objective_function_VPRM_old_GPP,
                bounds_GPP,
                maxiter=maxiter,  # Number of generations
                disp=True,
            )
            optimized_params = result.x

            [Topt, PAR0, lambd] = result.x
            optimized_params = [
                Topt,
                PAR0,
                alpha,
                beta,
                lambd,
            ]

        elif CO2_parametrization == "new":

            result = differential_evolution(
                objective_function_VPRM_new_Reco,
                bounds_Reco,
                maxiter=maxiter,  # Number of generations
                disp=True,
            )

            optimized_params_t = result.x
            [
                beta,
                T_crit,
                T_mult,
                alpha1,
                alpha2,
                gamma,
                theta1,
                theta2,
                theta3,
            ] = optimized_params_t

            Reco_optimized_0 = VPRM_new_only_Reco(
                *optimized_params_t,
                T2M,
                LSWI,
                LSWI_min,
                LSWI_max,
                EVI,
            )
            # df_year["GPP_calc"] = -(df_year[nee] - Reco_optimized_0)
            # df_year.loc[df_year["GPP_calc"] < 0, "GPP_calc"] = 0
            df_year["GPP_calc"] = df_year[
                gpp
            ]  # TODO make switch here if GPP_calc is needed

            result = differential_evolution(
                objective_function_VPRM_new_GPP,
                bounds_GPP,
                maxiter=maxiter,  # Number of generations
                disp=True,
            )
            [Topt, PAR0, lambd] = result.x
            optimized_params = [
                Topt,
                PAR0,
                lambd,
                beta,
                T_crit,
                T_mult,
                alpha1,
                alpha2,
                gamma,
                theta1,
                theta2,
                theta3,
            ]
        ########################## optimize Reco with LinGPP ######################
        elif CO2_parametrization == "migli":
            boundaries = {
                "ENF": {
                    "RLAI": (1.02, 0.42),
                    "alphaLAI": (0.42, 0.08),
                    "k2": (0.478, 0.013),
                    "E0(K)": (124.833, 4.656),
                    "alpha_p": (0.604, 0.065),
                    "K (mm)": (0.222, 0.070),
                },
                "DBF": {
                    "RLAI": (1.27, 0.50),
                    "alphaLAI": (0.34, 0.10),
                    "k2": (0.247, 0.009),
                    "E0(K)": (87.655, 4.405),
                    "alpha_p": (0.796, 0.031),
                    "K (mm)": (0.184, 0.064),
                },
                "GRA": {
                    "RLAI": (0.41, 0.71),
                    "alphaLAI": (1.14, 0.33),
                    "k2": (0.578, 0.062),
                    "E0(K)": (101.181, 6.362),
                    "alpha_p": (0.670, 0.052),
                    "K (mm)": (0.765, 1.589),
                },
                "CRO": {
                    "RLAI": (0.25, 0.66),
                    "alphaLAI": (0.40, 0.11),
                    "k2": (0.244, 0.016),
                    "E0(K)": (129.498, 5.618),
                    "alpha_p": (0.934, 0.065),
                    "K (mm)": (0.035, 3.018),
                },
                "SAV": {
                    "RLAI": (0.42, 0.39),
                    "alphaLAI": (0.57, 0.17),
                    "k2": (0.654, 0.024),
                    "E0(K)": (81.537, 7.030),
                    "alpha_p": (0.474, 0.018),
                    "K (mm)": (0.567, 0.119),
                },
                "SHB": {
                    "RLAI": (0.42, 0.39),
                    "alphaLAI": (0.57, 0.17),
                    "k2": (0.354, 0.021),
                    "E0(K)": (156.746, 8.222),
                    "alpha_p": (0.850, 0.070),
                    "K (mm)": (0.097, 1.304),
                },
                "EBF": {
                    "RLAI": (-0.47, 0.50),
                    "alphaLAI": (0.82, 0.13),
                    "k2": (0.602, 0.044),
                    "E0(K)": (52.753, 4.351),
                    "alpha_p": (0.593, 0.032),
                    "K (mm)": (2.019, 1.052),
                },
                "MF": {
                    "RLAI": (0.78, 0.18),
                    "alphaLAI": (0.44, 0.04),
                    "k2": (0.391, 0.068),
                    "E0(K)": (176.542, 8.222),
                    "alpha_p": (0.703, 0.083),
                    "K (mm)": (2.831, 4.847),
                },
                "WET": {
                    "RLAI": (0.78, 0.18),
                    "alphaLAI": (0.44, 0.04),
                    "k2": (0.398, 0.013),
                    "E0(K)": (144.705, 8.762),
                    "alpha_p": (0.582, 0.163),
                    "K (mm)": (0.054, 0.593),
                },
            }
            # bounds_migli = get_bounds_for_migliavacca(boundaries, target_pft) # used for V11
            global_boundaries = get_global_bounds_for_migliavacca(boundaries)
            result = differential_evolution(
                objective_function_migliavacca_LinGPP,
                bounds=global_boundaries,
                maxiter=maxiter,
                disp=True,
            )
            R_lai0, alpha_lai, k2, E0, alpha_p, k_mm = result.x
            # use Reco from migliavacce to tuning GPP of VPRM
            Reco_optimized = migliavacca_LinGPP(
                T_ref,
                T0,
                E0,
                k_mm,
                k2,
                alpha_p,
                alpha_lai,
                max_lai,
                R_lai0,
                df_year["GPP_avrg"],
                df_year["P_avrg"],
                T2M + 273.5,
            )

            # df_year["GPP_calc"] = -(df_year[nee] - Reco_optimized.tolist())
            # df_year.loc[df_year["GPP_calc"] < 0, "GPP_calc"] = 0
            df_year["GPP_calc"] = df_year[
                gpp
            ]  # TODO make swith here if GPP_calc is needed Used for V13

            bounds_GPP = [
                (Topt_min, Topt_max),  # Bounds for Topt
                (1, 6000),  # Bounds for PAR0
                (0.01, 1),  # Bounds for lambd
            ]

            result = differential_evolution(
                objective_function_VPRM_old_GPP,
                bounds_GPP,
                maxiter=maxiter,  # Number of generations
                disp=True,
            )
            optimized_params = result.x

            [Topt, PAR0, lambd] = result.x
            optimized_params = [
                Topt,
                PAR0,
                alpha,
                beta,
                lambd,
            ]

        ############### Calculate model predictions with optimized parameters ###########
        if CO2_parametrization == "old":
            GPP_optimized, Reco_optimized = VPRM_old(
                *optimized_params,
                Tmin,
                Tmax,
                T2M,
                LSWI,
                LSWI_min,
                LSWI_max,
                EVI,
                PAR,
                VPRM_veg_ID,
                VEGFRA,
            )

        elif CO2_parametrization == "new":
            GPP_optimized, Reco_optimized = VPRM_new(
                *optimized_params,
                Tmin,
                Tmax,
                T2M,
                LSWI,
                LSWI_min,
                LSWI_max,
                EVI,
                PAR,
                VPRM_veg_ID,
                VEGFRA,
            )
        elif CO2_parametrization == "migli":
            GPP_optimized, Reco_optimized_VPRM = VPRM_old(
                *optimized_params,
                Tmin,
                Tmax,
                T2M,
                LSWI,
                LSWI_min,
                LSWI_max,
                EVI,
                PAR,
                VPRM_veg_ID,
                VEGFRA,
            )
            Reco_optimized = migliavacca_LinGPP(
                T_ref,
                T0,
                E0,
                k_mm,
                k2,
                alpha_p,
                alpha_lai,
                max_lai,
                R_lai0,
                df_year["GPP_avrg"],
                df_year["P_avrg"],
                T2M + 273.15,
            )

            Reco_optimized = Reco_optimized.tolist()

        ########################## plot the data ######################
        plot_measured_vs_optimized_VPRM(
            site_name,
            timestamp,
            df_year,
            nee,
            reco_from_nee,
            df_year["GPP_first_guess"],
            GPP_optimized,
            df_year["Reco_first_guess"],
            Reco_optimized,
            base_path,
            folder,
            CO2_parametrization,
            year,
            opt_method,
            maxiter,
            gpp,
        )
        ########################## Calculate error measures ##########################
        # TODO: correlate df_year[gpp] with GPP_optimized if using gpp for optimization.
        mask = (
            ~np.isnan(df_year[nee])
            & ~np.isnan(Reco_optimized)
            & ~np.isnan(GPP_optimized)
        )
        if np.sum(mask) == 0:
            print("Found array with 0 sample(s) while a minimum of 1 is required.")
            R2_NEE = 0
            R2_GPP = 0
            R2_Reco = 0
            rmse_GPP = 0
            rmse_Reco = 0
            rmse_NEE = 0
            NSE_NEE = 0
        else:
            R2_NEE = r2_score(
                df_year[nee][mask],
                np.array(Reco_optimized)[mask] - np.array(GPP_optimized)[mask],
            )
            R2_GPP = r2_score(df_year["GPP_calc"][mask], np.array(GPP_optimized)[mask])
            R2_Reco = r2_score(
                df_year[reco_from_nee][mask], np.array(Reco_optimized)[mask]
            )

            rmse_GPP = np.sqrt(
                mean_squared_error(
                    df_year["GPP_calc"][mask], np.array(GPP_optimized)[mask]
                )
            )
            rmse_Reco = np.sqrt(
                mean_squared_error(
                    df_year[reco_from_nee][mask], np.array(Reco_optimized)[mask]
                )
            )
            rmse_NEE = np.sqrt(
                mean_squared_error(
                    df_year[nee][mask],
                    np.array(Reco_optimized)[mask] - np.array(GPP_optimized)[mask],
                )
            )

            NSE_NEE = calculate_NSE(
                df_year[nee][mask],
                np.array(Reco_optimized)[mask] - np.array(GPP_optimized)[mask],
            )

        ########################## Save results to Excel ##########################
        if CO2_parametrization == "old":
            # Calculate AIC for the old model
            AIC = calculate_AIC(
                17520, rmse_NEE**2, 5
            )  # Assuming 5 parameters for the old model
            data_to_append = pd.DataFrame(
                {
                    "site_ID": [site_name],
                    "PFT": [target_pft],
                    "Year": [year],
                    "Topt": [optimized_params[0]],
                    "PAR0": [optimized_params[1]],
                    "alpha": [optimized_params[2]],
                    "beta": [optimized_params[3]],
                    "lambd": [optimized_params[4]],
                    "R2_GPP": [R2_GPP],
                    "R2_Reco": [R2_Reco],
                    "R2_NEE": [R2_NEE],
                    "RMSE_GPP": [rmse_GPP],
                    "RMSE_Reco": [rmse_Reco],
                    "RMSE_NEE": [rmse_NEE],
                    "AIC": [AIC],
                    "NSE_NEE": [NSE_NEE],
                    "Dropped_NaNs": [(nan_sum).any()],
                    "T_mean": [df_year[t_air].mean()],
                    "T_max": [df_year[t_air].resample("D").max().mean()],
                    "lat": [latitude],
                    "lon": [longitude],
                    "elev": [elevation],
                }
            )
        elif CO2_parametrization == "new":
            # Calculate AIC for the new model
            AIC = calculate_AIC(
                17520, rmse_NEE**2, 12
            )  # Assuming 5 parameters for the new model
            data_to_append = pd.DataFrame(
                {
                    "site_ID": [site_name],
                    "PFT": [target_pft],
                    "Year": [year],
                    "Topt": [optimized_params[0]],
                    "PAR0": [optimized_params[1]],
                    "lambd": [optimized_params[2]],
                    "beta": [optimized_params[3]],
                    "T_crit": [optimized_params[4]],
                    "T_mult": [optimized_params[5]],
                    "alpha1": [optimized_params[6]],
                    "alpha2": [optimized_params[7]],
                    "gamma": [optimized_params[8]],
                    "theta1": [optimized_params[9]],
                    "theta2": [optimized_params[10]],
                    "theta3": [optimized_params[11]],
                    "R2_GPP": [R2_GPP],
                    "R2_Reco": [R2_Reco],
                    "R2_NEE": [R2_NEE],
                    "RMSE_GPP": [rmse_GPP],
                    "RMSE_Reco": [rmse_Reco],
                    "RMSE_NEE": [rmse_NEE],
                    "AIC": [AIC],
                    "NSE_NEE": [NSE_NEE],
                    "T_mean": [df_year[t_air].mean()],
                    "T_max": [df_year[t_air].resample("D").max().mean()],
                    "lat": [latitude],
                    "lon": [longitude],
                    "elev": [elevation],
                }
            )
        elif CO2_parametrization == "migli":
            # Calculate AIC for the new model
            AIC = calculate_AIC(
                17520, rmse_NEE**2, 12
            )  # Assuming 5 parameters for the new model
            data_to_append = pd.DataFrame(
                {
                    "site_ID": [site_name],
                    "PFT": [target_pft],
                    "Year": [year],
                    "Topt": [Topt_min],
                    "PAR0": [optimized_params[1]],
                    "lambd": [optimized_params[4]],
                    "RLAI": [R_lai0],
                    "alphaLAI": [alpha_lai],
                    "k2": [k2],
                    "E0(K)": [E0],
                    "alpha_p": [alpha_p],
                    "K (mm)": [k_mm],
                    "R2_GPP": [R2_GPP],
                    "R2_Reco": [R2_Reco],
                    "R2_NEE": [R2_NEE],
                    "RMSE_GPP": [rmse_GPP],
                    "RMSE_Reco": [rmse_Reco],
                    "RMSE_NEE": [rmse_NEE],
                    "AIC": [AIC],
                    "NSE_NEE": [NSE_NEE],
                    "T_mean": [df_year[t_air].mean()],
                    "T_max": [df_year[t_air].resample("D").max().mean()],
                    "lat": [latitude],
                    "lon": [longitude],
                    "elev": [elevation],
                }
            )

        print(data_to_append)
        optimized_params_df = pd.concat(
            [optimized_params_df, data_to_append], ignore_index=True
        )
        optimized_params_df_all = pd.concat(
            [optimized_params_df_all, data_to_append], ignore_index=True
        )

    optimized_params_df.to_excel(
        base_path
        + folder
        + "/"
        + site_name
        + "_"
        + target_pft
        + "_optimized_params_"
        + CO2_parametrization
        + "_"
        + opt_method
        + "_"
        + str(maxiter)
        + ".xlsx",
        index=False,
    )

    ########################## plot each site year ################################

    plot_site_input(
        df_site_and_modis, timestamp, site_name, folder, base_path, variables
    )

    plot_measured_vs_modeled(
        df_site_and_modis,
        site_name,
        folder,
        base_path,
        CO2_parametrization,
        gpp,
        r_eco,
        nee,
    )

    print(f"{site_name} completed successfully")


if __name__ == "__main__":
    main()
