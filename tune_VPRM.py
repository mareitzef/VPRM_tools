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
    plot_site_year,
    plot_measured_vs_modeled,
)
import argparse
import sys


############################## base settings from parser #############################################
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
            "-v", "--VPRM_old_or_new", type=str, help="VPRM old or new argument"
        )
        args = parser.parse_args()

        base_path = args.base_path
        maxiter = args.maxiter
        opt_method = args.opt_method
        VPRM_old_or_new = args.VPRM_old_or_new
        folder = args.folder
    else:  # to run locally for single cases

        base_path = "/home/madse/Downloads/Fluxnet_Data/"
        maxiter = 1  # (default=100 takes ages)
        opt_method = "diff_evo_V3"  # "diff_evo_V2"
        VPRM_old_or_new = "old"  # "old","new"
        folder = "FLX_IT-PT1_FLUXNET2015_FULLSET_2002-2004_1-4"

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

        residuals_Reco = (np.array(Reco_VPRM) - df_year[nee_mean]) * df_year[night]
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
        residuals_Reco = (np.array(Reco_VPRM) - df_year[nee_mean]) * df_year[night]
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
            latitude = site_info["lat"][i].replace(",", ".")
            longitude = site_info["lon"][i].replace(",", ".")
            elevation = site_info["elev"][i]
        i += 1

    ############################### Read FLUXNET Data ##############################################

    timestamp = "TIMESTAMP_START"
    t_air = "TA_F"
    gpp = "GPP_DT_VUT_REF"
    r_eco = "RECO_DT_VUT_REF"  # is just used for plotting and R2 out of interest
    nee = "NEE_VUT_REF"
    nee_qc = "NEE_VUT_REF_QC"
    night = "NIGHT"
    nee_mean = "NEE_VUT_MEAN"
    # TODO test: nee_cut = 'NEE_CUT_REF' and 'nee = 'NEE_VUT_REF''
    sw_in = "SW_IN_F"
    columns_to_copy = [
        timestamp,
        night,
        t_air,
        gpp,
        r_eco,
        nee,
        sw_in,
        nee_qc,
        nee_mean,  # TODO test the difference for NEE_VUT_MEAN
    ]
    converters = {k: lambda x: float(x) for k in columns_to_copy}
    df_site = pd.read_csv(file_path, usecols=columns_to_copy, converters=converters)
    df_site[timestamp] = pd.to_datetime(df_site[timestamp], format="%Y%m%d%H%M")
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

    ##################################### read  MODIS data ##################################

    mod_files = glob(os.path.join(modis_path, "*_MOD*.xlsx"))
    myd_files = glob(os.path.join(modis_path, "*_MYD*.xlsx"))

    # Process MOD and MYD files
    dfs_modis = []
    for file_list in [mod_files, myd_files]:
        for file_path in file_list:
            df = pd.read_excel(file_path)[["calendar_date", "value"]].assign(
                value=lambda x: x["value"] * 0.0001
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

    nan_values = df_site_and_modis.isna()
    nan_sum = nan_values.sum()
    df_site_and_modis = df_site_and_modis.dropna()
    df_site_and_modis.reset_index(drop=True, inplace=True)

    if nan_sum.any():
        print("WARNING: There are NaN values dropped from df_site_and_modis DataFrame")
        print(nan_sum)

    ############################# prepare input variables  #############################
    # just use fluxnet qualities 0 and 1 - new in V3
    df_site_and_modis.loc[df_site_and_modis[nee_qc] > 1, nee] = np.nan
    # create extra column for daytime NEE
    # df_site_and_modis[nee_mean] = df_site_and_modis[nee].copy() # TODO: find the bug here, I want to use NEE_VUT_REF here
    # only the respiration of nee_mean is used
    df_site_and_modis.loc[df_site_and_modis[nee_mean] < 0, nee_mean] = np.nan
    # calculate LSWI from MODIS Bands 2 and 6
    df_site_and_modis["LSWI"] = (
        df_site_and_modis["sur_refl_b02"] - df_site_and_modis["sur_refl_b06"]
    ) / (df_site_and_modis["sur_refl_b02"] + df_site_and_modis["sur_refl_b06"])
    T2M = df_site_and_modis[t_air]
    EVI = df_site_and_modis["250m_16_days_EVI"]
    PAR = df_site_and_modis["PAR"]
    LSWI = df_site_and_modis["LSWI"]
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
    )  # TODO: site-specific minimum LSWI across a full year  (from a multi-year mean), can it be below zero?

    variables = [
        nee,
        nee_mean,
        gpp,
        r_eco,
        t_air,
        "PAR",
        "LSWI",
        "250m_16_days_EVI",
    ]

    # Parameters are set constant for physical reason of no PSN above and below (true for the Alps)
    Tmin = 0
    Tmax = 45

    #############################  first guess  of parameters #########################
    Topt = 20.0
    # adopted from VPRM_table_Europe with values for Wetland from Gourdji 2022
    if VPRM_old_or_new == "old":
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
        }
    ###################### table from Gourdji 2022 for VPRM_new ############################
    elif VPRM_old_or_new == "new":
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
        }
    else:
        print("ERROR - you have to choose VPRM_old_or_new")

    ###################### select first guess according to PFT ##########################
    df_VPRM_table_first_guess = pd.DataFrame(VPRM_table_first_guess)
    parameters = df_VPRM_table_first_guess[
        df_VPRM_table_first_guess["PFT"] == target_pft
    ].iloc[0]
    if VPRM_old_or_new == "old":
        PAR0 = parameters["PAR0"]
        alpha = parameters["alpha"]
        beta = parameters["beta"]
        lambd = -parameters["lambda"]
        VPRM_veg_ID = parameters["VPRM_veg_ID"]
    if VPRM_old_or_new == "new":
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

    ###################### run VPRM with first Guess for comparison plot #################
    if VPRM_old_or_new == "old":
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
    elif VPRM_old_or_new == "new":
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

    df_site_and_modis["GPP_VPRM_first_guess"] = GPP_VPRM
    df_site_and_modis["Reco_VPRM_first_guess"] = Reco_VPRM

    ###################### optimization for each site year  ####################
    optimized_params_df = pd.DataFrame()

    start_year = df_site_and_modis[timestamp].dt.year.min()
    end_year = df_site_and_modis[timestamp].dt.year.max() + 1

    for year in range(start_year, end_year):
        # Filter data for the current year
        df_year = df_site_and_modis[
            df_site_and_modis[timestamp].dt.year == year
        ].reset_index(drop=True)

        # Extract relevant columns
        T2M = df_year[t_air].reset_index(drop=True)
        LSWI = df_year["LSWI"].reset_index(drop=True)
        EVI = df_year["250m_16_days_EVI"].reset_index(drop=True)
        PAR = df_year["PAR"].reset_index(drop=True)

        ####################### Set bounds which are valid for all PFTs ################
        if VPRM_old_or_new == "old":
            bounds_Reco = [
                (0.01, 5),  # Bounds for alpha
                (0.01, 6),  # Bounds for beta
            ]
            bounds_GPP = [
                (0, 50),  # Bounds for Topt
                (1, 6000),  # Bounds for PAR0
                (0.01, 1),  # Bounds for lambd
            ]
        elif VPRM_old_or_new == "new":
            bounds_GPP = [
                (0, 50),  # Bounds for Topt
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
        if VPRM_old_or_new == "old":
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

            Reco_VPRM_optimized_0 = VPRM_old_only_Reco(
                alpha,
                beta,
                T2M,
            )

            df_year["GPP_calc"] = -(df_year[nee] - Reco_VPRM_optimized_0)
            df_year.loc[df_year["GPP_calc"] < 0, "GPP_calc"] = 0

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

        elif VPRM_old_or_new == "new":

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

            Reco_VPRM_optimized_0 = VPRM_new_only_Reco(
                *optimized_params_t,
                T2M,
                LSWI,
                LSWI_min,
                LSWI_max,
                EVI,
            )
            df_year["GPP_calc"] = -(df_year[nee] - Reco_VPRM_optimized_0)
            df_year.loc[df_year["GPP_calc"] < 0, "GPP_calc"] = 0

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

        ############### Calculate model predictions with optimized parameters ###########
        if VPRM_old_or_new == "old":
            GPP_VPRM_optimized, Reco_VPRM_optimized = VPRM_old(
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

        elif VPRM_old_or_new == "new":
            GPP_VPRM_optimized, Reco_VPRM_optimized = VPRM_new(
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
        ########################## TODO: optimize Reco with LinGPP ######################

        ########################## plot the data ######################
        plot_measured_vs_optimized_VPRM(
            site_name,
            timestamp,
            df_year,
            nee,
            nee_mean,
            df_year["GPP_VPRM_first_guess"],
            GPP_VPRM_optimized,
            df_year["Reco_VPRM_first_guess"],
            Reco_VPRM_optimized,
            base_path,
            folder,
            VPRM_old_or_new,
            year,
            opt_method,
            maxiter,
        )
        ########################## Calculate error measures ##########################

        mask = (
            ~np.isnan(df_year[nee])
            & ~np.isnan(Reco_VPRM_optimized)
            & ~np.isnan(GPP_VPRM_optimized)
        )
        R2_NEE = r2_score(
            df_year[nee][mask],
            np.array(Reco_VPRM_optimized)[mask] - np.array(GPP_VPRM_optimized)[mask],
        )

        R2_GPP = r2_score(df_year[gpp][mask], np.array(GPP_VPRM_optimized)[mask])
        R2_Reco = r2_score(df_year[r_eco][mask], np.array(Reco_VPRM_optimized)[mask])

        rmse_GPP = np.sqrt(
            mean_squared_error(df_year[gpp][mask], np.array(GPP_VPRM_optimized)[mask])
        )
        rmse_Reco = np.sqrt(
            mean_squared_error(
                df_year[r_eco][mask], np.array(Reco_VPRM_optimized)[mask]
            )
        )
        rmse_NEE = np.sqrt(
            mean_squared_error(
                df_year[nee][mask],
                np.array(Reco_VPRM_optimized)[mask]
                - np.array(GPP_VPRM_optimized)[mask],
            )
        )
        NSE_GPP = calculate_NSE(df_year[gpp][mask], np.array(GPP_VPRM_optimized)[mask])
        NSE_Reco = calculate_NSE(
            df_year[r_eco][mask], np.array(Reco_VPRM_optimized)[mask]
        )
        NSE_NEE = calculate_NSE(
            df_year[nee][mask],
            np.array(Reco_VPRM_optimized)[mask] - np.array(GPP_VPRM_optimized)[mask],
        )

        ########################## Save results to Excel ##########################
        if VPRM_old_or_new == "old":
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
                    "NSE_GPP": [NSE_GPP],
                    "NSE_Reco": [NSE_Reco],
                    "NSE_NEE": [NSE_NEE],
                    "T_mean": [df_year[t_air].mean()],
                    "lat": [latitude],
                    "lon": [longitude],
                    "elev": [elevation],
                }
            )
        elif VPRM_old_or_new == "new":
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
                    "NSE_GPP": [NSE_GPP],
                    "NSE_Reco": [NSE_Reco],
                    "NSE_NEE": [NSE_NEE],
                    "T_mean": [df_year[t_air].mean()],
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
        + VPRM_old_or_new
        + "_"
        + opt_method
        + "_"
        + str(maxiter)
        + ".xlsx",
        index=False,
    )

    ########################## plot each site year ################################

    plot_site_year(
        df_site_and_modis, timestamp, site_name, folder, base_path, variables
    )

    plot_measured_vs_modeled(
        df_site_and_modis,
        site_name,
        folder,
        base_path,
        VPRM_old_or_new,
        gpp,
        r_eco,
        nee,
    )


if __name__ == "__main__":
    main()
