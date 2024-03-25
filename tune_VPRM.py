import pandas as pd
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from VPRM_for_timeseries import VPRM_for_timeseries
from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_squared_error
from deap import base, creator, tools, algorithms
import random
############################## base settings #############################################

base_path = '/home/madse/Downloads/Fluxnet_Data/'
site_info = pd.read_csv("/home/madse/Downloads/Fluxnet_Data/site_info_Alps_lat44-50_lon5-17.csv")
maxiter = 100 # TODO: maxiter should be 100-1000
opt_method = 'deap'

# TODO: ll flux tower NEE data was u-star filtered using site-specific thresholds determined 
# visually by plotting averaged nighttime NEE along binned u-star intervals (Barr et al., 2013).
###########################################################################################


folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
flx_folders = [folder for folder in folders if folder.startswith("FLX_")]

optimized_params_df_all = pd.DataFrame()

for folder in flx_folders:
    print(folder)

    site_name = '_'.join(folder.split("_")[1:2])
    file_base = '_'.join(folder.split("_")[0:4])
    years = '_'.join(folder.split("_")[4:6])
    file_path = base_path+folder+'/'+file_base+'_HH_'+years+'.csv'

    # get PFT of site
    i = 0 
    for site_i in site_info['site']: 
        if site_name == site_i:
            target_pft = site_info['pft'][i]
            latitude = site_info['lat'][i]
            longitude = site_info['lon'][i]
            elevation = site_info['elev'][i]
        i += 1
    
    # to run part of  the script only on undone sites
    # if glob(os.path.join(base_path+folder, '*optimized_params.xlsx')): 
    #     continue

    if file_path.endswith('.xlsx'): # exception for FAIR site
        timestamp = 'Date Time'
        t_air = 'Tair'
        gpp = 'GPP_DT'
        r_eco = 'Reco' 
        sw_in = 'Rg'
        columns_to_copy = [timestamp,t_air ,gpp, r_eco , sw_in]
        df_site = pd.read_excel(file_path, skiprows=[1], usecols=columns_to_copy)
        df_site[timestamp] = pd.to_datetime(df_site[timestamp], format='%Y%m%d%H%M')
        df_site.set_index(timestamp, inplace=True)
        # Define the modis_path for MODIS files
        modis_path = "/home/madse/Dropbox/PhD/WRF_2024/Tools"
    elif file_path.endswith('.csv'):
        timestamp = 'TIMESTAMP_START'
        t_air = 'TA_F'
        gpp = 'GPP_DT_VUT_REF'
        r_eco = 'RECO_DT_VUT_REF'
        nee = 'NEE_VUT_REF'
        # TODO test: nee_cut = 'NEE_CUT_REF' and 'nee = 'NEE_VUT_REF''
        sw_in = 'SW_IN_F'
        columns_to_copy = [timestamp, t_air ,gpp, r_eco, nee, sw_in]
        df_site = pd.read_csv(file_path, usecols=columns_to_copy)
        df_site[timestamp] = pd.to_datetime(df_site[timestamp], format='%Y%m%d%H%M')
        df_site.set_index(timestamp, inplace=True)
        # TODO: where is the ERROR?? needs /3 at Neustift
        # df_site[gpp]=df_site[gpp]/3 
        # df_site[r_eco]=df_site[r_eco]/3

        # Define the modis_path for MODIS files
        modis_path = "/home/madse/Downloads/Fluxnet_Data/"+folder+"/"
    else:
        raise ValueError("Unsupported file format. Only Excel (.xlsx) and CSV (.csv) files are supported.")


    df_site.loc[df_site[t_air] < -40, t_air] = np.nan
    df_site.loc[df_site[sw_in] < 0,sw_in] = np.nan
    df_site = df_site.rolling(window=3, min_periods=1).mean()
    df_site_hour = df_site.resample('H').mean()

    df_site_hour['PAR'] = df_site_hour[sw_in]*0.505 # assuming that PAR = 50.5% of globar radiation (Mahadevan 2008)
    df_site_hour.drop(columns=[sw_in], inplace=True)


    mod_files = glob(os.path.join(modis_path, "*_MOD*.xlsx"))
    myd_files = glob(os.path.join(modis_path, "*_MYD*.xlsx"))

    # Process MOD and MYD files
    dfs_modis = []
    for file_list in [mod_files, myd_files]:
        for file_path in file_list:
            df = pd.read_excel(file_path)[['calendar_date', 'value']].assign(value=lambda x: x['value'] * 0.0001) # sclaing factor from user guide: https://lpdaac.usgs.gov/documents/103/MOD13_User_Guide_V6.pdf
            file_parts = file_path.split('/')[-1]
            file_parts2 = file_parts.split('_')[2:6]
            if file_parts2[0] == '250m':
                file_type = '_'.join(file_parts2)
            else: 
                file_type = '_'.join(file_parts2[:-1])
            df.rename(columns={'value': file_type}, inplace=True)
            dfs_modis.append(df)

    # Merge dataframes based on 'calendar_date'
    df_modis = dfs_modis[0]
    for df in dfs_modis[1:]:
        df_modis = pd.merge(df_modis, df, on='calendar_date', how='outer', suffixes=('_x', '_y'))

    for column in df_modis.columns:
        if column.endswith('_x'):
            base_column = column[:-2]  # Remove suffix '_x'
            if base_column + '_y' in df_modis.columns:
                df_modis[column].fillna(df_modis[base_column + '_y'], inplace=True)
                df_modis.drop(columns=[base_column + '_y'], inplace=True)
                df_modis.rename(columns={column: base_column}, inplace=True)

    df_modis.sort_values(by='calendar_date', inplace=True)
    df_modis.reset_index(drop=True, inplace=True)
    df_modis['calendar_date'] = pd.to_datetime(df_modis['calendar_date'])
    df_modis.set_index('calendar_date', inplace=True)
    # Interpolate the DataFrame linearly to fill missing values and resample to hourly frequency
    df_modis.loc[df_modis['sur_refl_b02'] < 0, 'sur_refl_b02'] = np.nan
    df_modis.loc[df_modis['sur_refl_b06'] < 0, 'sur_refl_b06'] = np.nan
    df_modis['250m_16_days_EVI'] = df_modis['250m_16_days_EVI']
    df_modis_hour = df_modis.resample('H').interpolate(method='linear')
    df_modis_hour.reset_index(inplace=True)
    df_modis_hour.rename(columns={'calendar_date': timestamp}, inplace=True)
    df_modis_hour.set_index(timestamp, inplace=True)

    # Perform inner join to keep only matching dates
    df_site_and_modis = pd.merge(df_site_hour, df_modis_hour, left_index=True, right_index=True, how='inner')
    df_site_and_modis.reset_index(inplace=True)

    nan_values = df_site_and_modis.isna()
    nan_summary = nan_values.sum()

    if nan_summary.any():
        print("WARNING: There are NaN values in the DataFrame")

    df_site_and_modis['LSWI']=(df_site_and_modis['sur_refl_b02']- df_site_and_modis['sur_refl_b06'])/(df_site_and_modis['sur_refl_b02']+df_site_and_modis['sur_refl_b06'])


    T2M = df_site_and_modis[t_air]
    EVI = df_site_and_modis['250m_16_days_EVI'] 
    PAR =  df_site_and_modis['PAR']
    LSWI =  df_site_and_modis['LSWI']
    LSWI_max = df_site_and_modis['LSWI'][2100:4200].max() # TODO: LSWImax is the site-specific multiyear maximum daily LSWI from May to October
    LSWI_min = df_site_and_modis['LSWI'].min() #site-specific minimum LSWI across a full year (from a multi-year mean) 

    # Parameters
    Tmin = 0
    Tmax = 40
    Topt = 20.0

    # VPRM_table_Europe = {
    #     'PFT': ['Evergreen', 'Deciduous', 'Mixed Forest', 'Shrubland', 'Savanna', 'Cropland', 'Grassland'],
    #     'VEGTYP': [1, 2, 3, 4, 5, 6, 7],  
    #     'PAR0': [270.2, 271.4, 236.6, 363.0, 682.0, 690.3, 229.1],
    #     'lambda': [-0.3084, -0.1955, -0.2856, -0.0874, -0.1141, -0.1350, -0.1748],
    #     'alpha': [0.1797, 0.1495, 0.2258, 0.0239, 0.0049, 0.1699, 0.0881],
    #     'beta': [0.8800, 0.8233, 0.4321, 0.0000, 0.0000, -0.0144, 0.5843]
    # }

    VPRM_table_Europe_wet = {
        'PFT': ['ENF', 'DBF', 'MF', 'SHB', 'WET', 'CRO', 'GRA'],
        'VEGTYP': [1, 2, 3, 4, 5, 6, 7],  
        'PAR0': [270.2, 271.4, 236.6, 363.0, 579, 690.3, 229.1],
        'lambda': [-0.3084, -0.1955, -0.2856, -0.0874, -0.0752, -0.1350, -0.1748],
        'alpha': [0.1797, 0.1495, 0.2258, 0.0239, 0.111, 0.1699, 0.0881],
        'beta': [0.8800, 0.8233, 0.4321, 0.0000, 0.82, -0.0144, 0.5843]
    }

    # TODO:
    # VPRM_new_table_gourji = {
    #     'PFT_long': ['Deciduous Broadleaf Forest & Urban', 'Evergreen / Mixed Forest >40°N', 'Evergreen / Mixed Forest <40°N', 'Shrub/Savannah', 'Grass/Pasture/Dev-open', 'Wetlands', 'Crops other', 'Crops corn'],
    #     'PFT': ['DBF', 'EMFlt40', 'EMFgt40', 'SHB', 'GRA', 'WET', 'CRO', 'CRC'],
    #     'Tmin': [0, 0, 0, 0, 0, 0, 0, 0],
    #     'Tmax': [45, 45, 45, 45, 45, 45, 45, 45],
    #     'Topt': [23, 18, 20, 7, 20, 29, 26, 35],
    #     'Tcrit': [-15, 1, 0, 5, 1, 6, 7, -1],
    #     'Tmult': [0.55, 0.05, 0, 0.41, 0.14, 0.14, 0.05, 0],
    #     'lambda': [-0.1023, -0.1097, -0.0920, -0.0996, -0.1273, -0.1227, -0.0732, -0.0997],
    #     'PAR0': [539, 506, 896, 811, 673, 456, 1019, 1829],
    #     'beta': [0.12, 0.47, 0.28, 1.53, -6.18, -0.82, -1.20, -0.02],
    #     'alpha1': [0.065, 0.088, 0.025, 0.004, 0.853, 0.261, 0.234, 0.083],
    #     'alpha2': [0.0024, 0.0047, 0.0058, 0.0049, -0.0250, -0.0051, -0.0060, -0.0018],
    #     'gamma': [4.61, 1.39, 4.18, 0.09, 5.19, 3.46, 3.85, 4.89],
    #     'theta1': [0.116, -0.530, -0.729, -1.787, 1.749, -0.7, 0.032, 0.150],
    #     'theta2': [-0.0005, 0.2063, 0.1961, 0.4537, -0.2829, 0.0990, -0.0429, -0.1324],
    #     'theta3': [0.0009, -0.0054, -0.0055, -0.0138, 0.0166, 0.0018, 0.0090, 0.0156]
    # }

    df_VPRM_table_Europe = pd.DataFrame(VPRM_table_Europe_wet)
    parameters = df_VPRM_table_Europe[df_VPRM_table_Europe['PFT'] == target_pft].iloc[0]
    PAR0 = parameters['PAR0']
    alpha = parameters['alpha']
    beta = parameters['beta']
    lambd = -parameters['lambda']
    VEGTYP = parameters['VEGTYP']
    VEGFRA = 1


    GPP_VPRM, Reco_VPRM = VPRM_for_timeseries(Tmin,Tmax,Topt,PAR0,alpha,beta,lambd,T2M,LSWI,LSWI_max,EVI,PAR,VEGTYP,VEGFRA)

    df_site_and_modis['GPP_VPRM'] = GPP_VPRM
    df_site_and_modis['Reco_VPRM'] = Reco_VPRM


    optimized_params_df = pd.DataFrame(columns=['Year', 'Tmin_opt', 'Tmax_opt', 'Topt_opt', 'PAR0_opt', 'alpha_opt', 'beta_opt', 'lambd_opt'])

    start_year = df_site_and_modis[timestamp].dt.year.min()
    end_year = df_site_and_modis[timestamp].dt.year.max()+1

    for year in range(start_year,end_year):  
        # Filter data for the current year
        df_year = df_site_and_modis[df_site_and_modis[timestamp].dt.year == year].reset_index(drop=True)
        
        # Extract relevant columns
        T2M = df_year[t_air].reset_index(drop=True)
        LSWI = df_year['LSWI'].reset_index(drop=True)
        LSWI_max = df_year['LSWI'][2100:4200].max()  # Assuming LSWI_max is the max LSWI value for the year
        EVI = df_year['250m_16_days_EVI'].reset_index(drop=True)
        PAR = df_year['PAR'].reset_index(drop=True)
        
        # Define the objective function
        def objective_function_VPRM(x):
            Tmin, Tmax, Topt, PAR0, alpha, beta, lambd = x
            GPP_VPRM, Reco_VPRM = VPRM_for_timeseries(Tmin, Tmax, Topt, PAR0, alpha, beta, lambd, T2M, LSWI, LSWI_max, EVI, PAR, VEGTYP, VEGFRA)
            residuals_NEE = (np.array(Reco_VPRM) - np.array(GPP_VPRM)) - df_year[nee] 
            return np.sum(residuals_NEE**2)

        # Set initial guess for parameters
        initial_guess = [Tmin, Tmax, Topt, PAR0, alpha, beta, lambd]

        # Set bounds which are valid for all PFT
        bounds = [(-10, 10),  # Bounds for Tmin 
                (30,50),  # Bounds for Tmax
                (0,50),  # Bounds for Topt
                (0,6000),  # Bounds for PAR0
                (0,5),  # Bounds for alpha
                (-3,6),  # Bounds for beta
                (0,1)]  # Bounds for lambd
        
        if opt_method == 'minimize':
            # Run optimization
            options = {'maxiter': maxiter, 'disp': False} 
            result = minimize(objective_function_VPRM, initial_guess, bounds=bounds, options=options)
            optimized_params = result.x
            # Calculate model predictions with optimized parameters
            GPP_VPRM_optimized, Reco_VPRM_optimized = VPRM_for_timeseries(*optimized_params, T2M, LSWI, LSWI_max, EVI, PAR, VEGTYP, VEGFRA)
            # Calculate R²
            R2_GPP = r2_score(df_year[gpp], np.array(GPP_VPRM_optimized))
            R2_Reco = r2_score(df_year[r_eco], np.array(Reco_VPRM_optimized))
            R2_NEE = r2_score(df_year[nee], np.array(Reco_VPRM_optimized) - np.array(GPP_VPRM_optimized))
            # Calculate other measures
            # root mean squared error
            rmse_GPP = np.sqrt(mean_squared_error(df_year[gpp], np.array(GPP_VPRM_optimized)))
            rmse_Reco = np.sqrt(mean_squared_error(df_year[r_eco], np.array(Reco_VPRM_optimized)))
            rmse_NEE =np.sqrt( mean_squared_error(df_year[nee], np.array(Reco_VPRM_optimized) - np.array(GPP_VPRM_optimized)))
            percent_of_sum = sum(np.array(Reco_VPRM_optimized) - np.array(GPP_VPRM_optimized))/sum(df_year[nee])
        
        elif opt_method == 'deap':
            # Define the evaluation function
            def objective_function_VPRM(individual):
                Tmin, Tmax, Topt, PAR0, alpha, beta, lambd = individual
                GPP_VPRM, Reco_VPRM = VPRM_for_timeseries(Tmin, Tmax, Topt, PAR0, alpha, beta, lambd, T2M, LSWI, LSWI_max, EVI, PAR, VEGTYP, VEGFRA)
                residuals_NEE = (np.array(Reco_VPRM) - np.array(GPP_VPRM)) - df_year[nee] 
                return (np.sum(residuals_NEE**2),)  # Return as tuple

            # Create the DEAP types
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)
            # Initialize the DEAP toolbox
            toolbox = base.Toolbox()
            # Register the evaluation function
            toolbox.register("evaluate", objective_function_VPRM)
            # Define the range for each parameter
            toolbox.register("attr_float", random.uniform, -10, 10)  # Range for Tmin
            toolbox.register("attr_float", random.uniform, 30, 50)   # Range for Tmax
            toolbox.register("attr_float", random.uniform, 0, 50)    # Range for Topt
            toolbox.register("attr_float", random.uniform, 0, 6000)  # Range for PAR0
            toolbox.register("attr_float", random.uniform, 0, 5)     # Range for alpha
            toolbox.register("attr_float", random.uniform, -3, 6)    # Range for beta
            toolbox.register("attr_float", random.uniform, 0, 1)      # Range for lambd
            # Define the individual (parameter set) as a list of floats
            toolbox.register("individual", tools.initCycle, creator.Individual, 
                            (toolbox.attr_float,)*7, n=1)
            # Define the population as a list of individuals
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            # Define the mating, mutation, and selection operators
            toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Blend crossover
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)  # Gaussian mutation
            toolbox.register("select", tools.selTournament, tournsize=3)
            # Set the algorithm parameters
            NGEN = 100  # Number of generations, should be 100
            MU = 100     # Population size
            CXPB = 0.7  # Crossover probability
            MUTPB = 0.2 # Mutation probability
            # Create an initial population
            pop = toolbox.population(n=MU)
            # Evaluate the entire population
            fitnesses = list(map(toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
            # Begin the evolution
            for gen in range(NGEN):
                print(gen)
                # Select the next generation individuals
                offspring = algorithms.varAnd(pop, toolbox, cxpb=CXPB, mutpb=MUTPB)              
                # Evaluate the offspring
                fits = toolbox.map(toolbox.evaluate, offspring)
                for ind, fit in zip(offspring, fits):
                    ind.fitness.values = fit
                # Select the next generation
                pop = toolbox.select(offspring + pop, k=MU)

            # Retrieve the best individual
            optimized_params = tools.selBest(pop, k=1)[0]
            
            GPP_VPRM_optimized, Reco_VPRM_optimized = VPRM_for_timeseries(*optimized_params, T2M, LSWI, LSWI_max, EVI, PAR, VEGTYP, VEGFRA)
            # Calculate R²
            R2_GPP = r2_score(df_year[gpp], np.array(GPP_VPRM_optimized))
            R2_Reco = r2_score(df_year[r_eco], np.array(Reco_VPRM_optimized))
            R2_NEE = r2_score(df_year[nee], np.array(Reco_VPRM_optimized) - np.array(GPP_VPRM_optimized))
            # Calculate other measures
            # root mean squared error
            rmse_GPP = np.sqrt(mean_squared_error(df_year[gpp], np.array(GPP_VPRM_optimized)))
            rmse_Reco = np.sqrt(mean_squared_error(df_year[r_eco], np.array(Reco_VPRM_optimized)))
            rmse_NEE =np.sqrt( mean_squared_error(df_year[nee], np.array(Reco_VPRM_optimized) - np.array(GPP_VPRM_optimized)))
            percent_of_sum = sum(np.array(Reco_VPRM_optimized) - np.array(GPP_VPRM_optimized))/sum(df_year[nee])

            print(optimized_params)



        data_to_append = pd.DataFrame({'site_ID': [site_name],
                                'PFT': [target_pft],
                                'method': [opt_method],
                                'Year': [year],
                                'Tmin_opt': [optimized_params[0]],
                                'Tmax_opt': [optimized_params[1]],
                                'Topt_opt': [optimized_params[2]],
                                'PAR0_opt': [optimized_params[3]],
                                'alpha_opt': [optimized_params[4]],
                                'beta_opt': [optimized_params[5]],
                                'lambd_opt': [optimized_params[6]],
                                'R2_GPP': [R2_GPP],
                                'R2_Reco': [R2_Reco],
                                'R2_NEE': [R2_NEE],
                                'RMSE_GPP': [rmse_GPP],
                                'RMSE_Reco': [rmse_Reco],
                                'RMSE_NEE': [rmse_NEE],
                                'percent_NEE_sum': [percent_of_sum],
                                'T_mean':[df_year[t_air].mean()],
                                'lat': [latitude],
                                'lon': [longitude],
                                'elev': [elevation]})
        optimized_params_df = pd.concat([optimized_params_df, data_to_append], ignore_index=True)
        optimized_params_df_all = pd.concat([optimized_params_df_all, data_to_append],ignore_index=True)



    optimized_params_df.to_excel(base_path+folder+'/'+site_name+'_'+target_pft+'_optimized_params_maxiter'+str(opt_method)+'.xlsx', index=False)

    # # TODO: plot the difference
    # params_difference = original_params - result.x
    variables = [t_air, nee, gpp, r_eco, 'PAR', 'LSWI','250m_16_days_EVI',]

    df_site_and_modis.set_index(timestamp, inplace=True)

    fig, axes = plt.subplots(nrows=len(variables), ncols=1, figsize=(10, 6*len(variables)))
    for i, var in enumerate(variables):
        axes[i].plot(df_site_and_modis.index, df_site_and_modis[var], label=var,linestyle='', marker='o', markersize=1)
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel(var)
        axes[i].legend()
        axes[i].grid(True)
    plt.tight_layout()
    plt.savefig(base_path+folder+'/check_input.png')

    # Plot measured vs. standard model for all years
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    axes[0].plot(df_site_and_modis.index, df_site_and_modis[gpp], label='Measured GPP', color='blue', linestyle='', marker='o', markersize=1)
    axes[0].plot(df_site_and_modis.index, df_site_and_modis['GPP_VPRM'], label='Modeled GPP', color='green', linestyle='', marker='o', markersize=1)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('GPP')
    axes[0].set_title('Comparison of Measured and Modeled GPP')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(df_site_and_modis.index, df_site_and_modis[r_eco], label='Measured Reco', color='blue', linestyle='', marker='o', markersize=1)
    axes[1].plot(df_site_and_modis.index, df_site_and_modis['Reco_VPRM'], label='Modeled Reco', color='green', linestyle='', marker='o', markersize=1)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel(r_eco)
    axes[1].set_title('Comparison of Measured and Modeled Reco')
    axes[1].legend()
    axes[1].grid(True)
    axes[2].plot(df_site_and_modis.index, df_site_and_modis[nee], label='Measured NEE', color='blue', linestyle='', marker='o', markersize=1)
    axes[2].plot(df_site_and_modis.index, df_site_and_modis['Reco_VPRM'] - df_site_and_modis['GPP_VPRM'], label='Modeled NEE', color='green', linestyle='', marker='o', markersize=1)
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel(nee)
    axes[2].set_title('Comparison of Measured and Modeled NEE')
    axes[2].legend()
    axes[2].grid(True)
    plt.tight_layout()
    plt.savefig(base_path+folder+'/compare_fluxes.png')

    # Plot measured vs. standard model for each year with optimized parameters
    i = 0
    for year in range(start_year,end_year):  # Assuming data spans from 2002 to 2012
        # Filter data for the current year
        df_year = df_site_and_modis[df_site_and_modis.index.year == year]
        
        # Extract relevant columns
        T2M = df_year[t_air].reset_index(drop=True)
        LSWI = df_year['LSWI'].reset_index(drop=True)
        LSWI_max = df_year['LSWI'][2100:4200].max()  # Assuming LSWI_max is the max LSWI value for the year
        EVI = df_year['250m_16_days_EVI'].reset_index(drop=True)
        PAR = df_year['PAR'].reset_index(drop=True)
        GPP_VPRM_opt, Reco_VPRM_opt = VPRM_for_timeseries(
            optimized_params_df['Tmin_opt'][i], 
            optimized_params_df['Tmax_opt'][i], 
            optimized_params_df['Topt_opt'][i], 
            optimized_params_df['PAR0_opt'][i], 
            optimized_params_df['alpha_opt'][i], 
            optimized_params_df['beta_opt'][i],
            optimized_params_df['lambd_opt'][i], 
            T2M, 
            LSWI, 
            LSWI_max, 
            EVI, 
            PAR, 
            VEGTYP, 
            VEGFRA)
        i = i+1
        # Update DataFrame with optimized results
        df_year.loc[:,'GPP_VPRM_opt'] = np.array(GPP_VPRM_opt)
        df_year.loc[:,'Reco_VPRM_opt'] = np.array(Reco_VPRM_opt)

        # Create a figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 18))

        # Plot comparison of GPP
        axes[0].plot(df_year.index, df_year['GPP_VPRM'], linestyle='', marker='o', markersize=1, label='Modeled GPP', color='green')
        axes[0].plot(df_year.index, df_year['GPP_VPRM_opt'], linestyle='', marker='o', markersize=1, label='Modeled GPP optimized', color='red')
        axes[0].plot(df_year.index, df_year[gpp], linestyle='', marker='o', markersize=1, label='Measured GPP', color='blue')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('GPP')
        axes[0].set_title('Comparison of Measured and Modeled GPP')
        axes[0].legend()
        axes[0].grid(True)
        axes[1].plot(df_year.index, df_year['Reco_VPRM'], linestyle='', marker='o', markersize=1, label='Modeled Reco', color='green')
        axes[1].plot(df_year.index, df_year['Reco_VPRM_opt'], linestyle='', marker='o', markersize=1, label='Modeled Reco optimized', color='red')
        axes[1].plot(df_year.index, df_year[r_eco], linestyle='', marker='o', markersize=1, label='Measured Reco', color='blue')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Reco')
        axes[1].set_title('Comparison of Measured and Modeled Reco')
        axes[1].legend()
        axes[1].grid(True)
        axes[2].plot(df_year.index, df_year['Reco_VPRM'] - df_year['GPP_VPRM'], linestyle='', marker='o', markersize=1, label='Modeled NEE', color='green')
        axes[2].plot(df_year.index, df_year['Reco_VPRM_opt'] - df_year['GPP_VPRM_opt'], linestyle='', marker='o', markersize=1, label='Modeled NEE optimized', color='red')
        axes[2].plot(df_year.index, df_year[nee], linestyle='', marker='o', markersize=1, label='Measured NEE', color='blue')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('NEE')
        axes[2].set_title('Comparison of Measured and Modeled NEE')
        axes[2].legend()
        axes[2].grid(True)
        plt.tight_layout()
        plt.savefig(base_path + folder + '/compare_optimized_fluxes'+str(year)+'_'+str(opt_method)+'.png')

        del df_year
        #plt.show()

optimized_params_df_all.to_excel(base_path+'all_optimized_params_maxiter'+str(maxiter)+'.xlsx', index=False)
