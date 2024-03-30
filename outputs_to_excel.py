def write_all_to_excel(
    optimized_params_df_all, base_path, VPRM_old_or_new, opt_method, maxiter
):

    optimized_params_df_all.to_excel(
        base_path
        + "all_optimized_params_VPRM_"
        + VPRM_old_or_new
        + "_"
        + opt_method
        + "_"
        + str(maxiter)
        + ".xlsx",
        index=False,
    )


def write_filtered_params_to_excel(
    optimized_params_df_all, base_path, VPRM_old_or_new, opt_method, maxiter
):

    parameters_to_plot = [
        "Topt",
        "PAR0",
        "lambd",
        "alpha1",
        "alpha2",
        "beta",
        "T_crit",
        "T_mult",
        "gamma",
        "theta1",
        "theta2",
        "theta3",
    ]

    # Create an empty DataFrame to store the filtered mean and median values
    filtered_mean_median_df = pd.DataFrame(
        columns=["PFT", "Parameter", "Filtered_Mean", "Filtered_Median"]
    )

    # Group the data by PFT
    grouped = optimized_params_df_all.groupby("PFT")

    # Iterate over each parameter
    for parameter in parameters_to_plot:
        # Iterate over each group (PFT)
        for pft, group_data in grouped:
            # Calculate the percentiles for filtering
            q_low = group_data[parameter].quantile(0.05)
            q_high = group_data[parameter].quantile(0.95)

            # Filter the data
            filtered_data = group_data[
                (group_data[parameter] >= q_low) & (group_data[parameter] <= q_high)
            ]

            # Calculate the filtered mean and median
            filtered_mean = filtered_data[parameter].mean()
            filtered_median = filtered_data[parameter].median()

            # Append the results to the DataFrame
            filtered_mean_median_df = filtered_mean_median_df.append(
                {
                    "PFT": pft,
                    "Parameter": parameter,
                    "Filtered_Mean": filtered_mean,
                    "Filtered_Median": filtered_median,
                },
                ignore_index=True,
            )

    # Save the DataFrame to an Excel file
    filtered_mean_median_df.to_excel(
        base_path
        + "paramerters_mean_and_median_per_PFT_VPRM_"
        + VPRM_old_or_new
        + "_"
        + str(opt_method)
        + "_"
        + str(maxiter)
        + ".eps",
        "filtered_mean_median_values.xlsx",
        index=False,
    )
