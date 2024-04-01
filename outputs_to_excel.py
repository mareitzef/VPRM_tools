import pandas as pd


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
