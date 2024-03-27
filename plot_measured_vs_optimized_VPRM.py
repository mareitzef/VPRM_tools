import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_measured_vs_optimized_VPRM(
    df_year,
    nee,
    gpp,
    GPP_VPRM,
    GPP_VPRM_opt,
    r_eco,
    Reco_VPRM,
    Reco_VPRM_opt,
    base_path,
    folder,
    VPRM_old_or_new,
    year,
    opt_method,
    maxiter,
):

    fig, axes = plt.subplots(3, 1, figsize=(10, 18))

    # Plot comparison of GPP
    axes[0].plot(
        df_year.index,
        GPP_VPRM,
        linestyle="",
        marker="o",
        markersize=1,
        label="Modeled GPP",
        color="green",
    )
    axes[0].plot(
        df_year.index,
        GPP_VPRM_opt,
        linestyle="",
        marker="o",
        markersize=1,
        label="Modeled GPP optimized",
        color="red",
    )
    axes[0].plot(
        df_year.index,
        df_year[gpp],
        linestyle="",
        marker="o",
        markersize=1,
        label="Measured GPP",
        color="blue",
    )
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("GPP")
    axes[0].set_title("Comparison of Measured and Modeled GPP")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(
        df_year.index,
        Reco_VPRM,
        linestyle="",
        marker="o",
        markersize=1,
        label="Modeled Reco",
        color="green",
    )
    axes[1].plot(
        df_year.index,
        Reco_VPRM_opt,
        linestyle="",
        marker="o",
        markersize=1,
        label="Modeled Reco optimized",
        color="red",
    )
    axes[1].plot(
        df_year.index,
        df_year[r_eco],
        linestyle="",
        marker="o",
        markersize=1,
        label="Measured Reco",
        color="blue",
    )
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Reco")
    axes[1].set_title("Comparison of Measured and Modeled Reco")
    axes[1].legend()
    axes[1].grid(True)
    axes[2].plot(
        df_year.index,
        Reco_VPRM - GPP_VPRM,
        linestyle="",
        marker="o",
        markersize=1,
        label="Modeled NEE",
        color="green",
    )
    axes[2].plot(
        df_year.index,
        np.array(Reco_VPRM_opt) - np.array(GPP_VPRM_opt),
        linestyle="",
        marker="o",
        markersize=1,
        label="Modeled NEE optimized",
        color="red",
    )
    axes[2].plot(
        df_year.index,
        df_year[nee],
        linestyle="",
        marker="o",
        markersize=1,
        label="Measured NEE",
        color="blue",
    )
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("NEE")
    axes[2].set_title("Comparison of Measured and Modeled NEE")
    axes[2].legend()
    axes[2].grid(True)
    plt.tight_layout()

    plt.savefig(
        base_path
        + folder
        + "/compare_optimized_fluxes_VPRM_"
        + VPRM_old_or_new
        + "_"
        + str(year)
        + "_"
        + str(opt_method)
        + "_"
        + str(maxiter)
        + ".png"
    )
    plt.close(fig)

    return
