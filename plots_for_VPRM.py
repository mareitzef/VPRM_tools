import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_measured_vs_optimized_VPRM(
    site_name,
    timestamp,
    df_year,
    nee,
    nee_mean,
    GPP_VPRM,
    GPP_VPRM_opt,
    Reco_VPRM,
    Reco_VPRM_opt,
    base_path,
    folder,
    VPRM_old_or_new,
    year,
    opt_method,
    maxiter,
):
    df_year.set_index(timestamp, inplace=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 18))

    # Plot comparison of Reco
    axes[0].plot(
        df_year.index,
        Reco_VPRM,
        linestyle="",
        marker="o",
        markersize=1,
        label="Modeled Reco first guess",
        color="green",
    )
    axes[0].plot(
        df_year.index,
        Reco_VPRM_opt,
        linestyle="",
        marker="o",
        markersize=1,
        label="Modeled Reco optimized",
        color="red",
    )
    axes[0].plot(
        df_year.index,
        df_year[nee_mean] * df_year["NIGHT"],
        linestyle="",
        marker="o",
        markersize=1,
        label="Measured nighttime NEE MEAN",
        color="blue",
    )
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Reco")
    axes[0].set_title("Measured and Modeled Reco")
    axes[0].legend()
    axes[0].grid(True)

    # Plot comparison of GPP
    axes[1].plot(
        df_year.index,
        GPP_VPRM,
        linestyle="",
        marker="o",
        markersize=1,
        label="Modeled GPP first guess",
        color="green",
    )
    axes[1].plot(
        df_year.index,
        GPP_VPRM_opt,
        linestyle="",
        marker="o",
        markersize=1,
        label="Modeled GPP optimized",
        color="red",
    )

    df_year["GPP_calc"] = -(df_year[nee] - Reco_VPRM_opt)
    df_year.loc[df_year["GPP_calc"] < 0, "GPP_calc"] = 0

    axes[1].plot(
        df_year.index,
        df_year["GPP_calc"],
        linestyle="",
        marker="o",
        markersize=1,
        label="'Measured' GPP (NEE - Reco_modeled )",
        color="blue",
    )
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("GPP")
    axes[1].set_title(site_name + " - Measured and Modeled GPP")
    axes[1].legend()
    axes[1].grid(True)

    # Plot comparison of NEE
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
    axes[2].set_title("Measured and Modeled NEE")
    axes[2].legend()
    axes[2].grid(True)
    plt.tight_layout()

    plt.savefig(
        base_path
        + folder
        + "/"
        + site_name
        + "_opt_fluxes_VPRM_"
        + VPRM_old_or_new
        + "_"
        + str(year)
        + "_"
        + str(opt_method)
        + "_"
        + str(maxiter)
        + ".eps",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return


def plot_site_year(
    df_site_and_modis, timestamp, site_name, folder, base_path, variables
):

    df_site_and_modis.set_index(timestamp, inplace=True)

    fig, axes = plt.subplots(
        nrows=len(variables), ncols=1, figsize=(10, 6 * len(variables))
    )
    for i, var in enumerate(variables):
        axes[i].plot(
            df_site_and_modis.index,
            df_site_and_modis[var],
            label=var,
            linestyle="",
            marker="o",
            markersize=1,
        )
        axes[i].set_xlabel("Date")
        axes[i].set_ylabel(var)
        axes[i].legend()
        axes[i].grid(True)
    axes[0].set_title(site_name + " - input data")
    plt.tight_layout()
    plt.savefig(
        base_path + folder + "/" + site_name + "_check_input.eps",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_measured_vs_modeled(
    df_site_and_modis, site_name, folder, base_path, VPRM_old_or_new, gpp, r_eco, nee
):
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    axes[0].plot(
        df_site_and_modis.index,
        df_site_and_modis[gpp],
        label="Measured GPP",
        color="blue",
        linestyle="",
        marker="o",
        markersize=1,
    )
    axes[0].plot(
        df_site_and_modis.index,
        df_site_and_modis["GPP_VPRM_first_guess"],
        label="Modeled GPP",
        color="green",
        linestyle="",
        marker="o",
        markersize=1,
    )
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("GPP")
    axes[0].set_title(site_name + " - Measured and Modeled GPP")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(
        df_site_and_modis.index,
        df_site_and_modis[r_eco],
        label="Measured Reco",
        color="blue",
        linestyle="",
        marker="o",
        markersize=1,
    )
    axes[1].plot(
        df_site_and_modis.index,
        df_site_and_modis["Reco_VPRM_first_guess"],
        label="Modeled Reco",
        color="green",
        linestyle="",
        marker="o",
        markersize=1,
    )
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Reco")
    axes[1].set_title("Measured and Modeled Reco")
    axes[1].legend()
    axes[1].grid(True)
    axes[2].plot(
        df_site_and_modis.index,
        df_site_and_modis[nee],
        label="Measured NEE",
        color="blue",
        linestyle="",
        marker="o",
        markersize=1,
    )
    axes[2].plot(
        df_site_and_modis.index,
        df_site_and_modis["Reco_VPRM_first_guess"]
        - df_site_and_modis["GPP_VPRM_first_guess"],
        label="Modeled NEE",
        color="green",
        linestyle="",
        marker="o",
        markersize=1,
    )
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("NEE")
    axes[2].set_title("Measured and Modeled NEE")
    axes[2].legend()
    axes[2].grid(True)
    plt.tight_layout()
    plt.savefig(
        base_path
        + folder
        + "/"
        + site_name
        + "_first_guess_fluxes_VPRM_"
        + VPRM_old_or_new
        + "_all_years.eps",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
