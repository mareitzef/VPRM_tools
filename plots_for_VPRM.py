import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_measured_vs_optimized_VPRM(
    site_name,
    timestamp,
    df_year,
    nee,
    r_eco,
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
    gpp,
):
    df_year.set_index(timestamp, inplace=True)
    font_size = 14

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
        df_year[r_eco],  # TODO switch for V13* df_year["NIGHT"],
        linestyle="",
        marker="o",
        markersize=1,
        label=r_eco,
        color="blue",
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
    axes[0].set_xlabel("Date", fontsize=font_size + 2)
    axes[0].set_ylabel("Reco", fontsize=font_size + 2)
    axes[0].set_title(site_name + " - Measured and Modeled GPP", fontsize=font_size + 2)
    axes[0].legend(fontsize=font_size)
    axes[0].grid(True)
    axes[0].tick_params(axis="both", which="major", labelsize=font_size)

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
        df_year[gpp],
        linestyle="",
        marker="o",
        markersize=1,
        label=gpp,
        color="blue",
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
    axes[1].set_xlabel("Date", fontsize=font_size + 2)
    axes[1].set_ylabel("GPP", fontsize=font_size + 2)
    axes[1].legend(fontsize=font_size)
    axes[1].grid(True)
    axes[1].tick_params(axis="both", which="major", labelsize=font_size)

    # Plot comparison of NEE
    axes[2].plot(
        df_year.index,
        Reco_VPRM - GPP_VPRM,
        linestyle="",
        marker="o",
        markersize=1,
        label="Modeled NEE first guess",
        color="green",
    )
    axes[2].plot(
        df_year.index,
        df_year[nee],
        linestyle="",
        marker="o",
        markersize=1,
        label=nee,
        color="blue",
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
    axes[2].set_xlabel("Date", fontsize=font_size + 2)
    axes[2].set_ylabel("NEE", fontsize=font_size + 2)
    axes[2].legend(fontsize=font_size)
    axes[2].grid(True)
    axes[2].tick_params(axis="both", which="major", labelsize=font_size)
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
        + ".png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    return


def plot_site_input(
    df_site_and_modis, timestamp, site_name, folder, base_path, variables
):

    df_site_and_modis.set_index(timestamp, inplace=True)
    font_size = 12

    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8.27, 11.69))
    for i, var in enumerate(variables):
        row_index = i // 2  # Calculate the row index
        col_index = i % 2  # Calculate the column index

        axes[row_index, col_index].plot(
            df_site_and_modis.index,
            df_site_and_modis[var],
            label=var,
            linestyle="",
            marker="o",
            markersize=1,
        )
        axes[row_index, col_index].set_xlabel("Date", fontsize=font_size + 2)
        axes[row_index, col_index].set_ylabel(var, fontsize=font_size + 2)
        # axes[row_index, col_index].legend(fontsize=font_size)
        axes[row_index, col_index].grid(True)
        axes[row_index, col_index].tick_params(
            axis="both", which="major", labelsize=font_size, rotation=45
        )

    axes[0, 0].set_title(site_name + " - input data", fontsize=font_size + 2)
    plt.tight_layout()
    plt.savefig(
        base_path + folder + "/" + site_name + "_check_input.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_measured_vs_modeled(
    df_site_and_modis, site_name, folder, base_path, VPRM_old_or_new, gpp, r_eco, nee
):
    font_size = 14
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
        df_site_and_modis["GPP_first_guess"],
        label="Modeled GPP",
        color="green",
        linestyle="",
        marker="o",
        markersize=1,
    )
    axes[0].set_xlabel("Date", fontsize=font_size + 2)
    axes[0].set_ylabel("GPP", fontsize=font_size + 2)
    axes[0].set_title(site_name + " - Measured and Modeled GPP", fontsize=font_size + 2)
    axes[0].legend(fontsize=font_size)
    axes[0].tick_params(axis="both", which="major", labelsize=font_size)
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
        df_site_and_modis["Reco_first_guess"],
        label="Modeled Reco",
        color="green",
        linestyle="",
        marker="o",
        markersize=1,
    )
    axes[1].set_xlabel("Date", fontsize=font_size + 2)
    axes[1].set_ylabel("Reco", fontsize=font_size + 2)
    axes[1].set_title("Measured and Modeled Reco", fontsize=font_size + 2)
    axes[1].tick_params(axis="both", which="major", labelsize=font_size)
    axes[1].legend(fontsize=font_size)
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
        df_site_and_modis["Reco_first_guess"] - df_site_and_modis["GPP_first_guess"],
        label="Modeled NEE",
        color="green",
        linestyle="",
        marker="o",
        markersize=1,
    )
    axes[2].set_xlabel("Date", fontsize=font_size + 2)
    axes[2].set_ylabel("NEE", fontsize=font_size + 2)
    axes[2].set_title("Measured and Modeled NEE", fontsize=font_size + 2)
    axes[2].tick_params(axis="both", which="major", labelsize=font_size)
    axes[2].legend(fontsize=font_size)
    axes[2].grid(True)
    plt.tight_layout()
    plt.savefig(
        base_path
        + folder
        + "/"
        + site_name
        + "_first_guess_fluxes_VPRM_"
        + VPRM_old_or_new
        + "_all_years.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)
