library(MODISTools)
library(openxlsx)
library(lubridate)
library(stringr)
library(readr)

######################################################################################
# This code automates the process of retrieving MODIS data for multiple sites within
# specified date ranges and saving them for subsequent analysis.
#
# - Reads a CSV file containing information about various sites, such as latitude and longitude.
# - Defines a function convert_to_float to convert coordinates from string format to numeric format.
# - Specifies a list of folders corresponding to different site datasets.
# - Iterates through each folder to extract site-specific information such as name, latitude, and longitude.
# - Creates a dataframe with site information.
# - Fetches MODIS data for each site using the MODISTools package, extracting specific bands for
# - vegetation indices and land surface water index (LSWI).
# - Writes the extracted data to Excel files for further analysis.
######################################################################################

site_info <- read_csv("site_info_europe.csv", col_types = cols(lat = col_character(), lon = col_character()))

convert_to_float <- function(coord) {
    as.numeric(gsub(",", ".", coord))
}

base_folder <- "/home/madse/Downloads/Fluxnet_Data/"
folders <- c(
    "FLX_RU-Fyo_FLUXNET2015_FULLSET_1998-2014_2-4",
    "FLX_IT-Col_FLUXNET2015_FULLSET_1996-2014_1-4",
    "FLX_FI-Sod_FLUXNET2015_FULLSET_2001-2014_1-4",
    "FLX_DE-Tha_FLUXNET2015_FULLSET_1996-2014_1-4",
    "FLX_NL-Loo_FLUXNET2015_FULLSET_1996-2014_1-4",
    "FLX_DE-Spw_FLUXNET2015_FULLSET_2010-2014_1-4"
)

# folders for Europe
# folders <- c(
#     "FLX_BE-Bra_FLUXNET2015_FULLSET_1996-2014_2-4",
#     "FLX_BE-Lon_FLUXNET2015_FULLSET_2004-2014_1-4",
#     "FLX_BE-Vie_FLUXNET2015_FULLSET_1996-2014_1-4",
#     "FLX_CZ-BK1_FLUXNET2015_FULLSET_2004-2014_2-4",
#     "FLX_CZ-BK2_FLUXNET2015_FULLSET_2004-2012_2-4",
#     "FLX_DE-Akm_FLUXNET2015_FULLSET_2009-2014_1-4",
#     "FLX_DE-Geb_FLUXNET2015_FULLSET_2001-2014_1-4",
#     "FLX_DE-Gri_FLUXNET2015_FULLSET_2004-2014_1-4",
#     "FLX_DE-Hai_FLUXNET2015_FULLSET_2000-2012_1-4",
#     "FLX_DE-Kli_FLUXNET2015_FULLSET_2004-2014_1-4",
#     "FLX_DE-Lnf_FLUXNET2015_FULLSET_2002-2012_1-4",
#     "FLX_DE-Obe_FLUXNET2015_FULLSET_2008-2014_1-4",
#     "FLX_DE-RuR_FLUXNET2015_FULLSET_2011-2014_1-4",
#     "FLX_DE-RuS_FLUXNET2015_FULLSET_2011-2014_1-4",
#     "FLX_DE-Seh_FLUXNET2015_FULLSET_2007-2010_1-4",
#     "FLX_DE-Spw_FLUXNET2015_FULLSET_2010-2014_1-4",
#     "FLX_DE-Tha_FLUXNET2015_FULLSET_1996-2014_1-4",
#     "FLX_DE-Zrk_FLUXNET2015_FULLSET_2013-2014_2-4",
#     "FLX_DK-Eng_FLUXNET2015_FULLSET_2005-2008_1-4",
#     "FLX_DK-Fou_FLUXNET2015_FULLSET_2005-2005_1-4",
#     "FLX_DK-Sor_FLUXNET2015_FULLSET_1996-2014_2-4",
#     "FLX_ES-Amo_FLUXNET2015_FULLSET_2007-2012_1-4",
#     "FLX_ES-LgS_FLUXNET2015_FULLSET_2007-2009_1-4",
#     "FLX_ES-LJu_FLUXNET2015_FULLSET_2004-2013_1-4",
#     "FLX_ES-Ln2_FLUXNET2015_FULLSET_2009-2009_1-4",
#     "FLX_FI-Hyy_FLUXNET2015_FULLSET_1996-2014_1-4",
#     "FLX_FI-Jok_FLUXNET2015_FULLSET_2000-2003_1-4",
#     "FLX_FI-Let_FLUXNET2015_FULLSET_2009-2012_1-4",
#     "FLX_FI-Lom_FLUXNET2015_FULLSET_2007-2009_1-4",
#     "FLX_FI-Sod_FLUXNET2015_FULLSET_2001-2014_1-4",
#     "FLX_FR-Fon_FLUXNET2015_FULLSET_2005-2014_1-4",
#     "FLX_FR-Gri_FLUXNET2015_FULLSET_2004-2014_1-4",
#     "FLX_FR-LBr_FLUXNET2015_FULLSET_1996-2008_1-4",
#     "FLX_FR-Pue_FLUXNET2015_FULLSET_2000-2014_2-4",
#     "FLX_IT-BCi_FLUXNET2015_FULLSET_2004-2014_2-4",
#     "FLX_IT-CA1_FLUXNET2015_FULLSET_2011-2014_2-4",
#     "FLX_IT-CA2_FLUXNET2015_FULLSET_2011-2014_2-4",
#     "FLX_IT-CA3_FLUXNET2015_FULLSET_2011-2014_2-4",
#     "FLX_IT-Col_FLUXNET2015_FULLSET_1996-2014_1-4",
#     "FLX_IT-Cp2_FLUXNET2015_FULLSET_2012-2014_2-4",
#     "FLX_IT-Cpz_FLUXNET2015_FULLSET_1997-2009_1-4",
#     "FLX_IT-Noe_FLUXNET2015_FULLSET_2004-2014_2-4",
#     "FLX_IT-Ro1_FLUXNET2015_FULLSET_2000-2008_1-4",
#     "FLX_IT-Ro2_FLUXNET2015_FULLSET_2002-2012_1-4",
#     "FLX_IT-SR2_FLUXNET2015_FULLSET_2013-2014_1-4",
#     "FLX_IT-SRo_FLUXNET2015_FULLSET_1999-2012_1-4",
#     "FLX_NL-Hor_FLUXNET2015_FULLSET_2004-2011_1-4",
#     "FLX_NL-Loo_FLUXNET2015_FULLSET_1996-2014_1-4",
#     "FLX_RU-Fyo_FLUXNET2015_FULLSET_1998-2014_2-4"
# )
# folders for the Alps
# folders <- c(
#     "FLX_AT-Mie_FLUXNET2015_FULLSET_2022-2022_1-4/",
#     "FLX_DE-SfN_FLUXNET2015_FULLSET_2012-2014_1-4/",
#     "FLX_CH-Cha_FLUXNET2015_FULLSET_2005-2014_2-4/",
#     "FLX_IT-Isp_FLUXNET2015_FULLSET_2013-2014_1-4/",
#     "FLX_CH-Dav_FLUXNET2015_FULLSET_1997-2014_1-4/",
#     "FLX_IT-La2_FLUXNET2015_FULLSET_2000-2002_1-4/",
#     "FLX_CH-Fru_FLUXNET2015_FULLSET_2005-2014_2-4/",
#     "FLX_IT-Lav_FLUXNET2015_FULLSET_2003-2014_2-4/",
#     "FLX_CH-Lae_FLUXNET2015_FULLSET_2004-2014_1-4/",
#     "FLX_IT-MBo_FLUXNET2015_FULLSET_2003-2013_1-4/",
#     "FLX_CH-Oe1_FLUXNET2015_FULLSET_2002-2008_2-4/",
#     "FLX_IT-PT1_FLUXNET2015_FULLSET_2002-2004_1-4/",
#     "FLX_CH-Oe2_FLUXNET2015_FULLSET_2004-2014_1-4/",
#     "FLX_IT-Ren_FLUXNET2015_FULLSET_1998-2013_1-4/",
#     "FLX_CZ-wet_FLUXNET2015_FULLSET_2006-2014_1-4/",
#     "FLX_IT-Tor_FLUXNET2015_FULLSET_2008-2014_2-4/",
#     "FLX_AT-Neu_FLUXNET2015_FULLSET_2002-2012_1-4/",
#     "FLX_DE-Lkb_FLUXNET2015_FULLSET_2009-2013_1-4/",
#     "FLX_IT-Isp_FLUXNET2015_FULLSET_2013-2014_1-4"
# )

# Loop through each site folder
for (folder in folders) {
    site_name <- str_extract(folder, "(?<=FLX_)[A-Z]{2}-[A-Za-z0-9]+")
    year_range <- gsub(".*_(\\d{4})-(\\d{4})_.*", "\\1-\\2", folder)
    start_year <- as.numeric(str_extract(year_range, "\\d{4}"))
    end_year <- as.numeric(str_extract(year_range, "(?<=-)\\d{4}"))
    start_date <- ymd(paste0(start_year, "-01-01")) - days(20)
    end_date <- ymd(paste0(end_year, "-12-31")) + days(20)
    start_date <- format(start_date, "%Y-%m-%d")
    end_date <- format(end_date, "%Y-%m-%d")

    # find coordinates of site
    for (i in seq_len(nrow(site_info))) {
        if (site_name == site_info$site[i]) {
            lat <- convert_to_float(site_info$lat[i])
            lon <- convert_to_float(site_info$lon[i])
            cat("Site Name:", site_name, "\n")
            cat("Latitude:", lat, "\n")
            cat("Longitude:", lon, "\n\n")
            break
        }
    }



    # Create data frame with site information
    path <- paste0(base_folder, folder, "/")
    df <- data.frame(
        "site_name" = site_name,
        "lat" = lat,
        "lon" = lon
    )

    # GEt MODIS Data
    products <- c("MOD13Q1", "MYD13Q1")
    bands <- c("250m_16_days_EVI", "250m_16_days_NDVI")
    for (product in products) {
        for (band in bands) {
            subsets <- mt_batch_subset(
                df = df,
                product = product,
                band = band,
                internal = TRUE,
                start = start_date,
                end = end_date
            )
            # print(str(subsets))
            file_name <- paste0(path, site_name, "_", product, "_", band, "_", start_date, "_", end_date, ".xlsx")
            write.xlsx(subsets, file = file_name)
        }
    }
    # Land Surface Water Index (LSWI):
    #   LSWI =  (NIR (Band 2) - SWIR (Band 6)) / (NIR + SWIR)
    products <- c("MOD09A1", "MYD09A1")
    bands <- c("sur_refl_b02", "sur_refl_b06")
    for (product in products) {
        for (band in bands) {
            subsets <- mt_batch_subset(
                df = df,
                product = product,
                band = band,
                internal = TRUE,
                start = start_date,
                end = end_date
            )
            # print(str(subsets))
            file_name <- paste0(path, site_name, "_", product, "_", band, "_", start_date, "_", end_date, ".xlsx")
            write.xlsx(subsets, file = file_name)
        }
    }
    # Land Surface Water Index (LSWI):
    #   LSWI =  (NIR (Band 2) - SWIR (Band 6)) / (NIR + SWIR)
    products <- c("MOD15A2H", "MYD15A2H")
    bands <- c("Fpar_500m", "Lai_500m") # TODO: doe we need this: "FparStdDev_500m", "LaiStdDev_500m"
    for (product in products) {
        for (band in bands) {
            subsets <- mt_batch_subset(
                df = df,
                product = product,
                band = band,
                internal = TRUE,
                start = start_date,
                end = end_date
            )
            # print(str(subsets))
            file_name <- paste0(path, site_name, "_", product, "_", band, "_", start_date, "_", end_date, ".xlsx")
            write.xlsx(subsets, file = file_name)
        }
    }
}
