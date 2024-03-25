library(MODISTools)
library(openxlsx)
library(lubridate)
library(stringr)
library(readr)

site_info <- read_csv("/home/madse/Downloads/Fluxnet_Data/site_info_Alps_lat44-50_lon5-17.csv", col_types = cols(lat = col_character(), lon = col_character()))

convert_to_float <- function(coord) {
    as.numeric(gsub(",", ".", coord))
}

folders <- c(
    "FLX_DE-SfN_FLUXNET2015_FULLSET_2012-2014_1-4/",
    "FLX_CH-Cha_FLUXNET2015_FULLSET_2005-2014_2-4/",
    "FLX_IT-Isp_FLUXNET2015_FULLSET_2013-2014_1-4/",
    "FLX_CH-Dav_FLUXNET2015_FULLSET_1997-2014_1-4/",
    "FLX_IT-La2_FLUXNET2015_FULLSET_2000-2002_1-4/",
    "FLX_CH-Fru_FLUXNET2015_FULLSET_2005-2014_2-4/",
    "FLX_IT-Lav_FLUXNET2015_FULLSET_2003-2014_2-4/",
    "FLX_CH-Lae_FLUXNET2015_FULLSET_2004-2014_1-4/",
    "FLX_IT-MBo_FLUXNET2015_FULLSET_2003-2013_1-4/",
    "FLX_CH-Oe1_FLUXNET2015_FULLSET_2002-2008_2-4/",
    "FLX_IT-PT1_FLUXNET2015_FULLSET_2002-2004_1-4/",
    "FLX_CH-Oe2_FLUXNET2015_FULLSET_2004-2014_1-4/",
    "FLX_IT-Ren_FLUXNET2015_FULLSET_1998-2013_1-4/",
    "FLX_CZ-wet_FLUXNET2015_FULLSET_2006-2014_1-4/",
    "FLX_IT-Tor_FLUXNET2015_FULLSET_2008-2014_2-4/",
    "FLX_AT-Neu_FLUXNET2015_FULLSET_2002-2012_1-4/",
    "FLX_DE-Lkb_FLUXNET2015_FULLSET_2009-2013_1-4/"
)

# Loop through each site folder
for (folder in folders) {
    site_name <- str_extract(folder, "(?<=FLX_)[A-Z]{2}-[A-Za-z]+") # TODO: numbers are omitted try spliting
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
    path <- paste0("/home/madse/Downloads/Fluxnet_Data/", folder, "/")
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
}
