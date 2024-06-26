# load the library
library(MODISTools)
library(openxlsx)

# necessary Modis Products for VPRM
# MOD13Q1/MYD13Q1
# MOD09A1/MYD09A1
product <- "MCD15A3H"

products <- mt_products()
head(products)
bands <- mt_bands(product)
bands
# dates <- mt_dates(product = product, lat, lon)
# head(dates)
