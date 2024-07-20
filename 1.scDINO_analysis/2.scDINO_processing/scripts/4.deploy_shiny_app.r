suppressPackageStartupMessages(suppressWarnings(library(dplyr)))
suppressPackageStartupMessages(suppressWarnings(library(ggplot2)))
suppressPackageStartupMessages(suppressWarnings(library(rsconnect)))

Sys.setenv(RSCONNECT_NAME='lippincm')
Sys.setenv(RSCONNECT_TOKEN='26C65D7226A3C01914DAF5FA0AFC3AF9')
Sys.setenv(RSCONNECT_SECRET='bk50r2IoqGRgjmr8cMiJn/WntXxMSnH9VqQUZgmI')

Sys.getenv("RSCONNECT_NAME")

# connect to the shinyapps.io account
rsconnect::setAccountInfo(
  name = Sys.getenv("RSCONNECT_NAME"),
  token = Sys.getenv("RSCONNECT_TOKEN"),
  secret = Sys.getenv("RSCONNECT_SECRET")
)

# deploy the app
rsconnect::deployApp(appDir = "../temporal_shiny_app", appName = "temporal_shiny_app")
