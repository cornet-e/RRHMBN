# Liste des packages à charger automatiquement
packages <- c("curl","httr","dplyr", "ggplot2", "tidyr", "readr", "stringr","shiny","plotly","lubridate","data.table","forcats","purrr","tibble","caret","survival","randomForest","stats","mgcv","popEpi","Epi","jsonlite","DT","GWarK")

# Fonction pour charger les packages (en installant ceux qui ne sont pas déjà installés)
load_packages <- function(packages) {
  for (pkg in packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      install.packages(pkg)
    }
    library(pkg, character.only = TRUE)
  }
}

# Charger les packages
load_packages(packages)
