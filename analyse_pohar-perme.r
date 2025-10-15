#!/usr/bin/env Rscript

suppressMessages({
  library(relsurv)
  library(survival)
  library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
output_file <- args[2]

# Lecture des données transmises par Python
hm <- read_csv(input_file, show_col_types = FALSE)

if (!all(c("time", "status") %in% names(hm))) {
  stop("Colonnes 'time' et 'status' manquantes dans le fichier.")
}

# Charger le jeu de population
load("fr_ratetable.rda")  # contient `fr_ratetable`

# Calcul de la survie nette (Pohar-Perme)
res <- rs.surv(Surv(time, status) ~ 1,
               data = hm,
               ratetable = fr_ratetable,
               method = "pohar-perme")

summary_res <- summary(res)

# Exporter un résumé au format CSV
df_res <- data.frame(
  time = summary_res$time,
  surv = summary_res$surv,
  lower = summary_res$lower,
  upper = summary_res$upper
)

write_csv(df_res, output_file)
cat("Résultats sauvegardés dans", output_file, "\n")
