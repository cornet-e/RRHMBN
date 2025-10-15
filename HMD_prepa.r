library(readr)
library(dplyr)
library(relsurv)

# Lire les données HMD (par exemple "LT.txt")
hmd <- read_table2("LT.txt", col_types = cols())

# Suppose qu’il y a colonne `qx` (probabilité de décès sur l’année)
hmd2 <- hmd %>%
  mutate(mx = -log(1 - qx)) %>%
  select(Year, Age, mx)

# Créer un ratetable (vérifie syntaxe selon relsurv version)
fr_ratetable <- as.rate.table(hmd2, ashaz = "mx", times = "Age", years = "Year")

# Sauvegarde
save(fr_ratetable, file = "fr_ratetable.rda")