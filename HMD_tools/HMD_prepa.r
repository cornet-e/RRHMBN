library(readr)
library(dplyr)
library(relsurv)

# Lire les données HMD (par exemple "LT.txt")
hmd <- read_table("LT.txt", col_types = cols())

# Suppose qu’il y a colonne `qx` (probabilité de décès sur l’année)
hmd2 <- hmd %>%
  mutate(mx = -log(1 - qx)) %>%
  select(Year, Age, mx)

# Créer un ratetable "manuellement" pour relsurv
# On crée une matrice Age x Year
ages <- sort(unique(hmd2$Age))
years <- sort(unique(hmd2$Year))

mx_mat <- matrix(NA, nrow = length(ages), ncol = length(years),
                 dimnames = list(Age = ages, Year = years))

# Remplir la matrice
for (i in seq_along(ages)){
  for (j in seq_along(years)){
    mx_mat[i, j] <- hmd2$mx[hmd2$Age == ages[i] & hmd2$Year == years[j]]
  }
}

# Convertir en objet ratetable compatible relsurv
fr_ratetable <- relsurv::ratetable(mx_mat, start = min(ages))

# Sauvegarde
save(fr_ratetable, file = "fr_ratetable.rda")