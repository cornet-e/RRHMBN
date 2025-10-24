# --- Chargement des packages ---
library(relsurv)
library(dplyr)
library(readr)
library(ggplot2)
library(tidyr)
library(plotly)
library(survival)
library(tibble)

# --- 1. Chargement des données ---
data <- read_csv("hm.csv") %>%
  # renommer et transformer les variables
  mutate(
    time = fup * 30.4375,    # transformer le suivi de mois en jours
    #time = fup/12,             # suivi en mois => en années
    status = event,           # statut (1 = décès, 0 = vivant)
    age_days = age * 365.24,                # age déjà en années => en jours
    sex_num = sex,  # 1 = homme, 2 = femme
    year_frac = as.numeric(format(DateDuDiag, "%Y")) +
      ((as.numeric(format(DateDuDiag, "%m"))-0.5)/12)
  )
options(digits=10)

# --- 2. Chargement de la table de mortalité ---
# --- Fonction pour lire un fichier txt et harmoniser les colonnes ---
read_lifetable <- function(file_path) {
  # lire le fichier txt (supposons tabulé ou séparé par espace)
  lifetable <- read.table(file_path, header = TRUE, sep = "", stringsAsFactors = FALSE, skip = 2)
  
  # renommer les colonnes
  lifetable <- lifetable %>%
    rename(
      year = Year,
      age = Age,
      rate = mx
    )
  
  return(lifetable)
}

# --- Fonction pour créer la ratetable ---
create_ratetable <- function(mlt, flt) {
  # séquences d'âges et d'années
  ages <- sort(unique(c(mlt$age, flt$age)))
  years <- sort(unique(c(mlt$year, flt$year)))
  
  # initialiser les matrices
  men_matrix <- matrix(NA, nrow = length(ages), ncol = length(years),
                       dimnames = list(ages, years))
  women_matrix <- matrix(NA, nrow = length(ages), ncol = length(years),
                         dimnames = list(ages, years))
  
  # remplir la matrice hommes
  for (i in seq_along(ages)) {
    for (j in seq_along(years)) {
      val <- mlt$rate[mlt$age == ages[i] & mlt$year == years[j]]
      men_matrix[i,j] <- ifelse(length(val) == 1, 1-val, NA)
    }
  }
  
  # remplir la matrice femmes
  for (i in seq_along(ages)) {
    for (j in seq_along(years)) {
      val <- flt$rate[flt$age == ages[i] & flt$year == years[j]]
      women_matrix[i,j] <- ifelse(length(val) == 1, 1-val, NA)
    }
  }
  
  # créer la ratetable
  ratetable <- transrate(
    men = men_matrix,
    women = women_matrix,
    yearlim = range(years),
    int.length = 1
  )
  
  return(ratetable)
}

create_single_ratetable <- function(lt, sex = c("men", "women")) {
  sex <- match.arg(sex)
  ages <- sort(unique(lt$age))
  years <- sort(unique(lt$year))
  
  rate_matrix <- matrix(NA, nrow = length(ages), ncol = length(years),
                        dimnames = list(ages, years))
  
  for (i in seq_along(ages)) {
    for (j in seq_along(years)) {
      val <- lt$rate[lt$age == ages[i] & lt$year == years[j]]
      rate_matrix[i, j] <- ifelse(length(val) == 1, 1 - val, NA)
    }
  }
  
  # matrice vide de même dimension pour le sexe non utilisé
  empty_matrix <- matrix(NA, nrow = length(ages), ncol = length(years),
                         dimnames = list(ages, years))
  
  ratetable <- transrate(
    men = if (sex == "men") rate_matrix else empty_matrix,
    women = if (sex == "women") rate_matrix else empty_matrix,
    yearlim = range(years),
    int.length = 1
  )
  
  return(ratetable)
}



# --- chargement des tables sources (TXT) ---
mlt <- read_lifetable("mltper_1x1.txt")  # fichier hommes
flt <- read_lifetable("fltper_1x1.txt")  # fichier femmes

# Ratetable combinée
fr.ratetable <- create_ratetable(mlt, flt)

# Ratetable homme uniquement
fr.ratetable_male <- create_single_ratetable(mlt, sex = "men")

# Ratetable femme uniquement
fr.ratetable_female <- create_single_ratetable(flt, sex = "women")



# --- 5. Analyse de survie relative selon Pohar Perme ---
fit.pp <- rs.surv(Surv(time, status) ~ 1,
                  data = data,
                  ratetable = fr.ratetable,
                  rmap = list(
                    age = age_days,
                    sex = sex_num,
                    year = year_frac),
                  method = "pohar-perme")

summary(fit.pp)
plot(fit.pp, conf.int = TRUE, main = "Survie relative (Pohar Perme)")

fit.pp.male <- rs.surv(Surv(time, status) ~ 1,
                  data = data %>% filter(sex_num == 1),
                  ratetable = fr.ratetable,
                  rmap = list(
                    age = age_days,
                    sex = sex_num,
                    year = year_frac),
                  method = "pohar-perme")

#summary(fit.pp.male)
plot(fit.pp.male, conf.int = TRUE, main = "Survie relative (Pohar Perme)")


fit.pp.female <- rs.surv(Surv(time, status) ~ 1,
                       data = data %>% filter(sex_num == 2),
                       ratetable = fr.ratetable,
                       rmap = list(
                         age = age_days,
                         sex = sex_num,
                         year = year_frac),
                       method = "pohar-perme")

#summary(fit.pp.female)
plot(fit.pp.female, conf.int = TRUE, main = "Survie relative (Pohar Perme)")


# Vérification de la survie attendue
fit_exp <- survexp(
  formula = Surv(time, status) ~ 1,
  data = data,
  ratetable = fr.ratetable,
  rmap = list(age = age_days, sex = sex_num, year = year_frac)
)

plot(fit_exp, main = "Survie attendue (population générale)", xlab = "Années", ylab = "Survie attendue")

fit_exp_male <- survexp(
  formula = Surv(time, status) ~ 1,
  data = data %>% filter(sex_num == 1),
  ratetable = fr.ratetable_male,
  rmap = list(age = age_days, sex = sex_num, year = year_frac)
)

plot(fit_exp_male, main = "Survie attendue (population générale, Hommes)", xlab = "Années", ylab = "Survie attendue")


fit_exp_female <- survexp(
  formula = Surv(time, status) ~ 1,
  data = data %>% filter(sex_num == 2),
  ratetable = fr.ratetable_female,
  rmap = list(age = age_days, sex = sex_num, year = year_frac)
)

plot(fit_exp_female, main = "Survie attendue (population générale, Femmes)", xlab = "Années", ylab = "Survie attendue")


# --- Extraction des résultats de manière sûre ---
extract_surv <- function(fit, data, ratetable, step = 1) {
  times <- seq(0, max(data$time), by = step)
  
  # survie relative (Pohar-Perme)
  s_rel <- summary(fit, times = times)
  
  # survie attendue à partir de la ratetable
  fit_exp <- survexp(
    formula = Surv(time, status) ~ 1,
    data = data,
    ratetable = ratetable,
    rmap = list(age = age_days, sex = sex_num, year = year_frac)
  )
  
  s_exp <- summary(fit_exp, times = times)
  
  tibble(
    time = s_rel$time,
    surv_rel = s_rel$surv,
    lower = s_rel$lower,
    upper = s_rel$upper,
    surv_exp = s_exp$surv
  )
}



df_plot <- extract_surv(fit.pp, data, fr.ratetable)


p <- ggplot(df_plot, aes(x = time)) +
  geom_line(aes(y = surv_rel), linewidth = 1.2, color = "blue") +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, color = NA) +
  geom_line(aes(y = surv_exp), linetype = "dashed", linewidth = 1, color = "black") + # expected survival
 
  labs(
    x = "Temps (jours)",
    y = "Survie relative",
    title = "Survie relative (Pohar Perme)",
    caption = "Courbe en pointillés : survie attendue"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.title = element_blank())

ggplotly(p, tooltip = c("time", "surv_rel", "lower", "upper", "surv_exp"))


##### MALES #####
# --- Extraction des résultats de manière sûre ---
extract_surv_male <- function(fit, data, ratetable, step = 1) {
  times <- seq(0, max(data$time), by = step)
  
  # survie relative (Pohar-Perme)
  s_rel <- summary(fit, times = times)
  
  # survie attendue à partir de la ratetable
  fit_exp <- survexp(
    formula = Surv(time, status) ~ 1,
    data = data %>% filter(sex_num == 1),
    ratetable = ratetable,
    rmap = list(age = age_days, sex = sex_num, year = year_frac)
  )
  
  s_exp <- summary(fit_exp, times = times)
  
  tibble(
    time = s_rel$time,
    surv_rel = s_rel$surv,
    lower = s_rel$lower,
    upper = s_rel$upper,
    surv_exp = s_exp$surv
  )
}



df_plot_male <- extract_surv_male(fit.pp.male, data, fr.ratetable_male)


p_male <- ggplot(df_plot_male, aes(x = time)) +
  geom_line(aes(y = surv_rel), linewidth = 1.2, color = "blue") +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, color = NA) +
  geom_line(aes(y = surv_exp), linetype = "dashed", linewidth = 1, color = "black") + # expected survival
  
  labs(
    x = "Temps (jours)",
    y = "Survie relative",
    title = "Survie relative (Pohar Perme) - Hommes",
    caption = "Courbe en pointillés : survie attendue"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.title = element_blank())

ggplotly(p_male, tooltip = c("time", "surv_rel", "lower", "upper", "surv_exp"))

##### FEMALES ####
# --- Extraction des résultats de manière sûre ---
extract_surv_female <- function(fit, data, ratetable, step = 1) {
  times <- seq(0, max(data$time), by = step)
  
  # survie relative (Pohar-Perme)
  s_rel <- summary(fit, times = times)
  
  # survie attendue à partir de la ratetable
  fit_exp <- survexp(
    formula = Surv(time, status) ~ 1,
    data = data %>% filter(sex_num == 2),
    ratetable = ratetable,
    rmap = list(age = age_days, sex = sex_num, year = year_frac)
  )
  
  s_exp <- summary(fit_exp, times = times)
  
  tibble(
    time = s_rel$time,
    surv_rel = s_rel$surv,
    lower = s_rel$lower,
    upper = s_rel$upper,
    surv_exp = s_exp$surv
  )
}



df_plot_female <- extract_surv_female(fit.pp.female, data, fr.ratetable_female)


p_female <- ggplot(df_plot_female, aes(x = time)) +
  geom_line(aes(y = surv_rel), linewidth = 1.2, color = "blue") +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, color = NA) +
  geom_line(aes(y = surv_exp), linetype = "dashed", linewidth = 1, color = "black") + # expected survival
  
  labs(
    x = "Temps (jours)",
    y = "Survie relative",
    title = "Survie relative (Pohar Perme) - Femmes",
    caption = "Courbe en pointillés : survie attendue"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.title = element_blank())

ggplotly(p_female, tooltip = c("time", "surv_rel", "lower", "upper", "surv_exp"))