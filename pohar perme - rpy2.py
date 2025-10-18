import streamlit as st
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import pandas as pd
import plotly.graph_objects as go

# --- Activer la conversion pandas <-> R ---
pandas2ri.activate()

# --- Import des packages R ---
relsurv = importr('relsurv')
dplyr = importr('dplyr')
readr = importr('readr')
survival = importr('survival')
tibble = importr('tibble')

st.title("Analyse de survie relative (Pohar Perme)")

# --- Upload des fichiers ---
data_file = st.file_uploader("CSV des patients (hm.csv)", type="csv")
mlt_file = st.file_uploader("Table hommes (mltper_1x1.txt)", type="txt")
flt_file = st.file_uploader("Table femmes (fltper_1x1.txt)", type="txt")

if data_file and mlt_file and flt_file:
    # Charger les fichiers dans R
    robjects.globalenv['data'] = readr.read_csv(data_file)
    robjects.globalenv['mlt_file'] = mlt_file.name
    robjects.globalenv['flt_file'] = flt_file.name

    # --- Code R complet ---
    robjects.r('''
    library(relsurv)
    library(dplyr)
    library(readr)
    library(survival)
    library(tibble)

    data <- read_csv("''' + data_file.name + '''") %>%
      mutate(
        time = fup * 30.4375,
        status = event,
        age_days = age * 365.24,
        sex_num = sex,
        year_frac = as.numeric(format(DateDuDiag, "%Y")) +
          ((as.numeric(format(DateDuDiag, "%m"))-0.5)/12)
      )

    read_lifetable <- function(file_path){
      lt <- read.table(file_path, header=TRUE, sep="", stringsAsFactors=FALSE, skip=2)
      lt <- lt %>% rename(year = Year, age = Age, rate = mx)
      return(lt)
    }

    create_single_ratetable <- function(lt, sex = c("men","women")){
      sex <- match.arg(sex)
      ages <- sort(unique(lt$age))
      years <- sort(unique(lt$year))
      rate_matrix <- matrix(NA, nrow=length(ages), ncol=length(years), dimnames=list(ages,years))
      for(i in seq_along(ages)){
        for(j in seq_along(years)){
          val <- lt$rate[lt$age==ages[i] & lt$year==years[j]]
          rate_matrix[i,j] <- ifelse(length(val)==1,1-val,NA)
        }
      }
      empty_matrix <- matrix(NA, nrow=length(ages), ncol=length(years), dimnames=list(ages,years))
      ratetable <- transrate(
        men = if(sex=="men") rate_matrix else empty_matrix,
        women = if(sex=="women") rate_matrix else empty_matrix,
        yearlim = range(years),
        int.length = 1
      )
      return(ratetable)
    }

    mlt <- read_lifetable("''' + mlt_file.name + '''")
    flt <- read_lifetable("''' + flt_file.name + '''")
    fr.ratetable <- transrate(men=as.matrix(mlt$rate), women=as.matrix(flt$rate), yearlim=range(c(mlt$year, flt$year)), int.length=1)

    extract_surv <- function(fit, data, ratetable, step=1){
      times <- seq(0,max(data$time),by=step)
      s_rel <- summary(fit, times=times)
      fit_exp <- survexp(Surv(time,status) ~ 1, data=data, ratetable=ratetable,
                         rmap=list(age=age_days, sex=sex_num, year=year_frac))
      s_exp <- summary(fit_exp, times=times)
      tibble(time=s_rel$time, surv_rel=s_rel$surv, lower=s_rel$lower, upper=s_rel$upper, surv_exp=s_exp$surv)
    }

    # --- Ajustement global ---
    fit_pp <- rs.surv(Surv(time,status) ~ 1, data=data, ratetable=fr.ratetable,
                      rmap=list(age=age_days, sex=sex_num, year=year_frac), method="pohar-perme")
    df_global <- extract_surv(fit_pp, data, fr.ratetable)

    # --- Ajustement hommes ---
    fit_male <- rs.surv(Surv(time,status) ~ 1, data=data %>% filter(sex_num==1), ratetable=fr.ratetable,
                        rmap=list(age=age_days, sex=sex_num, year=year_frac), method="pohar-perme")
    df_male <- extract_surv(fit_male, data %>% filter(sex_num==1), fr.ratetable)

    # --- Ajustement femmes ---
    fit_female <- rs.surv(Surv(time,status) ~ 1, data=data %>% filter(sex_num==2), ratetable=fr.ratetable,
                          rmap=list(age=age_days, sex=sex_num, year=year_frac), method="pohar-perme")
    df_female <- extract_surv(fit_female, data %>% filter(sex_num==2), fr.ratetable)
    ''')

    # Récupérer les résultats en pandas
    df_global = pandas2ri.rpy2py(robjects.globalenv['df_global'])
    df_male = pandas2ri.rpy2py(robjects.globalenv['df_male'])
    df_female = pandas2ri.rpy2py(robjects.globalenv['df_female'])

    # --- Onglets Streamlit ---
    tab1, tab2, tab3 = st.tabs(["Global", "Hommes", "Femmes"])

    def plot_surv(df, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['time'], y=df['surv_rel'], mode='lines', name='Survie relative', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=df['time'], y=df['surv_exp'], mode='lines', name='Survie attendue', line=dict(color='black', width=2, dash='dash')))
        fig.add_trace(go.Scatter(x=df['time'], y=df['lower'], mode='lines', name='IC bas', line=dict(color='blue', width=1), opacity=0.2))
        fig.add_trace(go.Scatter(x=df['time'], y=df['upper'], mode='lines', name='IC haut', line=dict(color='blue', width=1), opacity=0.2, fill='tonexty'))
        fig.update_layout(title=title, xaxis_title="Temps (jours)", yaxis_title="Survie", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with tab1:
        plot_surv(df_global, "Survie relative - Global")
    with tab2:
        plot_surv(df_male, "Survie relative - Hommes")
    with tab3:
        plot_surv(df_female, "Survie relative - Femmes")
