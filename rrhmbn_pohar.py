import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.title("Analyse de survie relative (Python)")

# --- Upload des fichiers ---
data_file = st.file_uploader("CSV des patients (hm.csv)", type="csv")
mlt_file = st.file_uploader("Table hommes (mltper_1x1.txt)", type="txt")
flt_file = st.file_uploader("Table femmes (fltper_1x1.txt)", type="txt")

if data_file and mlt_file and flt_file:

    # --- Chargement des données patients ---
    df = pd.read_csv(data_file)
    df['time'] = df['fup'] * 30.4375  # mois -> jours
    df['status'] = df['event']
    df['age_days'] = df['age'] * 365.24
    df['year_frac'] = pd.to_datetime(df['DateDuDiag']).dt.year + \
                      (pd.to_datetime(df['DateDuDiag']).dt.month - 0.5)/12

    # --- Chargement tables de mortalité robustes ---
    def read_lifetable(file_path):
        # détection automatique du séparateur
        lt = pd.read_csv(file_path, sep=None, engine='python', skiprows=2)
        # nettoyer les noms de colonnes
        lt.columns = [c.strip() for c in lt.columns]
        # renommer les colonnes principales
        rename_dict = {}
        for c in lt.columns:
            if c.lower().startswith('year'):
                rename_dict[c] = 'year'
            elif c.lower().startswith('age'):
                rename_dict[c] = 'age'
            elif c.lower().startswith('mx') or c.lower().startswith('rate'):
                rename_dict[c] = 'rate'
        lt = lt.rename(columns=rename_dict)
        return lt[['year','age','rate']]

    mlt = read_lifetable(mlt_file)
    flt = read_lifetable(flt_file)

    st.write("Colonnes mlt:", mlt.columns)
    st.write("Colonnes flt:", flt.columns)

    # --- Fonction survie attendue simple ---
    def expected_survival(age_days, year_frac, sex_num):
        surv = []
        for a, y, s in zip(age_days, year_frac, sex_num):
            age = int(a // 365.24)
            year = int(y)
            if s == 1:
                rates = mlt[(mlt['age']==age) & (mlt['year']==year)]['rate']
            else:
                rates = flt[(flt['age']==age) & (flt['year']==year)]['rate']
            r = rates.values[0] if len(rates) > 0 else 0
            surv.append(1-r)
        return np.array(surv)

    df['surv_exp'] = expected_survival(df['age_days'], df['year_frac'], df['sex'])

    # --- Fonction Kaplan-Meier ---
    def kaplan_meier(df_subset):
        df_sorted = df_subset.sort_values('time')
        times = np.unique(df_sorted['time'])
        surv = []
        cum_surv = 1.0
        for t in times:
            at_risk = df_sorted[df_sorted['time'] >= t]
            events = df_sorted[(df_sorted['time']==t) & (df_sorted['status']==1)]
            if len(at_risk) == 0:
                cum_surv = cum_surv
            else:
                cum_surv *= (1 - len(events)/len(at_risk))
            surv.append(cum_surv)
        return pd.DataFrame({'time': times, 'surv_rel': surv})

    # --- Extraire les courbes ---
    df_global = kaplan_meier(df)
    df_global['surv_exp'] = np.interp(df_global['time'], df['time'], df['surv_exp'])

    df_male = kaplan_meier(df[df['sex']==1])
    df_male['surv_exp'] = np.interp(df_male['time'], df['time'][df['sex']==1],
                                    df['surv_exp'][df['sex']==1])

    df_female = kaplan_meier(df[df['sex']==2])
    df_female['surv_exp'] = np.interp(df_female['time'], df['time'][df['sex']==2],
                                      df['surv_exp'][df['sex']==2])

    # --- Fonction Plotly ---
    def plot_surv(df_plot, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['surv_rel'],
                                 mode='lines', name='Survie relative', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['surv_exp'],
                                 mode='lines', name='Survie attendue', line=dict(color='black', width=2, dash='dash')))
        fig.update_layout(title=title,
                          xaxis_title="Temps (jours)",
                          yaxis_title="Survie",
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # --- Onglets Streamlit ---
    tab1, tab2, tab3 = st.tabs(["Global", "Hommes", "Femmes"])
    with tab1:
        plot_surv(df_global, "Survie relative - Global")
    with tab2:
        plot_surv(df_male, "Survie relative - Hommes")
    with tab3:
        plot_surv(df_female, "Survie relative - Femmes")
        