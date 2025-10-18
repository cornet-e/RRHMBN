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
        lt = pd.read_csv(file_path, sep=None, engine='python', skiprows=2)
        lt.columns = [c.strip() for c in lt.columns]
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

    # --- Kaplan-Meier avec IC de Greenwood ---
    def kaplan_meier_ic(df_subset, alpha=0.05):
        df_sorted = df_subset.sort_values('time')
        times = np.unique(df_sorted['time'])
        surv = []
        var = []
        cum_surv = 1.0
        cum_var = 0.0
        for t in times:
            at_risk = df_sorted[df_sorted['time'] >= t]
            events = df_sorted[(df_sorted['time']==t) & (df_sorted['status']==1)]
            d = len(events)
            n = len(at_risk)
            if n == 0:
                cum_surv = cum_surv
                cum_var = cum_var
            else:
                cum_surv *= (1 - d/n)
                if n-d > 0:
                    cum_var += d / (n*(n-d))
            surv.append(cum_surv)
            se = np.sqrt(cum_var) * cum_surv
            var.append(se)
        times = np.array(times)
        surv = np.array(surv)
        se = np.array(var)
        z = 1.96
        lower = np.maximum(0, surv - z*se)
        upper = np.minimum(1, surv + z*se)
        return pd.DataFrame({'time': times, 'surv_rel': surv, 'lower': lower, 'upper': upper})

    # --- Survie attendue cumulative ---
    def expected_survival_curve(df_subset, mlt, flt):
        df_sorted = df_subset.sort_values('time')
        times = np.unique(df_sorted['time'])
        surv_exp = []

        for t in times:
            at_risk = df_sorted[df_sorted['time'] >= t]
            cum_surv = 1.0
            for _, row in at_risk.iterrows():
                age = int(row['age_days'] // 365.24)
                year = int(row['year_frac'])
                sex = row['sex']
                if sex == 1:
                    rates = mlt[(mlt['age']==age) & (mlt['year']==year)]['rate']
                else:
                    rates = flt[(flt['age']==age) & (flt['year']==year)]['rate']
                r = rates.values[0] if len(rates) > 0 else 0
                cum_surv *= (1 - r)
            surv_exp.append(cum_surv)

        return pd.DataFrame({'time': times, 'surv_exp': surv_exp})

    # --- Calcul des courbes ---
    df_global = kaplan_meier_ic(df)
    df_global_exp = expected_survival_curve(df, mlt, flt)
    df_global = df_global.merge(df_global_exp, on='time', how='left')

    df_male = kaplan_meier_ic(df[df['sex']==1])
    df_male_exp = expected_survival_curve(df[df['sex']==1], mlt, flt)
    df_male = df_male.merge(df_male_exp, on='time', how='left')

    df_female = kaplan_meier_ic(df[df['sex']==2])
    df_female_exp = expected_survival_curve(df[df['sex']==2], mlt, flt)
    df_female = df_female.merge(df_female_exp, on='time', how='left')

    # --- Plotly avec IC et survie attendue ---
    def plot_surv(df_plot, title):
        fig = go.Figure()
        # survie relative
        fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['surv_rel'],
                                 mode='lines', name='Survie relative', line=dict(color='blue', width=2)))
        # IC
        fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['upper'],
                                 fill=None, mode='lines', line=dict(color='blue', width=1), showlegend=False))
        fig.add_trace(go.Scatter(x=df_plot['time'], y=df_plot['lower'],
                                 fill='tonexty', mode='lines', line=dict(color='blue', width=1), name='IC 95%'))
        # survie attendue
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
