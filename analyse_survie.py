import streamlit as st
import pandas as pd
import numpy as np
import pickle
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.statistics import multivariate_logrank_test
import plotly.graph_objs as go
import plotly.subplots as sp
import os

st.title("📈 Analyse de survie par sexe (Kaplan-Meier)")

# Charger les objets sauvegardés
if os.path.exists("hm.pkl") and os.path.exists("hm_libelle.pkl"):
    with open("hm.pkl", "rb") as f:
        hm = pickle.load(f)
    with open("hm_libelle.pkl", "rb") as f:
        hm_libelle = pickle.load(f)
else:
    st.error("Fichiers `hm.pkl` ou `hm_libelle.pkl` introuvables.")
    st.stop()

# Vérification des colonnes
for col in ['fup', 'event', 'sex']:
    if col not in hm.columns:
        st.error(f"Colonne manquante : {col}")
        st.stop()

st.success(f"Analyse de : **{hm_libelle}** ({len(hm)} cas)")

# Nettoyage des données
df = hm[['fup', 'event', 'sex']].dropna()
df['sex'] = df['sex'].astype(str)

# Kaplan-Meier par sexe
fig = go.Figure()
kmf = KaplanMeierFitter()
medians = {}
rmeans = {}
colors = {"0": "blue", "1": "red"}  # Ajuste selon les valeurs réelles

# Test log-rank
groups = df['sex'].unique()
if len(groups) != 2:
    st.error("Il faut exactement deux groupes pour le test log-rank.")
    st.stop()

# Séparer les deux groupes
group0 = df[df['sex'] == groups[0]]
group1 = df[df['sex'] == groups[1]]
result = logrank_test(group0['fup'], group1['fup'], group0['event'], group1['event'])
p_value = result.p_value

# Courbes de survie
for sex in groups:
    subset = df[df['sex'] == sex]
    kmf.fit(subset['fup'], subset['event'], label=f"Sexe {sex}")
    medians[sex] = kmf.median_survival_time_
    rmeans[sex] = kmf.conditional_time_to_event_.mean()
    
    fig.add_trace(go.Scatter(
        x=kmf.survival_function_.index,
        y=kmf.survival_function_[f"Sexe {sex}"],
        mode='lines',
        name=f"Sexe {sex}",
        line=dict(color=colors.get(sex, 'gray')),
    ))

    # IC (bande)
    fig.add_trace(go.Scatter(
        x=kmf.confidence_interval_.index,
        y=kmf.confidence_interval_[f"Sexe {sex}_upper_0.95"],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=kmf.confidence_interval_.index,
        y=kmf.confidence_interval_[f"Sexe {sex}_lower_0.95"],
        fill='tonexty',
        fillcolor='rgba(0,0,0,0.1)',
        mode='lines',
        line=dict(width=0),
        name=f"IC 95% {sex}"
    ))

# Annotations : médiane ou moyenne restreinte
annotations = []
for sex in groups:
    label = f"Sexe {sex}"
    if not np.isnan(medians[sex]):
        annotations.append(dict(
            x=medians[sex], y=0.1,
            text=f"Médiane : {round(medians[sex], 1)}",
            showarrow=True, arrowhead=1, ax=0, ay=-40,
            font=dict(color=colors.get(sex, "black"))
        ))
    else:
        annotations.append(dict(
            x=50, y=0.2 if sex == groups[0] else 0.1,
            text=f"Moyenne restreinte : {round(rmeans[sex], 1)}",
            showarrow=False,
            font=dict(color=colors.get(sex, "black"))
        ))

# Affichage
fig.update_layout(
    title=f"{hm_libelle} - Survie selon le sexe<br><sup>Test de log-rank p = {p_value:.3e}</sup>",
    xaxis_title="Temps (jours)",
    yaxis_title="Probabilité de survie",
    yaxis=dict(range=[0, 1]),
    annotations=annotations,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# Analyse Survie par sexe et groupe age

df2 = hm[['fup', 'event', 'sex', 'groupe_age']].dropna()
df2['sex'] = df2['sex'].astype(str)
df2['groupe_age'] = df2['groupe_age'].astype(str)

groupes_age = df2['groupe_age'].unique()
colors = {"0": "blue", "1": "red"}

# Création d'une figure avec un subplot par groupe d'âge (arrangé en 2 colonnes max)
cols = 2
rows = (len(groupes_age) + 1) // cols
fig2 = sp.make_subplots(rows=rows, cols=cols, subplot_titles=[f"Groupe d'âge: {ga}" for ga in groupes_age])

for i, ga in enumerate(groupes_age):
    subset = df2[df2['groupe_age'] == ga]
    if len(subset) == 0:
        continue

    kmf = KaplanMeierFitter()

    for sex in subset['sex'].unique():
        sex_subset = subset[subset['sex'] == sex]
        if len(sex_subset) == 0:
            continue

        kmf.fit(sex_subset['fup'], sex_subset['event'], label=f"Sexe {sex}")

        # Position subplot
        row = i // cols + 1
        col = i % cols + 1

        fig2.add_trace(
            go.Scatter(
                x=kmf.survival_function_.index,
                y=kmf.survival_function_[f"Sexe {sex}"],
                mode='lines',
                name=f"Sexe {sex}",
                line=dict(color=colors.get(sex, 'gray'))
            ),
            row=row, col=col
        )

        # IC
        fig2.add_trace(
            go.Scatter(
                x=kmf.confidence_interval_.index,
                y=kmf.confidence_interval_[f"Sexe {sex}_upper_0.95"],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ),
            row=row, col=col
        )
        fig2.add_trace(
            go.Scatter(
                x=kmf.confidence_interval_.index,
                y=kmf.confidence_interval_[f"Sexe {sex}_lower_0.95"],
                fill='tonexty',
                fillcolor='rgba(0,0,0,0.1)',
                mode='lines',
                line=dict(width=0),
                name=f"IC 95% {sex}",
                showlegend=False
            ),
            row=row, col=col
        )

fig2.update_layout(
    height=300 * rows, width=700,
    title_text=f"{hm_libelle} - Survie selon sexe ET groupe d'âge",
    template="plotly_white"
)

st.plotly_chart(fig2, use_container_width=True)


#  survie par sexe et groupe age calculé

bins = [-np.inf, 25, 50, 75, np.inf]
labels = ["0-25 ans", "26-50 ans", "51-75 ans", "75 ans+"]

hm3 = hm.copy()

hm3['grp_age'] = pd.cut(hm3['age'], bins=bins, labels=labels, right=False)

df3 = hm3[['fup', 'event', 'sex', 'grp_age']].dropna()
df3['sex'] = df3['sex'].astype(str)
df3['grp_age'] = df3['grp_age'].astype(str)


p_values = []

for grp in df3['grp_age'].unique():
    grp_data = df3[df3['grp_age'] == grp]
    sexes = grp_data['sex'].unique()
    if len(sexes) > 1:
        result = logrank_test(
            grp_data[grp_data['sex'] == sexes[0]]['fup'],
            grp_data[grp_data['sex'] == sexes[1]]['fup'],
            grp_data[grp_data['sex'] == sexes[0]]['event'],
            grp_data[grp_data['sex'] == sexes[1]]['event']
        )
        p_values.append((grp, result.p_value))
    else:
        p_values.append((grp, None))

# Affichage des p-values
st.write("### P-values log-rank par groupe d'âge")
pval_df = pd.DataFrame(p_values, columns=['Groupe d\'âge', 'P-value'])
st.table(pval_df)

groupes_age = df3['grp_age'].unique()
colors = {"0": "blue", "1": "red"}  # Ajuste si tes valeurs sex diffèrent
line_types = {"0": "solid", "1": "dash"}

cols = 2
rows = (len(groupes_age) + 1) // cols
fig3 = sp.make_subplots(rows=rows, cols=cols, subplot_titles=[f"Groupe d'âge: {ga}" for ga in groupes_age])

for i, ga in enumerate(groupes_age):
    subset = df3[df3['grp_age'] == ga]
    if len(subset) == 0:
        continue

    kmf = KaplanMeierFitter()

    row = i // cols + 1
    col = i % cols + 1

    for sex in subset['sex'].unique():
        sex_subset = subset[subset['sex'] == sex]
        if len(sex_subset) == 0:
            continue

        kmf.fit(sex_subset['fup'], sex_subset['event'], label=f"Sexe {sex}")

        fig3.add_trace(
            go.Scatter(
                x=kmf.survival_function_.index,
                y=kmf.survival_function_[f"Sexe {sex}"],
                mode='lines',
                name=f"Sexe {sex} - {ga}",
                line=dict(color=colors.get(sex, "gray"), dash=line_types.get(sex, "solid"))
            ),
            row=row, col=col
        )

        # IC
        fig3.add_trace(
            go.Scatter(
                x=kmf.confidence_interval_.index,
                y=kmf.confidence_interval_[f"Sexe {sex}_upper_0.95"],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ),
            row=row, col=col
        )
        fig3.add_trace(
            go.Scatter(
                x=kmf.confidence_interval_.index,
                y=kmf.confidence_interval_[f"Sexe {sex}_lower_0.95"],
                fill='tonexty',
                fillcolor='rgba(0,0,0,0.1)',
                mode='lines',
                line=dict(width=0),
                name=f"IC 95% {sex}",
                showlegend=False
            ),
            row=row, col=col
        )

fig3.update_layout(
    height=300 * rows, width=700,
    title_text=f"{hm_libelle} - Survie selon sexe ET groupe d'âge (tranches définies)",
    template="plotly_white"
)

st.plotly_chart(fig3, use_container_width=True)

# Calculer les quartiles de la colonne 'age'
hm4 = hm.copy()
quartiles_hm4 = hm4['age'].quantile([0, 0.25, 0.5, 0.75, 1.0]).values

# Créer les labels dynamiques pour les quartiles
labels = [
    f"Q1 - ≤ {round(quartiles_hm4[1])} ans",
    f"Q2 - {round(quartiles_hm4[1])} - {round(quartiles_hm4[2])} ans",
    f"Q3 - {round(quartiles_hm4[2])} - {round(quartiles_hm4[3])} ans",
    f"Q4 - > {round(quartiles_hm4[3])} ans"
]

# Créer la colonne 'grp_age' avec cut selon les quartiles

hm4['grp_age'] = pd.cut(
    hm4['age'],
    bins=quartiles_hm4,
    labels=labels,
    include_lowest=True,
    right=False
).astype(str)

# Préparer les données pour survie
df4 = hm4[['fup', 'event', 'sex', 'grp_age']].dropna()
df4['sex'] = df4['sex'].astype(str)

# Calculer p-values log-rank par groupe d'âge (quartile)
p_values_quartiles = []
for grp in df4['grp_age'].unique():
    grp_data = df4[df4['grp_age'] == grp]
    sexes = grp_data['sex'].unique()
    if len(sexes) > 1:
        result = logrank_test(
            grp_data[grp_data['sex'] == sexes[0]]['fup'],
            grp_data[grp_data['sex'] == sexes[1]]['fup'],
            grp_data[grp_data['sex'] == sexes[0]]['event'],
            grp_data[grp_data['sex'] == sexes[1]]['event']
        )
        p_values_quartiles.append((grp, result.p_value))
    else:
        p_values_quartiles.append((grp, None))

pval_quart_df = pd.DataFrame(p_values_quartiles, columns=["Groupe d'âge (Quartile)", "P-value"])
st.write("### P-values log-rank par groupe d'âge (Quartiles)")
st.table(pval_quart_df)

# Tracer les courbes Kaplan-Meier facettées sur les groupes d'âge (quartiles)
groupes_age = df4['grp_age'].unique()
colors = {"0": "blue", "1": "red"}  # Ajuster selon codage sex
line_types = {"0": "solid", "1": "dash"}

cols = 2
rows = (len(groupes_age) + 1) // cols
fig4 = sp.make_subplots(rows=rows, cols=cols, subplot_titles=[f"Groupe d'âge: {ga}" for ga in groupes_age])

kmf = KaplanMeierFitter()

for i, ga in enumerate(groupes_age):
    subset = df4[df4['grp_age'] == ga]
    if len(subset) == 0:
        continue

    row = i // cols + 1
    col = i % cols + 1

    for sex in subset['sex'].unique():
        sex_subset = subset[subset['sex'] == sex]
        if len(sex_subset) == 0:
            continue

        kmf.fit(sex_subset['fup'], sex_subset['event'], label=f"Sexe {sex}")

        fig4.add_trace(
            go.Scatter(
                x=kmf.survival_function_.index,
                y=kmf.survival_function_[f"Sexe {sex}"],
                mode='lines',
                name=f"Sexe {sex} - {ga}",
                line=dict(color=colors.get(sex, "gray"), dash=line_types.get(sex, "solid"))
            ),
            row=row, col=col
        )

        # IC
        fig4.add_trace(
            go.Scatter(
                x=kmf.confidence_interval_.index,
                y=kmf.confidence_interval_[f"Sexe {sex}_upper_0.95"],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ),
            row=row, col=col
        )
        fig4.add_trace(
            go.Scatter(
                x=kmf.confidence_interval_.index,
                y=kmf.confidence_interval_[f"Sexe {sex}_lower_0.95"],
                fill='tonexty',
                fillcolor='rgba(0,0,0,0.1)',
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ),
            row=row, col=col
        )

fig4.update_layout(
    height=300 * rows, width=700,
    title_text=f"{hm_libelle} - Survie selon sexe ET groupe d'âge par quartiles",
    template="plotly_white"
)

st.plotly_chart(fig4, use_container_width=True)

##### SELON SEX, GROUPE PAR ANNE DIAG (1997-2009 versus 2010-2022) ####

# Créer une colonne 'grp_annee_diag'
hm5 = hm.copy()
hm5['grp_annee_diag'] = np.where(hm5['annee_diag'] >= 2010, "2010-2022", "1997-2009")

# Préparer les données
df5 = hm5[['fup', 'event', 'sex', 'grp_annee_diag']].dropna()
df5['sex'] = df5['sex'].astype(str)
df5['grp_annee_diag'] = df5['grp_annee_diag'].astype(str)

# Calcul des p-values log-rank par groupe
from lifelines.statistics import logrank_test

p_values_diag = []
for grp in df5['grp_annee_diag'].unique():
    grp_data = df5[df5['grp_annee_diag'] == grp]
    sexes = grp_data['sex'].unique()
    if len(sexes) > 1:
        result = logrank_test(
            grp_data[grp_data['sex'] == sexes[0]]['fup'],
            grp_data[grp_data['sex'] == sexes[1]]['fup'],
            grp_data[grp_data['sex'] == sexes[0]]['event'],
            grp_data[grp_data['sex'] == sexes[1]]['event']
        )
        p_values_diag.append((grp, result.p_value))
    else:
        p_values_diag.append((grp, None))

pval_diag_df = pd.DataFrame(p_values_diag, columns=["Groupe d'année diagnostic", "P-value"])
st.write("### P-values log-rank selon le sexe par période de diagnostic")
st.table(pval_diag_df)

groupes_diag = df5['grp_annee_diag'].unique()
cols = 2
rows = (len(groupes_diag) + 1) // cols
fig5 = sp.make_subplots(rows=rows, cols=cols, subplot_titles=[f"Période: {g}" for g in groupes_diag])

kmf = KaplanMeierFitter()

for i, grp in enumerate(groupes_diag):
    subset = df5[df5['grp_annee_diag'] == grp]
    row = i // cols + 1
    col = i % cols + 1

    for sex in subset['sex'].unique():
        sex_subset = subset[subset['sex'] == sex]
        kmf.fit(sex_subset['fup'], sex_subset['event'], label=f"{'Homme' if sex == '1' else 'Femme'}")
        fig5.add_trace(
            go.Scatter(
                x=kmf.survival_function_.index,
                y=kmf.survival_function_[kmf._label],
                mode='lines',
                name=f"{kmf._label} - {grp}",
                line=dict(color="blue" if sex == "1" else "red",
                          dash="solid" if sex == "1" else "dash")
            ),
            row=row, col=col
        )
        # Ajouter les intervalles de confiance
        fig5.add_trace(
            go.Scatter(
                x=kmf.confidence_interval_.index,
                y=kmf.confidence_interval_[kmf._label + "_upper_0.95"],
                mode='lines', line=dict(width=0), showlegend=False
            ),
            row=row, col=col
        )
        fig5.add_trace(
            go.Scatter(
                x=kmf.confidence_interval_.index,
                y=kmf.confidence_interval_[kmf._label + "_lower_0.95"],
                fill='tonexty', fillcolor='rgba(0,0,0,0.1)',
                mode='lines', line=dict(width=0), showlegend=False
            ),
            row=row, col=col
        )

fig5.update_layout(
    height=300 * rows, width=800,
    title_text=f"{hm_libelle} - Survie selon sexe et période de diagnostic",
    template="plotly_white"
)

st.plotly_chart(fig5, use_container_width=True)

##### SELON ANNE DIAG (1997-2009 versus 2010-2022) PAR SEXE ####

# Créer une colonne 'grp_annee_diag' selon les années de diagnostic
hm6 = hm.copy()
hm6['grp_annee_diag'] = np.where(hm6['annee_diag'] >= 2010, "2010-2022", "1997-2009")

# Nettoyage et transformation
df6 = hm5[['fup', 'event', 'sex', 'grp_annee_diag']].dropna()
df6['sex'] = df6['sex'].astype(str)
df6['grp_annee_diag'] = df6['grp_annee_diag'].astype(str)

# P-values log-rank par sexe
from lifelines.statistics import logrank_test

p_values_diag_sex = []
for sexe in df6['sex'].unique():
    sexe_data = df6[df6['sex'] == sexe]
    groupes = sexe_data['grp_annee_diag'].unique()
    if len(groupes) > 1:
        result = logrank_test(
            sexe_data[sexe_data['grp_annee_diag'] == groupes[0]]['fup'],
            sexe_data[sexe_data['grp_annee_diag'] == groupes[1]]['fup'],
            sexe_data[sexe_data['grp_annee_diag'] == groupes[0]]['event'],
            sexe_data[sexe_data['grp_annee_diag'] == groupes[1]]['event']
        )
        p_values_diag_sex.append((sexe, result.p_value))
    else:
        p_values_diag_sex.append((sexe, None))

pval_diag_sex_df = pd.DataFrame(p_values_diag_sex, columns=["Sexe", "P-value"])
pval_diag_sex_df["Sexe"] = pval_diag_sex_df["Sexe"].replace({"1": "Homme", "2": "Femme"})

st.write("### P-values log-rank selon la période de diagnostic par sexe")
st.table(pval_diag_sex_df)

# Préparer le graphe avec facettes sur sexe
groupes_sexe = df6['sex'].unique()
rows = 1
cols = len(groupes_sexe)

fig6 = sp.make_subplots(rows=rows, cols=cols, subplot_titles=[f"Sexe : {'Homme' if g == '1' else 'Femme'}" for g in groupes_sexe])

kmf = KaplanMeierFitter()

for i, sexe in enumerate(groupes_sexe):
    subset = df6[df6['sex'] == sexe]
    row = 1
    col = i + 1

    for grp in subset['grp_annee_diag'].unique():
        subgrp = subset[subset['grp_annee_diag'] == grp]
        kmf.fit(subgrp['fup'], subgrp['event'], label=grp)
        fig6.add_trace(
            go.Scatter(
                x=kmf.survival_function_.index,
                y=kmf.survival_function_[grp],
                mode='lines',
                name=f"{grp} - {'Homme' if sexe == '1' else 'Femme'}",
                line=dict(
                    color="brown" if grp == "1997-2009" else "darkgreen",
                    dash="solid" if grp == "1997-2009" else "dash"
                )
            ),
            row=row, col=col
        )
        # IC
        fig6.add_trace(
            go.Scatter(
                x=kmf.confidence_interval_.index,
                y=kmf.confidence_interval_[grp + "_upper_0.95"],
                mode='lines', line=dict(width=0), showlegend=False
            ),
            row=row, col=col
        )
        fig6.add_trace(
            go.Scatter(
                x=kmf.confidence_interval_.index,
                y=kmf.confidence_interval_[grp + "_lower_0.95"],
                fill='tonexty', fillcolor='rgba(0,0,0,0.1)',
                mode='lines', line=dict(width=0), showlegend=False
            ),
            row=row, col=col
        )

fig6.update_layout(
    height=400, width=1000,
    title=f"{hm_libelle} – Survie selon la période de diagnostic\nFacetté par sexe",
    template="plotly_white"
)

st.plotly_chart(fig6, use_container_width=True)
