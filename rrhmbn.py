import streamlit as st
import pandas as pd
import pickle
import os
import io
import numpy as np
import pickle
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.statistics import multivariate_logrank_test
import plotly.graph_objs as go
import plotly.subplots as sp
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


st.set_page_config(page_title="RRHMBN - Import des données", layout="wide")
st.title("📊 Analyse RRHMBN - Import des données")

# Choix de la source de données
source_choice = st.radio(
    "📁 Sélectionnez la source des données :",
    ("Importer un fichier Excel", "Utiliser le fichier par défaut"),
    index=1
)

if source_choice == "Importer un fichier Excel":
    file = st.file_uploader("Charger le fichier Excel `.xlsx`", type=["xlsx"])
elif source_choice == "Utiliser le fichier par défaut":
    default_path = "rrhmbn.xlsx"  # <-- à adapter selon l’arborescence
    if os.path.exists(default_path):
        file = default_path
    else:
        st.error(f"❌ Le fichier par défaut n'a pas été trouvé : `{default_path}`")
        file = None
else:
    file = None

# Lecture du fichier
if file is not None:
    try:
        if isinstance(file, str):  # Fichier local
            rrhmbn_brut = pd.read_excel(file, engine="openpyxl")
        else:  # Fichier importé (Streamlit uploader)
            rrhmbn_brut = pd.read_excel(file, engine="openpyxl")

        ### Ajout colonnes 
        
        # --- Paramètres des chemins des fichiers ---
        fichier_select = "Import-select.xlsx"

        # --- Lecture des fichiers avec ligne d'en-tête ---
        #df_brut = pd.read_excel(fichier_brut, header=0)
        df_select = pd.read_excel(fichier_select, header=0)

        # --- Colonnes à extraire : on prend les noms de colonnes du fichier Import-select ---
        colonnes_requises = df_select.columns.str.strip().tolist()

        # --- Vérification des colonnes réellement présentes dans le fichier brut ---
        colonnes_presentes = [col for col in colonnes_requises if col in rrhmbn_brut.columns]
        colonnes_absentes = [col for col in colonnes_requises if col not in rrhmbn_brut.columns]

        # --- Affichage console pour information ---
        print("✅ Colonnes trouvées dans le fichier brut :")
        print(colonnes_presentes)

        if colonnes_absentes:
            print("\n⚠️ Colonnes absentes dans le fichier brut :")
            print(colonnes_absentes)

        # --- Extraction des colonnes présentes ---
        df_filtre = rrhmbn_brut[colonnes_presentes].copy()  # ✅ Copie sécurisée

        # Conversion explicite en datetime
        df_filtre["DateDuDiag"] = pd.to_datetime(df_filtre["DateDuDiag"], errors="coerce")
        df_filtre["DateDernieresNouvelles"] = pd.to_datetime(df_filtre["DateDernieresNouvelles"], errors="coerce")

        # Créer la colonne "annee_diag"
        df_filtre["annee_diag"] = df_filtre["DateDuDiag"].dt.year.astype("Int64")

        # Calcul de la durée de suivi en mois
        df_filtre["Follow_up_months"] = (
            (df_filtre["DateDernieresNouvelles"] - df_filtre["DateDuDiag"]).dt.days / 30.44
        ).round(1)

        # --- ajout colonne "Cas_valides" ---

        # Normaliser les colonnes en minuscules chaînes de caractères (utile si valeurs booléennes ou mixtes)
        colonnes_a_verifier = ["Exclusion", "A_Surveiller", "Pre_Saisie"]
        for col in colonnes_a_verifier:
            df_filtre[col] = df_filtre[col].astype(str).str.upper().str.strip()

        # Créer la colonne Cas_valides : 1 si toutes les 3 colonnes sont "FALSE", sinon 0 / pour année 1997 min et 2022 max
        df_filtre["Cas_valides"] = (
            (df_filtre["Exclusion"] == "FALSE") &
            (df_filtre["A_Surveiller"] == "FALSE") &
            (df_filtre["Pre_Saisie"] == "FALSE") &
            (df_filtre["annee_diag"] <= 2023) &
            (df_filtre["annee_diag"] >= 1997)
        ).astype(int)

        # Définir les bornes et les étiquettes personnalisées
        bornes = [0, 5, 10, 15, 25, 30, 35, 40, 45, 50, 55, 60,
                65, 70, 75, 80, 85, 90, 95, float('inf')]
        etiquettes = [
            "[00;05[", "[05;10[", "[10;15[", "[15;25[", "[25;30[", "[30;35[",
            "[35;40[", "[40;45[", "[45;50[", "[50;55[", "[55;60[", "[60;65[",
            "[65;70[", "[70;75[", "[75;80[", "[80;85[", "[85;90[", "[90;95[", "95+"
        ]

        # Créer la colonne "Groupe_age"
        df_filtre["groupe_age"] = pd.cut(
            df_filtre["Age_Au_Diag"],
            bins=bornes,
            labels=etiquettes,
            right=False  # pour avoir des intervalles du type [x;y[
        )

        

        # Charger le fichier ICDO
        df_icdo = pd.read_excel("ICDO.xlsx")

        # Nettoyer la clé de jointure dans les deux fichiers
        df_filtre["Code_MORPHO_Diag"] = df_filtre["Code_MORPHO_Diag"].apply(lambda x: str(int(x)).strip() if pd.notnull(x) else "")
        df_icdo["ICD-O-3"] = df_icdo["ICD-O-3"].astype(str).str.strip()

        # Merge sur le code morpho
        df_final = df_filtre.merge(
            df_icdo[[
                "ICD-O-3",
                "patho_sous_type", "patho_sous_type_label",
                "patho_groupe", "patho_groupe_label",
                "patho_sous_type_XT", "patho_sous_type_XT_label"
            ]],
            left_on="Code_MORPHO_Diag",
            right_on="ICD-O-3",
            how="left"
        )

        # supprimer la colonne 'ICD-O-3' (doublon)
        df_final = df_final.drop(columns=["ICD-O-3"])

        # Charger le fichier correspondance ICDO-ICD10-ICD11
        df_icdo2 = pd.read_excel("ICD-O_ICD11_ICD10_correspondance.xlsx")
        
        # Transformation du champ ICD-O code
        df_icdo2["ICD-O code"] = (
        df_icdo2["ICD-O code"]
        .astype(str)        # conversion en texte si ce n’est pas déjà le cas
        .str.replace("/", "", regex=False)  # suppression du "/"
        )

        # Merge sur le code ICD-O code
        df_final2 = df_final.merge(
            df_icdo2[[
                "ICD-O code",
                "ICD-11 code 1", "ICD-11 label 1",
                "ICD-11 code 2", "ICD-11 label 2",
                "icd10Code", "icd10Title"
            ]],
            left_on="Code_MORPHO_Diag",
            right_on="ICD-O code",
            how="left"
        )



        # --- Sauvegarde du fichier filtré ---
        # fichier_sortie = "rrhmbn.xlsx"
        # df_final.to_excel(fichier_sortie, index=False)
        #print(f"\n💾 Fichier sauvegardé : {fichier_sortie}")

        rrhmbn = df_final

        # 🔄 Renommer les colonnes critiques par leur nom actuel
        mapping = {
            "_SexePatient": "sex",
            "Age_Au_Diag": "age",
            "Follow_up_months": "fup",
            "EtatVital": "event"
        }

        # Appliquer le renommage uniquement si les colonnes existent
        rrhmbn.rename(columns={k: v for k, v in mapping.items() if k in rrhmbn.columns}, inplace=True)

        # Corriger les types
        for col in rrhmbn.columns:
            if rrhmbn[col].dtype == object:
                rrhmbn[col] = rrhmbn[col].astype(str)

        # Sélectionner les cas valides
        if "Cas_valides" in rrhmbn.columns:
            rrhmbn_valide = rrhmbn[rrhmbn["Cas_valides"] == 1]

            st.success(f"✅ {len(rrhmbn_valide)} cas valides chargés.")
            st.dataframe(rrhmbn_valide)
        else:
            st.warning("⚠️ La colonne 'Cas_valides' n'existe pas dans le fichier.")

    except Exception as e:
        st.error(f"❌ Une erreur est survenue lors de la lecture du fichier :\n\n{e}")
else:
    st.info("⏳ En attente de données...")

st.subheader("📅 Nombre de cas incidents par année")
incid = rrhmbn_valide["annee_diag"].value_counts().sort_index()
df_incid = incid.reset_index()
df_incid.columns = ["Année", "Nombre de cas"]
df_incid["Année"] = df_incid["Année"].astype(str)

# Affichage dans Streamlit
st.line_chart(data=df_incid, x="Année", y="Nombre de cas")

st.header("📊 Statistiques descriptives du registre")

# Copier le DataFrame filtré si tu veux appliquer les filtres précédents
df_tab = rrhmbn_valide.copy()

global_annees_min_max = st.slider("Choisir l'intervalle des années", min_value=1994, max_value=2025, value=(1997, 2023), key="slider_annees_1")

# Filtrage par année
df_tab = df_tab[
    (df_tab["annee_diag"] >= global_annees_min_max[0]) &
    (df_tab["annee_diag"] <= global_annees_min_max[1])
]
# Calculer les effectifs par patho_groupe_label, patho_sous_type_label et sex
tableau = df_tab.groupby(['patho_groupe_label', 'patho_sous_type_label', 'sex']).size().unstack(fill_value=0)
tableau = tableau.rename(columns={1: "Hommes", 2: "Femmes"})
tableau['Total'] = tableau['Hommes'] + tableau['Femmes']
tableau = tableau.reset_index()

# Trier par groupe et pathologie
tableau = tableau.sort_values(['patho_groupe_label', 'patho_sous_type_label'])

# Supprimer la répétition des groupes pour l'affichage
prev_group = None
patho_groupe_col = []
for g in tableau['patho_groupe_label']:
    if g == prev_group:
        patho_groupe_col.append("")
    else:
        patho_groupe_col.append(g)
        prev_group = g
tableau['Groupe'] = patho_groupe_col
tableau = tableau[['Groupe', 'patho_sous_type_label', 'Hommes', 'Femmes', 'Total']]

# Créer un dictionnaire pour assigner une couleur à chaque groupe
group_colors = {}
unique_groups = tableau['Groupe'].replace("", pd.NA).dropna().unique()
colors = ["#f5f5f5", "#e0e0e0"]  # alternance clair / foncé
for i, g in enumerate(unique_groups):
    group_colors[g] = colors[i % len(colors)]

# Fonction pour colorer les lignes selon le groupe
def color_rows(row):
    # Récupérer la couleur correspondant au groupe (si vide, chercher le précédent non vide)
    grp = row['Groupe']
    if grp == "":
        # Chercher la couleur du dernier groupe non vide
        grp_idx = tableau.index.get_loc(row.name)
        for j in range(grp_idx-1, -1, -1):
            if tableau.at[j, 'Groupe'] != "":
                grp = tableau.at[j, 'Groupe']
                break
    return ['background-color: {}'.format(group_colors.get(grp, 'white'))]*len(row)

# Styliser le tableau
styled_table = tableau.style.apply(color_rows, axis=1)\
    .set_properties(**{'text-align': 'center'})\
    .set_table_styles([{'selector':'th', 'props':[('text-align','center'), ('background-color','#d0d0d0')]}])

st.subheader("📊 Nombre de cas par pathologie et sexe (groupé par patho_groupe_label, nuances de gris)")
st.dataframe(styled_table, height=600)

st.subheader("🦠 Nombre de cas incidents par pathologie (Top 20)")
# === Cas ===
global_annees_min_max_2 = st.slider("Choisir l'intervalle des années", min_value=1994, max_value=2025, value=(1997, 2022), key="slider_annees_2")


# === Sélection du type ICD-O ===
icdo_choice = st.radio(
    "Filtrer selon le code morphologique (ICD-O)",
    options=["ICDO /1", "ICDO /3", "ICDO /1 et /3"],
    horizontal=True,
    index=2
)

df_cas_global = rrhmbn_valide.copy()

# Filtrage par année
df_cas_global = df_cas_global[
    (df_cas_global["annee_diag"] >= global_annees_min_max_2[0]) &
    (df_cas_global["annee_diag"] <= global_annees_min_max_2[1])
]

# Filtrage par code ICD-O
if icdo_choice == "ICDO /1":
    df_cas_global = df_cas_global[df_cas_global["Code_MORPHO_Diag"].astype(str).str.endswith("1")]
elif icdo_choice == "ICDO /3":
    df_cas_global = df_cas_global[df_cas_global["Code_MORPHO_Diag"].astype(str).str.endswith("3")]
# Si "ICDO /1 et /3", on garde tout


# === Filtre sur les pathologies ===
toutes_pathos = df_cas_global["patho_sous_type_label"].unique()
selected_pathos = st.multiselect(
    "Sélectionner les pathologies",
    options=sorted(toutes_pathos),
    default=sorted(toutes_pathos)  # toutes sélectionnées par défaut
)
# Appliquer le filtre
df_cas_global = df_cas_global[df_cas_global["patho_sous_type_label"].isin(selected_pathos)]

# Calcul du nombre total de cas
nb_total = len(df_cas_global)

st.markdown(f"**Nombre total de cas valides :** {nb_total:,}".replace(",", " "))

# 2. Nombre de cas par pathologie

# Récupérer le top 10
top_pathos = df_cas_global["patho_sous_type_label"].value_counts().head(20)
# Remettre sous forme de DataFrame, et trier pour affichage dans l'ordre décroissant
df_top_pathos = top_pathos.reset_index()
df_top_pathos.columns = ["Pathologie", "Nombre de cas"]
df_top_pathos = df_top_pathos.sort_values("Nombre de cas", ascending=False)
# st.bar_chart(top_pathos)

fig = px.bar(df_top_pathos, x="Pathologie", y="Nombre de cas",
             title="Top 20 des pathologies",
             text="Nombre de cas",
             color="Nombre de cas",
             color_continuous_scale="Reds")

fig.update_layout(xaxis={'categoryorder':'total descending'})
st.plotly_chart(fig)

# 3. Statistiques sur l'âge
#st.subheader("🎂 Statistiques d'âge")
#med_age = rrhmbn_valide["age"].median()
#q1, q3 = rrhmbn_valide["age"].quantile([0.25, 0.75])
#st.markdown(f"- Médiane : **{med_age:.1f}** ans")
#st.markdown(f"- 1er quartile : **{q1:.1f}** ans")
#st.markdown(f"- 3e quartile : **{q3:.1f}** ans")

# Boxplot de l'âge
#fig, ax = plt.subplots()
#sns.boxplot(x=rrhmbn_valide["age"], ax=ax, color="skyblue")
#ax.set_title("Répartition de l'âge")
#st.pyplot(fig)

# 4. Sex ratio
#st.subheader("🚻 Répartition par sexe")
#nb_h = (rrhmbn_valide["sex"] == 1).sum()
#nb_f = (rrhmbn_valide["sex"] == 2).sum()
#sex_ratio = nb_h / nb_f if nb_f > 0 else None
#st.markdown(f"- Hommes : **{nb_h}**")
#st.markdown(f"- Femmes : **{nb_f}**")
#st.markdown(f"- Sex ratio (H/F) : **{sex_ratio:.2f}**" if sex_ratio else "- Sex ratio : Non calculable")

# Pie chart
#sex_counts = rrhmbn_valide["sex"].map({1: "Homme", 2: "Femme"}).value_counts()
#fig, ax = plt.subplots()
#ax.pie(sex_counts, labels=sex_counts.index, autopct="%1.1f%%", startangle=90, colors=["lightblue", "lightcoral"])
#ax.axis("equal")
#st.pyplot(fig)

#  Incidence par année
incid = df_cas_global["annee_diag"].value_counts().sort_index()
df_incid = incid.reset_index()
df_incid.columns = ["Année", "Nombre de cas"]
df_incid["Année"] = df_incid["Année"].astype(str)

# Affichage dans Streamlit
st.line_chart(data=df_incid, x="Année", y="Nombre de cas")



st.markdown("---")
st.header("🧬 Sélection d’un sous-ensemble de pathologies (HM)")

hm, hm_libelle = None, None  # Initialisation pour éviter erreurs

if 'rrhmbn_valide' in locals():

    # 1. Ajout de l'option "Toutes les données"
    choix0 = st.radio(
        "Choix de classification :", 
        ["Toutes les données", "REPIH", "XT", "ICDO", "ADICAP"], 
        index=None
    )

    # Initialisation par défaut
    hm = rrhmbn_valide.copy()
    hm_libelle = "Ensemble des données"

    # 2. Gestion du cas "Toutes les données"
    if choix0 == "Toutes les données":
        hm = rrhmbn_valide.copy()
        hm_libelle = "Toutes les données"
        st.info(f"Vous avez sélectionné l'intégralité des données ({len(hm)} lignes).")

    elif choix0 == "REPIH":
        choix = st.radio("Type de sélection :", ["Groupe Patho", "Sous-Type Patho"])

        if choix == "Groupe Patho":
            libelles = rrhmbn_valide["patho_groupe_label"].dropna().unique()
            libelle_sel = st.selectbox("Sélectionnez un libellé de groupe patho :", ["Sélectionner un libellé"] + sorted(libelles))
            
            if libelle_sel != "Sélectionner un libellé":
                hm = rrhmbn_valide[rrhmbn_valide["patho_groupe_label"] == libelle_sel]
                hm_libelle = libelle_sel
            else:
                hm = pd.DataFrame() # Vide tant que rien n'est sélectionné

        elif choix == "Sous-Type Patho":
            sous_type_labels = rrhmbn_valide["patho_sous_type_label"].dropna().unique()
            sous_type_label_sel = st.selectbox("Sélectionnez un libellé de sous-type patho :", ["Sélectionner un libellé"] + sorted(sous_type_labels))

            if sous_type_label_sel != "Sélectionner un libellé":
                hm = rrhmbn_valide[rrhmbn_valide["patho_sous_type_label"] == sous_type_label_sel]
                hm_libelle = sous_type_label_sel
            else:
                hm = pd.DataFrame()

    elif choix0 == "XT":
        xt_labels = rrhmbn_valide["patho_sous_type_XT_label"].dropna().unique()
        xt_label_sel = st.selectbox("Sélectionnez un sous-type XT patho :", ["Sélectionner un libellé"] + sorted(xt_labels))
        
        if xt_label_sel != "Sélectionner un libellé":
            hm = rrhmbn_valide[rrhmbn_valide["patho_sous_type_XT_label"] == xt_label_sel]
            hm_libelle = xt_label_sel
        else:
            hm = pd.DataFrame()
    
    elif choix0 == "ICDO":
        icdo = rrhmbn_valide["Code_MORPHO_Diag"].dropna().unique()
        icdo_sel = st.multiselect("Sélectionnez un ou plusieurs codes ICDO :", sorted(icdo))
    
        if icdo_sel:
            hm = rrhmbn_valide[rrhmbn_valide["Code_MORPHO_Diag"].isin(icdo_sel)]
            hm_libelle = ", ".join(icdo_sel)
        else:
            hm = rrhmbn_valide.copy()
            hm_libelle = "Tous les codes ICDO"

    elif choix0 == "ADICAP":
        adicap = rrhmbn_valide["Code_ADICAP_Diag"].dropna().unique()
        adicap_sel = st.multiselect("Sélectionnez un ou plusieurs codes ADICAP :", sorted(adicap))
    
        if adicap_sel:
            hm = rrhmbn_valide[rrhmbn_valide["Code_ADICAP_Diag"].isin(adicap_sel)]
            hm_libelle = ", ".join(adicap_sel)
        else:
            hm = rrhmbn_valide.copy()
            hm_libelle = "Tous les codes ADICAP"

# affichage des correspondances de classification

st.subheader("Affichage d'un tableau de correspondance des classifications ICD-O / ICD-11 /  ICD-10")

if st.button("Afficher le tableau de correspondance"):
    df_correspondance = pd.read_excel("ICD-O_ICD11_ICD10_correspondance.xlsx")
    with st.expander("📊 Tableau de correspondance", expanded=True):
        st.dataframe(df_correspondance)


# Affichage et sauvegarde
if hm_libelle and 'hm' in locals() and not hm.empty:
    st.success(f"✅ Vous avez sélectionné : **{hm_libelle}** ({len(hm)} cas)")
    st.dataframe(hm)

    # Option pour sauvegarder
    if st.button("💾 Sauvegarder les objets `hm` et `hm_libelle`"):
        with open("hm.pkl", "wb") as f:
            pickle.dump(hm, f)
        with open("hm_libelle.pkl", "wb") as f:
            pickle.dump(hm_libelle, f)
        st.success("Objets sauvegardés avec succès (hm.pkl et hm_libelle.pkl)")
    

    st.title("Statistiques descriptives")

    ### Tableau des effectifs ###
    st.success(f"Analyse de : **{hm_libelle}** ({len(hm)} cas)")

    # Colonnes utiles
    hm["Calvados"] = hm["Code_INSEE_AuDiag"].str.startswith("14").astype(int)
    hm["Orne"] = hm["Code_INSEE_AuDiag"].str.startswith("61").astype(int)
    hm["Manche"] = hm["Code_INSEE_AuDiag"].str.startswith("50").astype(int)

    # Regrouper par année
    tableau = hm.groupby("annee_diag").agg(
        nb_hommes=("sex", lambda x: (x == 1).sum()),
        nb_femmes=("sex", lambda x: (x == 2).sum()),
        cas_calvados=("Calvados", "sum"),
        cas_orne=("Orne", "sum"),
        cas_manche=("Manche", "sum"),
    )

    # Ajouter total ligne
    tableau["total"] = tableau.sum(axis=1)

    # Calcul du % que représente chaque année sur le total global
    total_global = tableau["total"].sum()
    tableau["total (%)"] = (tableau["total"] / total_global * 100).round(1)

    # Fusionner uniquement cette colonne corrigée avec le tableau existant
    #tableau_final.update(tableau["total (%)"])

    st.subheader(f"📊 Tableau double entrée par année de diagnostic - {hm_libelle}")
    st.dataframe(tableau)

    ### Pyramide des âges ####
    st.success(f"Analyse de : **{hm_libelle}** ({len(hm)} cas)")

    # Définir les classes d'âge
    bornes = list(range(0, 100, 5)) + [120]
    labels = [f"{i:02d}-{i+4:02d}" for i in range(0, 95, 5)] + ["95+"]

    # Créer les classes d'âge
    hm["classe_age"] = pd.cut(hm["age"], bins=bornes, labels=labels, right=False)

    # Compter les effectifs par classe d'âge et sexe
    pyramide = hm.groupby(["classe_age", "sex"]).size().unstack(fill_value=0)

    # Mettre les hommes en négatif
    pyramide[1] = -pyramide.get(1, 0)  # Hommes
    pyramide[2] = pyramide.get(2, 0)   # Femmes

    # Tracer la pyramide
    fig, ax = plt.subplots(figsize=(8, 6))
    pyramide[[1, 2]].plot(kind='barh', ax=ax, color=["skyblue", "lightcoral"])
    ax.set_yticklabels(pyramide.index)
    ax.set_xlabel("Effectifs")
    ax.set_title("Pyramide des âges au diagnostic")
    ax.legend(["Hommes", "Femmes"])
    ax.axvline(0, color='black', linewidth=0.8)
    plt.tight_layout()

    # Affichage dans Streamlit
    st.subheader(f"📊 Pyramide des âges au diagnostic - {hm_libelle}")
    st.pyplot(fig)


    ### Statistiques descriptives ###
    
    # Fonction de résumé statistique
    def stats_age_par_sexe(df, age_col="age", sexe_col="sex"):
        stats = df.groupby(sexe_col)[age_col].agg(
            N="count",
            Moyenne="mean",
            Médiane="median",
            Q1=lambda x: x.quantile(0.25),
            Q3=lambda x: x.quantile(0.75),
            Min="min",
            Max="max"
        ).round(1)
        
        # Mapping des sexes si codés 1 = Homme, 2 = Femme
        stats.index = stats.index.map({1: "Hommes", 2: "Femmes"})
        
         # Calcul du sex-ratio (H/F)
        nb_h = stats.loc["Hommes", "N"] if "Hommes" in stats.index else 0
        nb_f = stats.loc["Femmes", "N"] if "Femmes" in stats.index else 0
        sex_ratio = round(nb_h / nb_f, 2) if nb_f > 0 else np.nan
        # Ajouter une ligne "Sex-ratio H/F"
        stats.loc["Sex-ratio H/F"] = [sex_ratio, "", "", "", "", "", ""]
        
        return stats.reset_index(names="Sexe")

    # Appliquer la fonction au sous-ensemble sélectionné
    st.subheader(f"📈 Statistiques descriptives de l'âge au diagnostic - {hm_libelle}")
    stats_age = stats_age_par_sexe(hm)
    st.dataframe(stats_age)

    #### INCIDENCE ####
    st.title(f"📈 Taux d'incidence (pour 100 000 habitants) - {hm_libelle}")

    ### --- 1. Chargement et préparation des données ---
    # === Cas ===
    annees_min_max = st.slider("Choisir l'intervalle des années", min_value=1994, max_value=2025, value=(1997, 2022))
    df_cas = hm.copy()
    df_cas = df_cas[
        (df_cas["annee_diag"] >= annees_min_max[0]) &
        (df_cas["annee_diag"] <= annees_min_max[1])
    ]

    # Harmonisation du sexe
    df_cas["sexe"] = df_cas["sex"].map({1: "Homme", 2: "Femme"})

    # Attribution du département à partir du code INSEE
    df_cas["Code_INSEE_AuDiag"] = df_cas["Code_INSEE_AuDiag"].astype(str)
    df_cas["code_dep"] = df_cas["Code_INSEE_AuDiag"].str[:2]
    df_cas["departement"] = df_cas["code_dep"].map({
        "14": "Calvados",
        "50": "Manche",
        "61": "Orne"
    })
    df_cas = df_cas[df_cas["departement"].notna()]

    # Ajout des tranches d'âge
    def get_age_tranche(age):
        if pd.isna(age):
            return np.nan
        tranche = int(age // 5 * 5)
        return "95+" if tranche >= 95 else f"{tranche}-{tranche+4}"

    df_cas["age_tranche"] = df_cas["age"].apply(get_age_tranche)

    # === Population ===
    df_pop = pd.read_excel("populations.XLSX")  # colonnes : annee, zone, sexe, 0-4, 5-9, ..., 85+
    # Supprimer la première colonne (par position)
    df_pop = df_pop.iloc[:, 1:]

    #st.dataframe(df_pop)

    # Ne garder que les sexes homme/femme
    df_pop = df_pop[df_pop["Sexe"].isin(["Hommes", "Femmes"])]
    df_pop["Sexe"] = df_pop["Sexe"].replace({"Hommes": "Homme", "Femmes": "Femme"})

    # 1. Renommer les colonnes clés
    df_pop.rename(columns={
        'Année': 'annee',
        'Zone': 'zone',
        'Sexe': 'sexe'
    }, inplace=True)

    # 2. Supprimer les colonnes inutiles
    df_pop.drop(columns=['Total', 'Unnamed: 25'], errors='ignore', inplace=True)

    # 3. Nettoyer les noms de colonnes d'âges : enlever " ans", remplacer espaces par '_', etc.
    def clean_age_label(col):
        col = col.strip()  # enlève espaces en début/fin
        if 'ans et plus' in col:
            # ex : "95 ans et plus" -> "95+"
            return col.split(' ')[0] + '+'
        else:
            # ex : "50 - 54 ans" -> "50-54"
            parts = col.split(' ')
            # on prend le 1er et 3e élément (ex: ["50", "-", "54", "ans"])
            if len(parts) >= 3:
                return parts[0] + '-' + parts[2]
            else:
                return col  # au cas où le format diffère


    age_cols = [col for col in df_pop.columns if col not in ['annee', 'zone', 'sexe']]
    new_age_cols = [clean_age_label(c) for c in age_cols]

    rename_dict = dict(zip(age_cols, new_age_cols))
    df_pop.rename(columns=rename_dict, inplace=True)

    #st.write("Colonnes du DataFrame population :")
    #st.write(df_pop.columns)

    # Restructurer en format long
    colonnes_fixes = ['annee', 'zone', 'sexe']
    colonnes_ages = [col for col in df_pop.columns if col not in colonnes_fixes]

    df_pop_long = df_pop.melt(
        id_vars=colonnes_fixes,
        value_vars=colonnes_ages,
        var_name="age_tranche",
        value_name="population"
    ).dropna(subset=["population"])

    #st.dataframe(df_pop_long)

    ### --- 2. Agrégation des cas par groupe ---

    df_cas["n"] = 1
    df_cas_grouped = df_cas.groupby(["annee_diag", "departement", "sexe", "age_tranche"], as_index=False)["n"].sum()

    #st.dataframe(df_cas)
    #st.dataframe(df_cas_grouped)

    ### --- 3. Préparation des populations de référence ---

    ref_europe = df_pop_long[df_pop_long["zone"].str.startswith("Europe")]
    ref_monde = df_pop_long[df_pop_long["zone"].str.startswith("Monde")]

    ### --- 4. Calculs des taux ---

    departements = ["Calvados", "Orne", "Manche"]
    sexes = ["Homme", "Femme"]
    annees = sorted(df_cas["annee_diag"].unique())

    resultats = []

    for dep in departements:
        for sexe in sexes:
            for annee in annees:
                # Cas par tranche
                cas = df_cas_grouped[
                    (df_cas_grouped["departement"] == dep) &
                    (df_cas_grouped["sexe"] == sexe) &
                    (df_cas_grouped["annee_diag"] == annee)
                ]

                # Pop locale
                pop_zone = df_pop_long[
                    (df_pop_long["zone"] == dep) &
                    (df_pop_long["sexe"] == sexe) &
                    (df_pop_long["annee"] == annee)
                ]

                # Fusion
                df = pd.merge(pop_zone, cas, on="age_tranche", how="left").fillna({"n": 0})

                # Forcer les types numériques
                df["population"] = pd.to_numeric(df["population"], errors="coerce")
                df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(0)

                # Calculs
                total_cas = df["n"].sum()
                total_pop = df["population"].sum()
                taux_brut = (total_cas / total_pop) * 100000 if total_pop > 0 else np.nan

                # Taux standardisé Europe
                ref_eu = ref_europe[
                    (ref_europe["sexe"] == sexe) &
                    (ref_europe["annee"] == annee)
                ][["age_tranche", "population"]].rename(columns={"population": "pop_ref"})
                df_eu = pd.merge(df, ref_eu, on="age_tranche", how="inner")
                
                df_eu["population"] = pd.to_numeric(df_eu["population"], errors="coerce")
                df_eu["n"] = pd.to_numeric(df_eu["n"], errors="coerce").fillna(0)
                df_eu["pop_ref"] = pd.to_numeric(df_eu["pop_ref"], errors="coerce")

                # Supprimer les lignes avec pop_ref manquante ou 0
                df_eu = df_eu[df_eu["pop_ref"].notna() & (df_eu["pop_ref"] > 0)]
                                
                taux_std_eu = ((df_eu["n"] / df_eu["population"]) * df_eu["pop_ref"]).sum() / df_eu["pop_ref"].sum() * 100000

                # Taux standardisé Monde
                ref_mo = ref_monde[
                    (ref_monde["sexe"] == sexe) &
                    (ref_monde["annee"] == annee)
                ][["age_tranche", "population"]].rename(columns={"population": "pop_ref"})
                df_mo = pd.merge(df, ref_mo, on="age_tranche", how="inner")

                df_mo["population"] = pd.to_numeric(df_mo["population"], errors="coerce")
                df_mo["n"] = pd.to_numeric(df_mo["n"], errors="coerce").fillna(0)
                df_mo["pop_ref"] = pd.to_numeric(df_mo["pop_ref"], errors="coerce")

                # Supprimer les lignes avec pop_ref manquante ou 0
                df_mo = df_mo[df_mo["pop_ref"].notna() & (df_mo["pop_ref"] > 0)]

                taux_std_mo = ((df_mo["n"] / df_mo["population"]) * df_mo["pop_ref"]).sum() / df_mo["pop_ref"].sum() * 100000

                # Sauvegarde
                resultats.append({
                    "Année": annee,
                    "departement": dep,
                    "sexe": sexe,
                    "taux_brut": round(taux_brut, 2),
                    "taux_std_europe": round(taux_std_eu, 2),
                    "taux_std_monde": round(taux_std_mo, 2)
                })

    ### --- 5. Export final ---

    df_resultats = pd.DataFrame(resultats)
    #df_resultats.to_csv("taux_incidence_resultats.csv", index=False)

    #print("✅ Calcul terminé. Résultats exportés dans taux_incidence_resultats.csv")

    st.dataframe(df_resultats)


    ############# INCIDENCE PAR SEXE ET PERIODE #############
    st.subheader(f"📊 Taux d'incidence (par sexe et période) - {hm_libelle}")

    pas = st.slider("Durée des périodes (en années)", min_value=1, max_value=10, value=1, step=1) # Exemple : périodes de 5 ans (modifiable dynamiquement)
    min_annee = df_cas["annee_diag"].min()
    max_annee = df_cas["annee_diag"].max()

    departements_disponibles = ["Calvados", "Orne", "Manche"]

    departements_selectionnes = st.multiselect(
        "Sélectionner les départements à inclure",
        options=departements_disponibles,
        default=departements_disponibles  # les 3 par défaut
    )

    df_cas = df_cas[df_cas["departement"].isin(departements_selectionnes)]



    # Création d'une fonction pour assigner une période à une année
    def get_periode(annee):
        if pd.isna(annee):
            return np.nan
        debut = int((annee - min_annee) // pas * pas + min_annee)
        fin = debut + pas - 1
        return f"{debut}-{fin}"

    df_cas["periode"] = df_cas["annee_diag"].apply(get_periode)
    df_cas_grouped = df_cas.groupby(["periode", "sexe", "age_tranche"], as_index=False)["n"].sum()

    df_pop_long["periode"] = df_pop_long["annee"].apply(get_periode)
    df_pop_grouped = df_pop_long.groupby(["periode", "zone", "sexe", "age_tranche"], as_index=False)["population"].mean()

    periodes = sorted(df_cas["periode"].dropna().unique())
    sexes = ["Homme", "Femme"]
    resultats_region = []

    for sexe in sexes:
        for periode in periodes:
            cas = df_cas_grouped[
                (df_cas_grouped["sexe"] == sexe) &
                (df_cas_grouped["periode"] == periode)
            ]

            # Pop locale (somme des 3 départements)
            pop_zone = df_pop_grouped[
                (df_pop_grouped["zone"].isin(departements_selectionnes)) &
                (df_pop_grouped["sexe"] == sexe) &
                (df_pop_grouped["periode"] == periode)
            ].groupby(["periode", "sexe", "age_tranche"], as_index=False)["population"].sum()

            df = pd.merge(pop_zone, cas, on=["age_tranche"], how="left").fillna({"n": 0})

            total_cas = df["n"].sum()
            total_pop = df["population"].sum()
            taux_brut = (total_cas / total_pop) * 100000 if total_pop > 0 else np.nan

            # Référence Europe
            ref_eu = df_pop_grouped[
                (df_pop_grouped["zone"].str.startswith("Europe")) &
                (df_pop_grouped["sexe"] == sexe) &
                (df_pop_grouped["periode"] == periode)
            ][["age_tranche", "population"]].rename(columns={"population": "pop_ref"})

            df_eu = pd.merge(df, ref_eu, on="age_tranche", how="inner")
            df_eu = df_eu[df_eu["pop_ref"].notna() & (df_eu["pop_ref"] > 0)]
            taux_std_eu = ((df_eu["n"] / df_eu["population"]) * df_eu["pop_ref"]).sum() / df_eu["pop_ref"].sum() * 100000

            # Référence Monde
            ref_mo = df_pop_grouped[
                (df_pop_grouped["zone"].str.startswith("Monde")) &
                (df_pop_grouped["sexe"] == sexe) &
                (df_pop_grouped["periode"] == periode)
            ][["age_tranche", "population"]].rename(columns={"population": "pop_ref"})

            df_mo = pd.merge(df, ref_mo, on="age_tranche", how="inner")
            df_mo = df_mo[df_mo["pop_ref"].notna() & (df_mo["pop_ref"] > 0)]
            taux_std_mo = ((df_mo["n"] / df_mo["population"]) * df_mo["pop_ref"]).sum() / df_mo["pop_ref"].sum() * 100000

            resultats_region.append({
                "Période": periode,
                "sexe": sexe,
                "taux_brut": round(taux_brut, 2),
                "taux_std_europe": round(taux_std_eu, 2),
                "taux_std_monde": round(taux_std_mo, 2)
            })

    df_resultats_region = pd.DataFrame(resultats_region)
 
    st.dataframe(df_resultats_region)


    ### Graphiques évolution taux incidence

    # Créer le nom de la zone dynamiquement
    zone_label = ", ".join(departements_selectionnes)

    st.subheader(f"📉 Évolution du taux brut d’incidence - {hm_libelle}")

    #fig, ax = plt.subplots(figsize=(10, 5))
    #sns.lineplot(
    #    data=df_resultats_region,
    #    x="Période", y="taux_brut", hue="sexe", marker="o", ax=ax
    #)
    #ax.set_title("Taux brut d’incidence par période (3 départements réunis)", fontsize=14)
    #ax.set_ylabel("Taux pour 100 000 hab.")
    #ax.set_xlabel("Période")
    #ax.grid(True)
    #plt.xticks(rotation=45)
    #st.pyplot(fig)

    ##fig plotly
    fig = px.line(
        df_resultats_region,
        x="Période", y="taux_brut",
        color="sexe", markers=True,
        title=f"Taux brut ({zone_label})",
        labels={"taux_brut": "Taux / 100 000 hab."}
    )
    fig.update_layout(title_font_size=16)
    st.plotly_chart(fig)


    ## Taux Std Europe
    st.subheader(f"🌍 Évolution du taux standardisé Europe - {hm_libelle}")

    #fig, ax = plt.subplots(figsize=(10, 5))
    #sns.lineplot(
    #    data=df_resultats_region,
    #    x="Période", y="taux_std_europe", hue="sexe", marker="o", ax=ax
    #)
    #ax.set_title("Taux standardisé Europe par période (3 départements réunis)", fontsize=14)
    #ax.set_ylabel("Taux pour 100 000 hab.")
    #ax.set_xlabel("Période")
    #ax.grid(True)
    #plt.xticks(rotation=45)
    #st.pyplot(fig)

    ##Fig plotly
    fig = px.line(
        df_resultats_region,
        x="Période", y="taux_std_europe",
        color="sexe", markers=True,
        title=f"Taux standardisé Europe ({zone_label})",
        labels={"taux_std_europe": "Taux / 100 000 hab."}
    )
    fig.update_layout(title_font_size=16)
    st.plotly_chart(fig)


    ## Taux Std Monde
    st.subheader(f"🌐 Évolution du taux standardisé Monde - {hm_libelle}")

    #fig, ax = plt.subplots(figsize=(10, 5))
    #sns.lineplot(
    #    data=df_resultats_region,
    #    x="Période", y="taux_std_monde", hue="sexe", marker="o", ax=ax
    #)
    #ax.set_title("Taux standardisé Monde par période (3 départements réunis)", fontsize=14)
    #ax.set_ylabel("Taux pour 100 000 hab.")
    #ax.set_xlabel("Période")
    #ax.grid(True)
    #plt.xticks(rotation=45)
    #st.pyplot(fig)

    ##Fig plotly
    fig = px.line(
        df_resultats_region,
        x="Période", y="taux_std_monde",
        color="sexe", markers=True,
        title=f"Taux standardisé Monde ({zone_label})",
        labels={"taux_std_monde": "Taux / 100 000 hab."}
    )
    fig.update_layout(title_font_size=16)
    st.plotly_chart(fig)


    ### INCIDENCE TOUT SEXE CONFONDU ###
    df_cas_grouped_total = df_cas.groupby(["periode", "age_tranche"], as_index=False)["n"].sum()

    ### Agrégation des populations : somme des sexes pour les cas et les pop ref
    pop_total = df_pop_grouped[
        (df_pop_grouped["zone"].isin(departements_selectionnes))
    ].groupby(["periode", "age_tranche"], as_index=False)["population"].sum()

    ref_europe_total = df_pop_grouped[df_pop_grouped["zone"].str.startswith("Europe")].groupby(
        ["periode", "age_tranche"], as_index=False)["population"].sum().rename(columns={"population": "pop_ref"})

    ref_monde_total = df_pop_grouped[df_pop_grouped["zone"].str.startswith("Monde")].groupby(
        ["periode", "age_tranche"], as_index=False)["population"].sum().rename(columns={"population": "pop_ref"})

    ### Calculs des taux globaux (2 sexes)
    resultats_total = []

    for periode in sorted(df_cas["periode"].dropna().unique()):
        cas = df_cas_grouped_total[df_cas_grouped_total["periode"] == periode]
        pop_zone = pop_total[pop_total["periode"] == periode]

        df = pd.merge(pop_zone, cas, on="age_tranche", how="left").fillna({"n": 0})

        total_cas = df["n"].sum()
        total_pop = df["population"].sum()
        taux_brut = (total_cas / total_pop) * 100000 if total_pop > 0 else np.nan

        # Standardisation Europe
        ref_eu = ref_europe_total[ref_europe_total["periode"] == periode]
        df_eu = pd.merge(df, ref_eu, on="age_tranche", how="inner")
        df_eu = df_eu[df_eu["pop_ref"].notna() & (df_eu["pop_ref"] > 0)]
        taux_std_eu = ((df_eu["n"] / df_eu["population"]) * df_eu["pop_ref"]).sum() / df_eu["pop_ref"].sum() * 100000

        # Standardisation Monde
        ref_mo = ref_monde_total[ref_monde_total["periode"] == periode]
        df_mo = pd.merge(df, ref_mo, on="age_tranche", how="inner")
        df_mo = df_mo[df_mo["pop_ref"].notna() & (df_mo["pop_ref"] > 0)]
        taux_std_mo = ((df_mo["n"] / df_mo["population"]) * df_mo["pop_ref"]).sum() / df_mo["pop_ref"].sum() * 100000

        resultats_total.append({
            "Période": periode,
            "taux_brut": round(taux_brut, 2),
            "taux_std_europe": round(taux_std_eu, 2),
            "taux_std_monde": round(taux_std_mo, 2)
        })

    df_resultats_total = pd.DataFrame(resultats_total)

    st.subheader(f"📊 Taux d’incidence – 2 sexes confondus - {hm_libelle}")
    st.dataframe(df_resultats_total)

    ### Graphiques
    zone_label = ", ".join(departements_selectionnes)

    # Taux brut
    fig = px.line(
        df_resultats_total, x="Période", y="taux_brut",
        markers=True,
        title=f"Taux brut d’incidence (2 sexes confondus) – {zone_label}",
        labels={"taux_brut": "Taux pour 100 000 hab."}
    )
    st.plotly_chart(fig)

    # Taux standardisé Europe
    fig = px.line(
        df_resultats_total, x="Période", y="taux_std_europe",
        markers=True,
        title=f"Taux standardisé Europe (2 sexes confondus) – {zone_label}",
        labels={"taux_std_europe": "Taux pour 100 000 hab."}
    )
    st.plotly_chart(fig)

    # Taux standardisé Monde
    fig = px.line(
        df_resultats_total, x="Période", y="taux_std_monde",
        markers=True,
        title=f"Taux standardisé Monde (2 sexes confondus) – {zone_label}",
        labels={"taux_std_monde": "Taux pour 100 000 hab."}
    )
    st.plotly_chart(fig)

    st.title("📈 Analyse de survie par sexe (Kaplan-Meier)")

# Charger les objets sauvegardés
#if os.path.exists("hm.pkl") and os.path.exists("hm_libelle.pkl"):
#    with open("hm.pkl", "rb") as f:
#        hm = pickle.load(f)
#    with open("hm_libelle.pkl", "rb") as f:
#        hm_libelle = pickle.load(f)
#else:
#    st.error("Fichiers `hm.pkl` ou `hm_libelle.pkl` introuvables.")
#    st.stop()


# #### SURVIE POPULATION GENERALE ####

# def courbe_survie_patient_age_depart(age_depart, sexe, annee, df_insee):
#     # Sélection du sexe
#     if sexe == 1:
#         df = df_insee["hommes"]
#     elif sexe == 2:
#         df = df_insee["femmes"]
#     else:
#         return None

#     # Nettoyage années
#     df = df[pd.to_numeric(df["Année"], errors="coerce").notna()]
#     df["Année"] = df["Année"].astype(int)
#     annee_dispo = df["Année"].unique()
#     annee_closest = min(annee_dispo, key=lambda x: abs(x - annee))

#     ligne = df[df["Année"] == annee_closest]
#     if ligne.empty:
#         return None
#     ligne = ligne.iloc[0]

#     age_groups = [col for col in df.columns if col != "Année"]

#     # Trouver l'indice du groupe d'âge contenant age_depart
#     # Les groupes d'âge ont des labels comme "30 à 34 ans", on va extraire min âge
#     def min_age_from_label(label):
#         if label == "<1":
#             return 0
#         else:
#             # Exemple : "30 à 34 ans" -> 30
#             parts = label.split(" à ")
#             try:
#                 return int(parts[0])
#             except:
#                 return None

#     min_ages = [min_age_from_label(grp) for grp in age_groups]

#     # Trouver l'indice de départ
#     start_idx = None
#     for i, min_a in enumerate(min_ages):
#         if min_a is not None and min_a >= age_depart:
#             start_idx = i
#             break
#     if start_idx is None:
#         return None  # âge de départ trop élevé

#     courbe = []
#     survivants = 1.0
#     # Calcul survie cumulative à partir de start_idx
#     for i in range(start_idx, len(age_groups)):
#         grp = age_groups[i]
#         taux = ligne[grp] / 1000  # taux de décès
#         courbe.append((i - start_idx, survivants))  # temps = nombre de tranches depuis âge départ
#         survivants *= (1 - taux)

#     df_courbe = pd.DataFrame(courbe, columns=["temps", "survie_theorique"])
#     # ajouter info âge départ en colonne pour traçage
#     df_courbe["age_depart"] = age_depart
#     return df_courbe

# def courbe_survie_moyenne_age_depart(hm, df_insee_dict, ages_depart):
#     survivances_par_age = {age: {} for age in ages_depart}
    
#     for _, row in hm.iterrows():
#         age_pat = row["age"]
#         sexe_pat = row["sex"]
#         annee_diag = row["annee_diag"]

#         for age_dep in ages_depart:
#             # On ne calcule la courbe que si l'âge du patient est >= âge de départ (logique)
#             if age_pat >= age_dep:
#                 courbe = courbe_survie_patient_age_depart(age_dep, sexe_pat, annee_diag, df_insee_dict)
#                 if courbe is not None:
#                     for _, ligne in courbe.iterrows():
#                         temps = ligne["temps"]
#                         surv = ligne["survie_theorique"]
#                         survivances_par_age[age_dep].setdefault(temps, []).append(surv)

#     # Calcul moyenne par âge de départ et par temps
#     result = []
#     for age_dep, data in survivances_par_age.items():
#         for temps, valeurs in data.items():
#             result.append({
#                 "age_depart": age_dep,
#                 "temps": temps,
#                 "S_theorique_moyenne": np.mean(valeurs)
#             })

#     df_result = pd.DataFrame(result)
#     return df_result

# ##### FIN DES DEF POUR SURVIE POP GENERALE #####


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
        xaxis_title="Temps (mois)",
        yaxis_title="Probabilité de survie",
        yaxis=dict(range=[0, 1]),
        annotations=annotations,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ### tracé courbe survie pop générale
    # df_insee = {
    #     "hommes": pd.read_excel("fm_t67_clean.xlsx", sheet_name="T67h"),
    #     "femmes": pd.read_excel("fm_t67_clean.xlsx", sheet_name="T67f")
    # }

    # ages_depart = [30, 45, 60, 75]

    # # Assure-toi que ton dataframe hm contient au moins 'age', 'sex', 'annee_diag'
    # # Exemple d'appel :
    # df_survie_moyenne = courbe_survie_moyenne_age_depart(hm, df_insee, ages_depart)

    # # Tracé
    # fig = go.Figure()

    # colors = {30:"blue", 45: "green", 60: "pink", 75: "purple"}
    # colors_sex = {"1": "red", "2": "gray"}  # adapte selon les valeurs réelles de ta variable 'sex'


    # for age_dep in ages_depart:
    #     df_sub = df_survie_moyenne[df_survie_moyenne["age_depart"] == age_dep]
    #     fig.add_trace(go.Scatter(
    #         x=df_sub["temps"] * 5 * 12,  # conversion tranches en mois (5 ans * 12 mois)
    #         y=df_sub["S_theorique_moyenne"],
    #         mode="lines+markers",
    #         name=f"Survie théorique dès {age_dep} ans",
    #         line=dict(color=colors.get(age_dep, "gray"), dash="dot")
    #     ))

    # # Ici tu peux ajouter ta courbe Kaplan-Meier si tu veux, par ex :
    # # Courbes de survie
    # for sex in groups:
    #     subset = df[df['sex'] == sex]
    #     kmf.fit(subset['fup'], subset['event'], label=f"Sexe {sex}")
    #     medians[sex] = kmf.median_survival_time_
    #     rmeans[sex] = kmf.conditional_time_to_event_.mean()
        
    #     fig.add_trace(go.Scatter(
    #         x=kmf.survival_function_.index,
    #         y=kmf.survival_function_[f"Sexe {sex}"],
    #         mode='lines',
    #         name=f"Sexe {sex}",
    #         line=dict(color=colors_sex.get(sex, 'gray')),
    #     ))

    #     # IC (bande)
    #     fig.add_trace(go.Scatter(
    #         x=kmf.confidence_interval_.index,
    #         y=kmf.confidence_interval_[f"Sexe {sex}_upper_0.95"],
    #         mode='lines',
    #         line=dict(width=0),
    #         showlegend=False
    #     ))
    #     fig.add_trace(go.Scatter(
    #         x=kmf.confidence_interval_.index,
    #         y=kmf.confidence_interval_[f"Sexe {sex}_lower_0.95"],
    #         fill='tonexty',
    #         fillcolor='rgba(0,0,0,0.1)',
    #         mode='lines',
    #         line=dict(width=0),
    #         name=f"IC 95% {sex}"
    #     ))

    # fig.update_layout(
    #     title=f"{hm_libelle} - Survie selon le sexe<br><sup>Test de log-rank p = {p_value:.3e}</sup>",
    #     xaxis_title="Temps (mois)",
    #     yaxis_title="Probabilité de survie",
    #     yaxis=dict(range=[0, 1]),
    #     template="plotly_white"
    # )

    # st.plotly_chart(fig, use_container_width=True)

    ### Analyse Survie par sexe et groupe age ###

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

    ### Survie par sexe et groupe d'âge "0-25 ans", "26-50 ans", "51-75 ans", "75 ans+"
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

    ### Survie par sexe et Quartiles calculés d'âge
    # Calculer les quartiles de la colonne 'age'
    hm4 = hm.copy()
    # Calculer les quartiles uniques
    quartiles_hm4 = sorted(hm4['age'].quantile([0, 0.25, 0.5, 0.75, 1.0]).unique())

    # Vérifier qu'on a bien au moins 4 tranches
    if len(quartiles_hm4) >= 4:
        # Créer les labels adaptés
        labels = []
        for i in range(1, len(quartiles_hm4)):
            if i == 1:
                labels.append(f"Q{i} - ≤ {round(quartiles_hm4[i])} ans")
            elif i == len(quartiles_hm4)-1:
                labels.append(f"Q{i} - > {round(quartiles_hm4[i-1])} ans")
            else:
                labels.append(f"Q{i} - {round(quartiles_hm4[i-1])} - {round(quartiles_hm4[i])} ans")

        # Découper selon les bornes uniques
        hm4['grp_age'] = pd.cut(
            hm4['age'],
            bins=quartiles_hm4,
            labels=labels,
            include_lowest=True,
            right=False
        ).astype(str)
    else:
        st.warning("Les quantiles d’âge ne permettent pas une division en quartiles distincts.")

    # Préparer les données pour survie
    df4 = hm4[['fup', 'event', 'sex', 'grp_age']].dropna()
    df4['sex'] = df4['sex'].astype(str)
    df4 = df4[df4['grp_age'] != 'nan']

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
        
    #    else:
    #        st.warning("Aucune donnée trouvée pour ce choix.")

    #else:
    #    st.error("Les données valides (rrhmbn_valide) ne sont pas encore chargées.")


# --- Section : Étude de la survie relative (Pohar-Perme) ---

    import requests

    st.markdown("## Étude de la survie nette (Pohar-Perme)")

    # --- Vérifie que la dataframe `hm` existe ---
    if 'hm' in locals() or 'hm' in globals():
        # st.write("Aperçu de la table `hm` :")
        # st.dataframe(hm)

        # Génération automatique du CSV
        csv_data = hm.to_csv(index=False).encode('utf-8')

        # Bouton de téléchargement
        st.download_button(
            label="📥 Télécharger hm.csv",
            data=csv_data,
            file_name="hm.csv",
            mime="text/csv"
        )

        URL = "https://shiny-upload.emvle.fr/upload"
        HEADERS = {}

        try:
            # Sauvegarde temporaire du CSV pour l'envoi
            with open("hm_temp.csv", "wb") as f:
                f.write(csv_data)
            # Envoi du fichier
            with open("hm_temp.csv", "rb") as f:
                files = {"file": ("hm.csv", f)}
                response = requests.post(URL, files=files, headers=HEADERS, timeout=10)

            st.write(f"Statut : {response.status_code}")
            st.write("Réponse complète du serveur :", response.text)

            if response.status_code == 200:
                st.success("CSV envoyé avec succès au serveur Shiny !")
            else:
                st.error(f"Échec de l'envoi (status {response.status_code}): {response.text}")

        except Exception as e:
            st.warning(f"Erreur lors de l'envoi : {e}")

        shiny_url = "https://shiny.emvle.fr/rrhmbn_v2"
        st.markdown(f"[🚀 Ouvrir le dashboard Shiny pour étude de la survie nette]({shiny_url})", unsafe_allow_html=False)
        
        

        # # Bouton pour envoyer le CSV via POST
        # if st.button("📤 Etude de la survie nette (méthode Pohar-Perme) [ouverture d'une nouvelle fenêtre]"):
        #     URL = "https://shiny-upload.emvle.fr/upload"
        #     HEADERS = {}

        #     try:
        #         # Sauvegarde temporaire du CSV pour l'envoi
        #         with open("hm_temp.csv", "wb") as f:
        #             f.write(csv_data)
        #         # Envoi du fichier
        #         with open("hm_temp.csv", "rb") as f:
        #             files = {"file": ("hm.csv", f)}
        #             response = requests.post(URL, files=files, headers=HEADERS, timeout=10)

        #         st.write(f"Statut : {response.status_code}")
        #         st.write("Réponse complète du serveur :", response.text)



        #         if response.status_code == 200:
        #             st.success("CSV envoyé avec succès au serveur Shiny !")

        #             # Lien vers le dashboard Shiny
        #             shiny_url = "https://shiny.emvle.fr/rrhmbn_v2"
        #             # st.markdown(f"[🚀 Ouvrir le dashboard Shiny pour étude de la survie nette]({shiny_url})", unsafe_allow_html=False)
        #             # Affiche un message et ouvre automatiquement le lien dans un nouvel onglet
        #             st.markdown(
        #                 f"""
        #                 <p>Le fichier a bien été transmis. Ouverture du tableau de bord Shiny...</p>
        #                 <p><a href="{shiny_url}" target="_blank">🚀 Ouvrir le dashboard manuellement</a></p>
        #                 <script>
        #                     // Attendre un court délai pour laisser Streamlit actualiser la page
        #                     setTimeout(() => {{
        #                         window.open("{shiny_url}", "_blank");
        #                     }}, 800);
        #                 </script>
        #                 """,
        #                 unsafe_allow_html=True
        #             )

        #         else:
        #             st.error(f"Échec de l'envoi (status {response.status_code}): {response.text}")

        #     except Exception as e:
        #         st.warning(f"Erreur lors de l'envoi : {e}")
        
        
    else:
        st.warning("⚠️ La dataframe `hm` n’a pas encore été générée.")