import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="RRHMBN - Import des données", layout="wide")
st.title("📊 Analyse RRHMBN - Import des données")

# Choix de la source de données
source_choice = st.radio(
    "📁 Sélectionnez la source des données :",
    ("Importer un fichier Excel", "Utiliser le fichier par défaut"),
    index=0
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
            rrhmbn = pd.read_excel(file, engine="openpyxl")
        else:  # Fichier importé
            rrhmbn = pd.read_excel(file, engine="openpyxl")

        # Renommer les colonnes critiques
        rrhmbn.columns.values[2] = "sex"
        rrhmbn.columns.values[12] = "age"
        rrhmbn.columns.values[16] = "fup"
        rrhmbn.columns.values[17] = "event"

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



st.markdown("---")
st.header("🧬 Sélection d’un sous-ensemble de pathologies (HM)")

if 'rrhmbn_valide' in locals():

    # Méthode de classification
    choix0 = st.radio("Choix de classification :", ["REPIH", "XT"])

    if choix0 == "REPIH":
        choix = st.radio("Type de sélection :", ["Groupe Patho", "Sous-Type Patho"])

        if choix == "Groupe Patho":
            groupes = rrhmbn_valide["patho_groupe"].dropna().unique()
            groupe_id = st.selectbox("Sélectionnez un code de groupe patho :", sorted(groupes))
            hm = rrhmbn_valide[rrhmbn_valide["patho_groupe"] == groupe_id]
            hm_libelle = hm["patho_groupe_label"].iloc[0] if not hm.empty else None

        elif choix == "Sous-Type Patho":
            sous_types = rrhmbn_valide["patho_sous_type"].dropna().unique()
            sous_type_id = st.selectbox("Sélectionnez un code de sous-type patho :", sorted(sous_types))
            hm = rrhmbn_valide[rrhmbn_valide["patho_sous_type"] == sous_type_id]
            hm_libelle = hm["patho_sous_type_label"].iloc[0] if not hm.empty else None

    elif choix0 == "XT":
        xt_labels = rrhmbn_valide["patho_sous_type_XT_label"].dropna().unique()
        xt_label_sel = st.selectbox("Sélectionnez un sous-type XT patho :", sorted(xt_labels))
        hm = rrhmbn_valide[rrhmbn_valide["patho_sous_type_XT_label"] == xt_label_sel]
        hm_libelle = hm["patho_sous_type_XT_label"].iloc[0] if not hm.empty else None

    # Affichage et sauvegarde
    if 'hm' in locals() and not hm.empty:
        st.success(f"✅ Vous avez sélectionné : **{hm_libelle}** ({len(hm)} cas)")
        st.dataframe(hm)

        # Option pour sauvegarder
        if st.button("💾 Sauvegarder les objets `hm` et `hm_libelle`"):
            with open("hm.pkl", "wb") as f:
                pickle.dump(hm, f)
            with open("hm_libelle.pkl", "wb") as f:
                pickle.dump(hm_libelle, f)
            st.success("Objets sauvegardés avec succès (hm.pkl et hm_libelle.pkl)")
    else:
        st.warning("Aucune donnée trouvée pour ce choix.")

else:
    st.error("Les données valides (rrhmbn_valide) ne sont pas encore chargées.")
