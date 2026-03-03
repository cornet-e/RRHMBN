import pandas as pd

fichier_brut = "rrhmbn_brut.xlsx"
fichier_select = "Import-select.xlsx"
fichier_sortie = "rrhmbn.xlsx"

df_brut = pd.read_excel(fichier_brut, header=0)
df_select = pd.read_excel(fichier_select, header=0)

# --- Normalisation des colonnes du fichier BRUT ---
# On crée un dictionnaire pour garder le nom original tout en cherchant en minuscules
colonnes_brut_nettoyees = {col.strip().lower(): col for col in df_brut.columns}

# --- Normalisation des colonnes REQUISES ---
colonnes_requises_brutes = df_select.columns.str.strip().tolist()

colonnes_finales_a_extraire = []
colonnes_absentes = []

for col in colonnes_requises_brutes:
    nom_mini = col.lower()
    if nom_mini in colonnes_brut_nettoyees:
        # On récupère le vrai nom (tel qu'il est dans le fichier brut)
        colonnes_finales_a_extraire.append(colonnes_brut_nettoyees[nom_mini])
    else:
        colonnes_absentes.append(col)

# --- Extraction et Sauvegarde ---
if colonnes_finales_a_extraire:
    df_final = df_brut[colonnes_finales_a_extraire].copy()
    df_final.to_excel(fichier_sortie, index=False)
    print(f"✅ Succès ! {len(colonnes_finales_a_extraire)} colonnes extraites.")
else:
    print("❌ Aucune colonne correspondante n'a été trouvée.")

if colonnes_absentes:
    print(f"⚠️ Colonnes toujours introuvables : {colonnes_absentes}")