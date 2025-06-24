import pandas as pd

# --- Paramètres des chemins des fichiers ---
fichier_brut = "rrhmbn_brut.xlsx"
fichier_select = "Import-select.xlsx"
fichier_sortie = "rrhmbn.xlsx"

# --- Lecture des fichiers avec ligne d'en-tête ---
df_brut = pd.read_excel(fichier_brut, header=0)
df_select = pd.read_excel(fichier_select, header=0)

# --- Colonnes à extraire : on prend les noms de colonnes du fichier Import-select ---
colonnes_requises = df_select.columns.str.strip().tolist()

# --- Vérification des colonnes réellement présentes dans le fichier brut ---
colonnes_presentes = [col for col in colonnes_requises if col in df_brut.columns]
colonnes_absentes = [col for col in colonnes_requises if col not in df_brut.columns]

# --- Affichage console pour information ---
print("✅ Colonnes trouvées dans le fichier brut :")
print(colonnes_presentes)

if colonnes_absentes:
    print("\n⚠️ Colonnes absentes dans le fichier brut :")
    print(colonnes_absentes)

# --- Extraction des colonnes présentes ---
df_final = df_brut[colonnes_presentes].copy()  # ✅ Copie sécurisée

# --- Sauvegarde du fichier final ---
df_final.to_excel(fichier_sortie, index=False)
print(f"\n💾 Fichier sauvegardé : {fichier_sortie}")


