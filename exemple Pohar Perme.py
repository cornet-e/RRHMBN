import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
import matplotlib.pyplot as plt

# Exemple de données de survie pour des patients
# 'time' : Temps de survie (mois)
# 'event' : 1 = Décès, 0 = Censuré
# 'age' : Âge du patient (en années)
# 'sex' : 0 = Femme, 1 = Homme

data = pd.DataFrame({
    'time': [12, 24, 36, 10, 60, 30, 15, 48, 72, 36],  # Temps de survie en mois
    'event': [1, 1, 1, 0, 1, 0, 1, 0, 1, 0],  # 1 = Décès, 0 = Censuré
    'age': [45, 50, 60, 40, 70, 55, 65, 60, 80, 58],  # Âge des patients
    'sex': [0, 1, 0, 1, 1, 0, 1, 1, 0, 0],  # 0 = Femme, 1 = Homme
})

# Estimation de la survie globale à l'aide de Kaplan-Meier
kmf = KaplanMeierFitter()
kmf.fit(data['time'], event_observed=data['event'])

# Visualisation de la courbe de survie globale
plt.figure(figsize=(8, 6))
kmf.plot()
plt.title("Courbe de survie - Cancer hématologique (sans cause connue)")
plt.xlabel("Temps (mois)")
plt.ylabel("Survie")
plt.show()

# Supposons que nous avons les taux de mortalité de la population générale (par tranche d'âge)
# Exemple simplifié : mortalité de la population générale par tranche d'âge
population_mortality_rate = pd.DataFrame({
    'age_group': ['<20', '20-40', '40-60', '>60'],
    'mortality_rate': [0.0005, 0.001, 0.005, 0.02]  # Probabilité de décès par tranche d'âge
})

# Pour cet exemple, supposons que les patients étudiés ont entre 40 et 60 ans
general_population_rate = 0.005  # Taux de mortalité de la population générale pour les 40-60 ans

# Estimation de la survie nette en ajustant la survie observée
# Ajustement simple : soustraction de la survie attendue (survie dans la population générale)
survival_adjustment = kmf.survival_function_.values - general_population_rate

# Visualisation de la survie nette ajustée
plt.figure(figsize=(8, 6))
plt.plot(kmf.survival_function_.index, survival_adjustment, label='Survie nette ajustée')
plt.title("Survie nette ajustée - Comparaison avec la population générale")
plt.xlabel("Temps (mois)")
plt.ylabel("Survie nette ajustée")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Intégration du modèle de Cox pour ajuster la survie en fonction des covariables
# --------------------------------------------------------------

# Ajustement du modèle de Cox avec les covariables (âge, sexe)
cph = CoxPHFitter()
cph.fit(data[['time', 'event', 'age', 'sex']], duration_col='time', event_col='event')

# Affichage des résultats du modèle de Cox
cph.print_summary()

# Estimation de la survie ajustée pour chaque patient avec le modèle de Cox
# Nous allons maintenant estimer la survie pour chaque patient en fonction des covariables
survival_function = cph.predict_survival_function(data[['age', 'sex']])

# Transposition pour assurer que chaque courbe de survie correspond à un patient
survival_function = survival_function.T  # Cela transpose les données, chaque ligne devient une courbe pour un patient

# Visualisation de la courbe de survie ajustée avec le modèle de Cox
plt.figure(figsize=(8, 6))
for i in range(len(survival_function)):
    plt.plot(survival_function.columns, survival_function.iloc[i], label=f"Patient {i+1}")

plt.title("Survie ajustée par le modèle de Cox")
plt.xlabel("Temps (mois)")
plt.ylabel("Survie ajustée")
plt.legend(loc='best')
plt.show()

