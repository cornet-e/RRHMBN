# example_pohar_rpy2.py
import pandas as pd
from rpy2.robjects import r, pandas2ri
import rpy2.robjects.packages as rpackages
from rpy2.robjects.conversion import localconverter

# activer conversion pandas <-> R
pandas2ri.activate()

# vérification / installation (optionnel)
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)  # choisir un miroir
if not rpackages.isinstalled('relsurv'):
    utils.install_packages('relsurv')

# importer relsurv
relsurv = rpackages.importr('relsurv')
survival = rpackages.importr('survival')  # pour Surv

# exemple de dataframe pandas (remplacer par votre vrai df)
df = pd.DataFrame({
    'time': [365, 400, 200, 900],         # en jours
    'status': [1,0,1,0],                  # 1=event (death), 0=censor
    'age': [65, 70, 55, 60],              # en années
    'sex': [1, 2, 1, 2],                  # ex: 1=male,2=female
    'year': pd.to_datetime(['2000-01-01','2000-06-01','2001-03-01','2002-07-01'])
})

# convertir vers R
with localconverter(rpy2.robjects.default_converter + pandas2ri.converter):
    r_df = pandas2ri.py2rpy(df)

# exemple d'appel : rs.surv (méthode 'pohar-perme')
# formule R : Surv(time, status) ~ 1   (pas de covariables, estimation marginale)
r.assign('r_df', r_df)
r('library(relsurv)')
# utiliser le jeu de tables de population intégré 'slopop' pour tester (exemple/demo)
r('data(slopop, package="relsurv")')  

# appeler rs.surv (method = "pohar-perme")
r('''
res <- rs.surv(Surv(time, status) ~ 1, data = r_df,
               ratetable = slopop, method = "pohar-perme")
# afficher un résumé (résultat 'survfit' R)
print(summary(res))
''')

# si vous voulez ramener le résultat en Python, récupérez res
res = r('res')
# par ex. récupérer times et surv :
times = list(r('res$time'))
surv  = list(r('res$surv'))
print("times:", times)
print("net survival (Pohar-Perme):", surv)
