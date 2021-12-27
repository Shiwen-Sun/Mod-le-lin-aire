#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#EXAMEN 2021 : Cet examen est composé de deux exercices realises idealement en binome et
#eventuellement seul. Les réponses seront données dans un notebook qui indiquera clairement
# les noms et prenoms du binome d'eleves, ou de l'eleve, l'ayant realise.
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Exercice 1 : Un medecin souhaite mettre en lien l'impact de differentes variables mesurees sur
#un 'score' qu'il estime pour quantifier le niveau d'une maladie. Les donnees sont
#sauvegardees dans le fichier 'obs2021_1.csv'. Idealement, il souhaiterait que seul un
#sous ensemble de ces variables permette d'expliquer le score.
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import pandas
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame

dataframe=pandas.read_csv("./obs2021_1.csv",sep=';')

listColNames=list(dataframe.columns)


XY=dataframe.values
ColNb_Y=listColNames.index('score')
print(XY)

Y=XY[:,ColNb_Y].reshape((XY.shape[0],1))   #reshape is to make sure that Y is a column vector
X = np.delete(XY, ColNb_Y, 1)

listColNames.pop(ColNb_Y)     #to make it contains the column names of X only


for Col in range(len(listColNames)):
  plt.plot(X[:,Col],Y[:],'.')
  plt.xlabel(listColNames[Col])
  plt.ylabel('score')
  plt.show()



#QUESTION 1.1 : Observez les donnees unes par unes. Est-ce que vous identifiez visuellement des liens entre
#certaines variables et la variable 'score'. Si oui, lesquels ?

# Il semblerait qu'il existe une relation linéaire entre la variable score et les variables 12 et 16 (décroissante pour var12 et croissante pour var16)



#QUESTION 1.2 :   On se demande si il est possible de predire le niveau de 'score' à partir d'une
#               seule des variables 'var02', 'var09' ou 'var16'.
#
#QUESTION 1.2.1 : Effectuez une regression lineaire simple entre 'score' et chacune de ces
#               variables.  Toutes les donnees seront utilisees pour l'apprentissage. Evaluez alors la
#               qualité des predictions, sur toutes les donnees, l'aide de la moyenne de l'erreur de
#               prediction sur toutes les donnees, l'aide de la moyenne de l'erreur de prediction au
#               carre (MSE). Quel est le risque potentiel en utilisant cette stratégie de validation
#               de l'apprentissage ?
#

print('Q 1.2.1')
from sklearn.linear_model import LinearRegression

# Régression linéaire pour var02

Var_lr = [2,9,16]

for i in Var_lr:
    
    n = i-1
    
    Var = X[:,n]

    x = Var[:,np.newaxis]

    lr = LinearRegression()
    lr.fit(x,Y)
    
    y_ = lr.predict(x)
    
    plt.plot(x, y_,'b-')
    plt.plot(Var,Y[:],'.')
    plt.xlabel(listColNames[n])
    plt.ylabel('score')
    plt.show()
    
    print(f"l'erreur de prédiction moyenne sur var{i} est : {np.mean(abs(y_-Y))}")
    
    print(f"la moyenne de l'erreur de prédiction au carré (MSE) sur var{i} est : {np.mean((y_-Y)**2)}\n")
    
    
# Le simple fait de vérifier la valeur de la MSE ne suffit pas pour valider le modèle linéaire obtenu, 
# on pourrait penser à calculer le niveau d'incertitude lié aux estimations de b0 et b1 (en supposant que le bruit de mesure est gaussien)
    
# De plus, la construction de la régression linéaire utilisant tous les points ne permet pas une bonne gestion des outliers : 
# en effet, la droite calculée sur l'ensemble des points minimise la distance des points à la droite, 
# mais un outlier (qu'on devrait éliminier du modèle d'apprentissage) peut modifier de façon importante la pente et l'ordonnée à l'origine obtenues.
# Il faudrait donc mettre en place une technique de détection des outliers. 
# Les b0 et b1 obtenus peuvent donc être très instables par rapport aux données d'apprentissage en entrée. 
# On voit notemment que les variables donnent des prédictions très différentes.



#QUESTION 1.2.2 : Evaluez a quel point les predictions sont stables a l'aide d'une methode de validation croisee
#               de type 5-folds.
#
print('Q 1.2.2')
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# première passe pour estimer alpha
kf = KFold(n_splits=5)
for alpha in [0.001,0.01,0.1,1.,10.,20.]:
  sum_mse_scores=0.
  for train, test in kf.split(X):
    X_train=X[train]
    y_train=Y[train]
    X_test=X[test]
    y_test=Y[test]
  
    lasso_regressor = Lasso(alpha=alpha)
    lasso_regressor.fit(X_train, y_train)
  
    y_pred_lasso = lasso_regressor.predict(X_test)
    mse_score_lasso = mean_squared_error(y_test, y_pred_lasso)
    sum_mse_scores+=mse_score_lasso
  print(alpha," total: ",sum_mse_scores)

print()

# deuxième passe pour estimer alpha
kf = KFold(n_splits=5)
for alpha in [7.,7.5,8.,8.5,9.]:
  sum_mse_scores=0.
  for train, test in kf.split(X):
    X_train=X[train]
    y_train=Y[train]
    X_test=X[test]
    y_test=Y[test]
  
    lasso_regressor = Lasso(alpha=alpha)
    lasso_regressor.fit(X_train, y_train)
  
    y_pred_lasso = lasso_regressor.predict(X_test)
    mse_score_lasso = mean_squared_error(y_test, y_pred_lasso)
    sum_mse_scores+=mse_score_lasso
  print('alpha = ', alpha," total: ",sum_mse_scores,)
print('\n')
  

# on obtient le meilleur score pour alpha = 7.5

# On a donc la meilleur régression linéaire avec un coefficient de pénalisation L1 de 7.5, 
# la valeur du coefficient étant assez important, les prédictions sont peu stables. 




#QUESTION 1.2.3 : Peut-on enfin dire si on observe une relation significative entre 'score'
#               et (independament) 'var02', 'var09' ou bien 'var16'. On peut le valider
#               a l'aide d'un test d'hypothese dont on decrira la procedure.

print('Q 1.2.3')
# On calcule donc la régression avec le coefficient alpha optimal de 7.5
lasso_regressor = Lasso(alpha=7.5)
lasso_regressor.fit(X_train, y_train)

print('la régression Lasso:\n\r', lasso_regressor.coef_,'\n')

# On observe que avec la régression Lasso, on obtient un seul coefficient non nul (pour la variable 16)
# Ainsi, on peut en conclure que la seule variable significativement corrélée avec score est la variable 16. 

# Afin de valider que la variable 16 seule suffit, on peut faire la vérification suivante: 
#   1. Calculer le l'indicateur de Mallows Cp pour un nombre de variables (k) allant de 0 à 18
#        Cela nous permet de déterminer le nombre optimal de variables qu'il faut sélectionner pour le modèle
#        (On devrait trouver que k=1 permet de minimiser le critère Cp)
#   2. Utiliser un algorithme de sélection de variable pour choisir la/les meilleures variables prédictives
#        On peut utiliser l'algorithme forward, backward ou stepwise pour le faire
#        (On devrait touver que la variable la plus pertinente est la variable 16)
# On peut alors arriver à une conclusion sur notre observation



#QUESTION 1.3 :   On s'interesse maintenant au lien entre la variable 'score' et 'var12'.
#               On peut remarquer que ces donnees contiennent deux valeurs aberrantes.
#
#QUESTION 1.3.1 : Definissez une procedure pour detecter automatiquement deux donnees aberrantes dans
#               un jeu de donnees.
#

print('Q 1.3.1')
import heapq 

# On définit s2 un estimateur de la variance
def s2(y_true, y_pred):
    n = len(y_true)
    SSE = np.sum((y_true - y_pred)**2)
    return SSE / (n-1)

# On définit une fonction permettant de calculer la distance de Cook du point d'indice i
def cook(x, y, i):
    lr = LinearRegression()
    xx = np.concatenate((x[:i], x[i+1:]), axis=0)
    yy = np.concatenate((y[:i], y[i+1:]), axis=0)
    lr.fit(xx, yy)
    
    lr_tot = LinearRegression()
    lr_tot.fit(x, y)
    y_tot = lr_tot.predict(x)
    
    y_pred = lr.predict(x)
    SSD = np.sum((y_pred - y_tot)**2)
    s_2 = s2(y_tot, y)
    
    return SSD / (2*s_2)

# n_outliers renvoie les indices des n points les plus aberrants selon la distance de Cook 
def n_outliers(X,Y,n):
    d_cook = [0]*len(Y)
    for i in range (len(Y)):
        d_cook[i] = cook(X,Y,i)   
    largest = heapq.nlargest(n,d_cook)
    index_largest = [0]*n
    for i in range(n):
        max = largest[i]
        index_largest[i] = d_cook.index(max)
    return index_largest


print('les indices des',n_outliers(X,Y,2)[0],'et',n_outliers(X,Y,2)[1],'sont les points les plus aberrants:')



#QUESTION 1.3.2 : Nous supprimerons dans la suite de cet exercice les deux observations qui sont aberrantes sur
#               la variable 'var12'. Comment auriez-vous traite ces observations si vous aviez absolument
#                voulu preserver l'information qu'elles contiennent dans les autres variables ?

# On peut calculer la régression linéaire entre la variable 12 et le score sans prendre en compte ces 2 observations
# Cela nous donnerait b0' et b1' (non pollué par les valeurs aberrantes)
# Ensuite, on peut remplacer la valeur de la variable 12 sur ces 2 observations aberrantes par la valeur prédite par le modèle avec b0' et b1'.
# Ainsi, les valeurs 'simulées' pour la variable 12 n'impacteront pas la régression linéaire, 
# et on conserve l'information de ces observations sur les autres variables. 



#QUESTION 1.4 :   Une fois les deux observations aberrantes de 'var12' supprimees, on souhaite selectionner les
#               variables de 'X' qui permettent de prédire au mieux 'score' a l'aide de la
#               regression multiple regularisee.

print('Q 1.4--Suppression des deux points aberrants')
# Suppression des deux points aberrants
X_12 = X[:,11] 
X_12 = X_12[:, np.newaxis] #On récupère la variable12
index_outliers = n_outliers(X_12,Y,2) #On calcule l'indice des points aberrants selon la variable 12

X_pretraite = np.delete(X, index_outliers, 0)
Y_pretraite = np.delete(Y,index_outliers)

Y_pretraite = Y_pretraite[:,np.newaxis]

# On vérifie que X et Y pré-traités ont la taille attendue
print(f"la dimension de X_pretraité est de {np.shape(X_pretraite)}")
print(f"la dimension de Y_pretraité est de {np.shape(Y_pretraite)}\n")

# On vérifie graphiquement que les deux points aberrants pour var12 ont été retirés 
plt.plot(X_pretraite[:,11],Y_pretraite[:],'.')
plt.xlabel(listColNames[11])
plt.ylabel('score')
plt.show() # On voit que les deux observations aberrantes ont bien été supprimées des données
    


#QUESTION 1.4.1 : Quelle strategie vous semble la plus appropriee pour selectionner les variables les plus
#               pertinentes ? Quel pretraitement allez-vous de meme effectuer sur les donnees.

# La plupart des variables ne semblent pas très corrélées au score, 
# on souhaite donc sélectionner un petit nombre de variables pertinentes pour expliquer le score
# C'est donc la régression Lasso qui semble la plus appropriée dans cette situation, car elle permet une sélection des variables pertinentes
# Afin de pouvoir appliquer une régression Lasso, il faudra déterminer le coefficient de pénalité alpha optimal.
# Pour cela, on peut utiliser une technique de validation croisée. 
# Dans le cas présent, la méthode des K-folds semble la plus adaptée car on a assez d'observations pour le faire.



#QUESTION 1.4.2 : Effectuez la procedure de selection des variables optimales en parametrant a la main le poids
#               entre la qualite de prediction et le niveau de regularisation.

print('Q 1.4.2')
# Première passe pour choisir alpha
k_f = KFold(n_splits=5)
for alpha in [0.001,0.01,0.1,1.,10.]:
  sum_mse_scores=0.
  for train, test in k_f.split(X_pretraite):
    X_train=X_pretraite[train]
    y_train=Y_pretraite[train]
    X_test=X_pretraite[test]
    y_test=Y_pretraite[test]
  
    lasso_regressor_ = Lasso(alpha=alpha)
    lasso_regressor_.fit(X_train, y_train)
  
    y_pred_lasso = lasso_regressor_.predict(X_test)
    mse_score_lasso = mean_squared_error(y_test, y_pred_lasso)
    sum_mse_scores+=mse_score_lasso
  print('alpha = ', alpha," total: ",sum_mse_scores)

print()

# Deuxième passe pour choisir alpha
k_f = KFold(n_splits=5)
# for alpha in [0.05,0.1,0.15,0.2,0.25,0.3]:
for alpha in [1.7,2.17,4.56]:
  sum_mse_scores=0.
  for train, test in k_f.split(X_pretraite):
    X_train=X_pretraite[train]
    y_train=Y_pretraite[train]
    X_test=X_pretraite[test]
    y_test=Y_pretraite[test]
  
    lasso_regressor_ = Lasso(alpha=alpha)
    lasso_regressor_.fit(X_train, y_train)
  
    y_pred_lasso = lasso_regressor_.predict(X_test)
    mse_score_lasso = mean_squared_error(y_test, y_pred_lasso)
    sum_mse_scores+=mse_score_lasso
  print('alpha = ',alpha," total: ",sum_mse_scores)

print()

# On choisis alpha = 0.15
alpha_optimal = 0.15

# On effectue la régression linéaire en utilisant le paramètre alpha optimal obtenu
lasso_regressor = Lasso(alpha=alpha_optimal)
lasso_regressor.fit(X_pretraite, Y_pretraite)
print('la régression linéaire:\n ',lasso_regressor.coef_,'\n')
# On observe qu'avec le paramètre optimal de pénalisation, la régression Lasso sélectionne 10 variables



#QUESTION 1.4.3 : Effectuez la procedure automatique de parametrisation de ce poids, de sorte a ce q'un maximum
#               de trois variables soit typiquement selectionne et que la qualite de prediction soit optimale.
#               Quelle methode de validation croisee vous semble la plus raisonnable ici ? La selection des
#               variables est-elle stable ?

print('Q 1.4.3')
import scipy
from scipy.optimize import NonlinearConstraint

def sum_mse_score(alpha):
    sum_mse_scores=0.
    for train, test in k_f.split(X_pretraite):
        X_train=X_pretraite[train]
        y_train=Y_pretraite[train]
        X_test=X_pretraite[test]
        y_test=Y_pretraite[test]
  
        lasso_regressor_ = Lasso(alpha=alpha)
        lasso_regressor_.fit(X_train, y_train)
  
        y_pred_lasso = lasso_regressor_.predict(X_test)
        mse_score_lasso = mean_squared_error(y_test, y_pred_lasso)
        sum_mse_scores+=mse_score_lasso
    return(sum_mse_scores)

def nb_param(alpha):
#     print(resultat_reg.coef_)
    lasso_regressor = Lasso(alpha=alpha)
    lasso_regressor.fit(X_pretraite, Y_pretraite)
    coef = lasso_regressor.coef_
    non_nul = [i for i, e in enumerate(coef) if e != 0]
    return len(non_nul)


from pylab import *
# from scipy.optimize import fmin_tnc
from scipy.optimize import fminbound
import warnings

warnings.filterwarnings("ignore")

xxx = linspace(0, 2, 100)
yyy = [sum_mse_score(x) for x in xxx]
plot(xxx,yyy)

xx2 = linspace(0, 2, 100)
yy2 = [nb_param(x) for x in xx2]
plot(xx2,yy2)

# On observe qu'on a un intervalle pour alpha pour lequel on obtient au plus 3 variables sélectionnées à la fin 
# On peut donc choisir le alpha optimal avec un maximum de 3 variables sélectionnées en minimisant la MSE sur cet intervalle

# 1. Rechercher l'intervalle de alpha où on obtient au plus 3 variables à la fin
# Le nombre de variables sélectionnées est décroissant en fonction de alpha
# L'intervalle est donc de la forme [alpha_ ; +infini]
# On cherche donc alpha_ pour lequel : x < alpha <=> on a sélectionné plus de 3 variables

def recherche_opt(fonction, borne_sup, precision): #on calcule opt_ avec une précision de 1e-(precision)
    opt_ = borne_sup #on observe graphiquement que l'optimal < 10
    l_precision = [1*10**(-p) for p in range(precision+1)]
    for i in l_precision:
        while fonction(opt_) <= 3 :
            opt_ = opt_ - i
        opt_ += i
    return(opt_)

alpha_min = recherche_opt(nb_param, 10, 10)

# 2. Rechercher le min de f(alpha) dans cette zone de alpha
alpha_optim = scipy.optimize.fminbound(sum_mse_score,alpha_min,10)
print('alpha_optim',alpha_optim)

# Résultat prenant le plus petit alpha permettant de sélectionner 
lasso_regressor = Lasso(alpha=alpha_min)
lasso_regressor.fit(X_pretraite, Y_pretraite)
print(f"Pour un alpha_min = {alpha_min}, les coef sont :\n {lasso_regressor.coef_}")

# Résultat pour alpha_optimal
lasso_regressor = Lasso(alpha=alpha_optim)
lasso_regressor.fit(X_pretraite, Y_pretraite)
print(f"Pour alpha_optim = {alpha_optim}, les coef sont :\n {lasso_regressor.coef_}")

print()

# Résultat pour un alpha entre les deux
alpha_interm = [0.08,0.1,0.15]
for alpha_ in alpha_interm:
    lasso_regressor = Lasso(alpha=alpha_)
    lasso_regressor.fit(X_pretraite, Y_pretraite)
    print(f"Pour un alpha = {alpha_}, les coef sont :\n {lasso_regressor.coef_}\n")
# On voit que pour la régression lasso a sélectionné la variable 18 pour alpha =0.08, 
# et que cette variable est éliminée pour alpha = 0.1,
# puis elle est à nouveau sélectionnée pour alpha = 0.15
# Donc la sélection des variables n'est pas stable autour du alpha optimal global
# Nénamoins, la sélection des variables est stable pour un alpha suffisamment grand, 
# c'est le cas notamment pour un alpha permettant de sélectionner 3 variables maximum. 





#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Exercice 2 : Lors d'essais cliniques, un groupe pharmaceutique souhaite savoir si la
#              concentration d'un produit dans un traitement pour la vue a le meme effet
#              sur deux sous populations. Les resultats d'observations sont regroupes dans
#              le fichier obs2021_2.csv. Dans chacun des groupes, on supposera que le lien
#              entre la concentration du produit et l'efficacite du traitement est lineaire.
#              Definissez et appliquez une methodologie pour tester si l'impact de cette
#              concentration est similaire dans les deux groupes ?
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import pandas
import numpy as np
import matplotlib.pyplot as plt

dataframe=pandas.read_csv("./obs2021_2.csv",sep=',')

plt.figure(figsize=(7,7))
plt.scatter(dataframe['concentration'], dataframe['Efficacite'], c=['r' if t == 'Groupe_1' else 'b' for t in dataframe['Groupe']])
plt.xlabel("concentration")
plt.ylabel("Efficacite")
plt.show()

print('Q 2')
# On fait une régression linéaire sur le groupe 1 
Groupe_1 = dataframe[dataframe['Groupe'] == 'Groupe_1']
X_gr1 = Groupe_1['concentration'].values
X_gr1 = X_gr1[:,np.newaxis]
Y_gr1 = Groupe_1['Efficacite'].values

lr_ = LinearRegression()
lr.fit(X_gr1,Y_gr1)
    
y_ = lr.predict(X_gr1)
   
b_gr1_1 = lr.coef_[0]
b_gr1_0 = lr.intercept_   
plt.plot(X_gr1, y_,'r-')
plt.plot(X_gr1,Y_gr1[:],'r.')


# On fait une régression linéaire sur le groupe 2
Groupe_2 = dataframe[dataframe['Groupe'] == 'Groupe_2']
X_gr2 = Groupe_2['concentration'].values
X_gr2 = X_gr2[:,np.newaxis]
Y_gr2 = Groupe_2['Efficacite'].values

lr_ = LinearRegression()
lr.fit(X_gr2,Y_gr2)
    
y_ = lr.predict(X_gr2)

b_gr2_1 = lr.coef_[0]
b_gr2_0 = lr.intercept_

plt.plot(X_gr2, y_,'b-')
plt.plot(X_gr2,Y_gr2[:],'b.')
plt.show()

# Il semble que les deux groupes aient des comportements assez différents, nous allons le vérifier par la suite



# METHODOLOGIE CHOISIE

# On a des comportements linéaires pour deux groupes, avec:
    # Une variable qualitative : le groupe
    # Une variable quantitative : la concentration 
    # Une sortie : l'efficacité
# Ainsi, la méthode la plus adaptée semble être l'analyse de covariance : 
    # on estime d'abord les modèles intra-groupes : calcul de beta_0 et beta_1 pour chacun des groupes 1 et 2
    # on teste les effets différentiels inter-groupes des paramètres de régressions
# Le résultat des tests permettront notamment de nous indiquer avec quel niveau de confiance on pourra affirmer que les groupes 1 et 2 ont le même comportement
# c'est-à-dire qu'ils ont un comportement qu'on peut décrire par des droites de même pente et de même ordonnée à l'origine



# APPLICATION

# On utilise pour l'analyse de la covariance la fonction ancova de la bibliothèque pingouin
# Cette fonction prend en entrée :
    # data : la datadrame étudiée
    # dv : la variable de sortie (y = efficacité)
    # covar : la variable explicative (x = concentration)
    # between : la variable qualitative séparant les données en sous-groupes (x_qualitatif = groupe)
# A partir de ces entrées, elle permet de déterminer s'il existe une différence statistiquement significative entre 
# les moyennes des groupes.

# ajouter la commande suivante si nécessaire : 
# pip install pingouin

from pingouin import ancova
print("l'analyse de la covariance la fonction ancova")
ancova = ancova(data=dataframe, dv='Efficacite',covar='concentration',between='Groupe')
print(ancova)


# RESULTAT

# De la table renvoyée, on peut voir que la p-value (p-unc) pour 'Groupe' est de 8.16 x 10^(-8)
# Cette p-valeur est très faible, la probabilité de faire une erreur en rejetant l'hypthèse nulle est donc très faible
# Il est donc plus probable que l'impact de la concentration sur ces deux groupes est différente. 






