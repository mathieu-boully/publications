# L3-SID INTRODUCTION AU MACHINE LEARNING
## TP 3 + projet: Apprentissage via scikit-learn

## Objectifs du TP
Le travail sera commencé en séance puis terminé sous forme de projet qui
devra être rendu dans un rapport (préférablement sous format `.pdf`)
pour l’ensemble du TP. Il devra s’agir d’un travail PERSONNEL. Vous noterez vos réponses ainsi que vos commentaires, vos choix, vos figures et vos discussions sur les
résultats directement dans le notebook (ou un fichier .py et un rapport séparé compressés dans une archive). Le devoir devra être déposé sur moodle avant le 25 mai. 
Dans ce TP, on utilisera la base `titanic`.

## Chargement de la base et sélection des données

* Téléchargez la base *titanic.csv* et chargez-la dans un DataFrame (pandas), noté *df* (pandas).
* Affichez les attributs de la base et leur nombre. De quel type sont-ils (réels, nominaux ou variables binaires,...) ? Combien y a-t-il de classes ?
*  vous devez maintenant nettoyer la base :
    * Retirer les colonnes qui vous semblent inutiles pour l'apprentissage à l'aide de *df.drop(index)*
    * Certaines variables sont nominatives et scikit learn ne traite que des variables numériques. Transformer les variables numériques en variables binaires à de *pandas.get\_dummies*
    * remplacer les valeurs nan par la moyenne à l'aide de *df.fillna*
* Afin de créer une base d'apprentissage et une base de test, créez une fonction *split(df,p)* qui séparera et retournera deux DataFrames, notés *train* et *test* dont la taille est déterminée par un pourcentage *p* passé en paramètre (60/40%, 70/30%,...). Appliquez la fonction à *df*.


```python
!pip install nbconvert
!pip install pandas
!pip install numpy
!pip install matplotlib.pyplot
!pip install matplotlib
!pip install csv --upgrade
!pip install xelatex 
```

    Requirement already satisfied: pandas in c:\programdata\anaconda3\lib\site-packages (1.0.5)
    Requirement already satisfied: pytz>=2017.2 in c:\programdata\anaconda3\lib\site-packages (from pandas) (2020.1)
    Requirement already satisfied: numpy>=1.13.3 in c:\programdata\anaconda3\lib\site-packages (from pandas) (1.18.5)
    Requirement already satisfied: python-dateutil>=2.6.1 in c:\programdata\anaconda3\lib\site-packages (from pandas) (2.8.1)
    Requirement already satisfied: six>=1.5 in c:\programdata\anaconda3\lib\site-packages (from python-dateutil>=2.6.1->pandas) (1.15.0)
    Requirement already satisfied: numpy in c:\programdata\anaconda3\lib\site-packages (1.18.5)
    

    ERROR: Could not find a version that satisfies the requirement matplotlib.pyplot (from versions: none)
    ERROR: No matching distribution found for matplotlib.pyplot
    

    Requirement already satisfied: matplotlib in c:\programdata\anaconda3\lib\site-packages (3.2.2)
    Requirement already satisfied: cycler>=0.10 in c:\programdata\anaconda3\lib\site-packages (from matplotlib) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\programdata\anaconda3\lib\site-packages (from matplotlib) (1.2.0)
    Requirement already satisfied: numpy>=1.11 in c:\programdata\anaconda3\lib\site-packages (from matplotlib) (1.18.5)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in c:\programdata\anaconda3\lib\site-packages (from matplotlib) (2.4.7)
    Requirement already satisfied: python-dateutil>=2.1 in c:\programdata\anaconda3\lib\site-packages (from matplotlib) (2.8.1)
    Requirement already satisfied: six in c:\programdata\anaconda3\lib\site-packages (from cycler>=0.10->matplotlib) (1.15.0)
    

    ERROR: Could not find a version that satisfies the requirement csv (from versions: none)
    ERROR: No matching distribution found for csv
    ERROR: Could not find a version that satisfies the requirement xelatex (from versions: none)
    ERROR: No matching distribution found for xelatex
    


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
import csv
```

## Les données


```python
df = pd.read_csv('titanic.csv') # chargement du jeu de données
df.head() # on affiche les 5 premières lignes
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.columns) # on affiche la liste des variables de la base
print(len(df.columns)) # on affiche le nombre d'attributs
print(df.shape) # on affiche la taille du dataframe
print(df.info()) # on affiche la structure du DataFrame
print(df.describe(include='O')) # descriptions des variables qualitatives
df.describe() # statistiques descriptives univariées
```

    Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
           'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
          dtype='object')
    12
    (891, 12)
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
    None
                                 Name   Sex    Ticket Cabin Embarked
    count                         891   891       891   204      889
    unique                        891     2       681   147        3
    top     Millet, Mr. Francis Davis  male  CA. 2343    G6        S
    freq                            1   577         7     4      644
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



Le DataFrame contient **891 observations** (=lignes) et **12 attributs** (=colonnes).

Nous avons des attributs de type quantitatifs et qualitatifs.

La variable *Cabin* a beaucoup de valeurs null.

Les attributs *Name*, *Ticket* et *Cabin* ont trop de valeurs uniques. Elles n'apportent pas dinformations.

On distingue **2 classes** (1 si le passager a survécu et 0 s’il est décédé) qui correspondent à la variable *Survived*. C'est la variable que nous cherchons à déterminer/prédire.


```python
# répartition de la classe Survived
print('Part des survivants : ', round(len(df[df['Survived']==1]) / len(df['Survived']) * 100, 1), '%')
print('Part des non-survivants : ', round(len(df[df['Survived']==0]) / len(df['Survived']) * 100, 1), '%')
df.groupby(['Sex']).mean() # comparaison des modalités de la variable Sex sur les autres variables
```

    Part des survivants :  38.4 %
    Part des non-survivants :  61.6 %
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>431.028662</td>
      <td>0.742038</td>
      <td>2.159236</td>
      <td>27.915709</td>
      <td>0.694268</td>
      <td>0.649682</td>
      <td>44.479818</td>
    </tr>
    <tr>
      <th>male</th>
      <td>454.147314</td>
      <td>0.188908</td>
      <td>2.389948</td>
      <td>30.726645</td>
      <td>0.429809</td>
      <td>0.235702</td>
      <td>25.523893</td>
    </tr>
  </tbody>
</table>
</div>



74% des personnes qui ont survécus des femmes contre 19% d'hommes.

## Nettoyage de la base

Les colonnes qui me semblent inutiles pour la base d'apprentissage sont *PassengerId*, *SibSp* (=nombre d’époux, de frères ou de soeurs présents à bord), *Parch* (=nombre de parents ou d’enfants présents à bord), *Name*, *Ticket* et *Cabin*.


```python
df = df.drop(['PassengerId','SibSp','Parch','Name','Ticket','Cabin','Embarked'], axis=1) # je supprime les variables inutiles
print(df.columns) # on affiche la nouvelle liste des variables d'apprentissage (features) et la classe (target)
```

    Index(['Survived', 'Pclass', 'Sex', 'Age', 'Fare'], dtype='object')
    

Nous retenons donc 4 variables explicatives (2 quantitatives et 2 qualitatives) ainsi que la classe à prédire *Survived*.


```python
df = pd.get_dummies(df, columns=['Pclass','Sex']) # convertion de la variable Pclass et Sex en type quantitatif binaire
```

On remarque des valeurs manquantes pour la variable *Age*


```python
df = df.fillna(df['Age'].mean()) # on remplace les valeurs nan par la moyenne
print(df.head()) # affichage du nouveau DataFrame nettoyé
df.info() # les valeurs nulles ont été remplacés par la moyenne
```

       Survived   Age     Fare  Pclass_1  Pclass_2  Pclass_3  Sex_female  Sex_male
    0         0  22.0   7.2500         0         0         1           0         1
    1         1  38.0  71.2833         1         0         0           1         0
    2         1  26.0   7.9250         0         0         1           1         0
    3         1  35.0  53.1000         1         0         0           1         0
    4         0  35.0   8.0500         0         0         1           0         1
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 8 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   Survived    891 non-null    int64  
     1   Age         891 non-null    float64
     2   Fare        891 non-null    float64
     3   Pclass_1    891 non-null    uint8  
     4   Pclass_2    891 non-null    uint8  
     5   Pclass_3    891 non-null    uint8  
     6   Sex_female  891 non-null    uint8  
     7   Sex_male    891 non-null    uint8  
    dtypes: float64(2), int64(1), uint8(5)
    memory usage: 25.4 KB
    

## Création d'une base d'apprentissage et d'une base de test

Avec la fonction **split** à la "main".


```python
def split(df,p): 
    # fonction split qui sépare et retourne deux DataFrames selon la valeur de p
    # entrées : 
        # - df : DataFrame complet
        # - p : taille de la séparation
    # sorties : 
        # - train : échantillon d'entraînement
        # - test : : échantillon de test
    train = df.sample(frac=p,replace=False)
    test = df.sample(frac=(1-p),replace=False)
    return train, test

# test de la fonction split (taille 60%, 40%)
train, test = split(df,0.6)
print('Nb lignes échantillon d\'entraînement : ', len(train)) # utilisé pour entrainer le modèle
print('Nb lignes échantillon de test : ', len(test)) # sert à évaluer la prformance du modèle
```

    Nb lignes échantillon d'entraînement :  535
    Nb lignes échantillon de test :  356
    

En utilisant la fonction **train_test_split** de scikit-learn.


```python
from sklearn.model_selection import train_test_split # importation de la fonction train_test_split
def split(df,y,p=0.6): 
    # fonction split qui sépare et retourne 2 DataFrames selon la valeur de p (=test_size)
    # entrées : 
        # - df : DataFrame complet
        # - y : target
        # - p : taille de test_size (par défaut p=0.4)
    # sorties : 
        # - train_test_split : fonction scikit-learn qui renvoie X_train, X_test, y_train et y_test    
    X = df.drop(y,axis=1) # features
    Y = df[y] # target
    return train_test_split(X,Y,train_size=p) # j'utilise la fonction train_test_split de scikit-learn

# test de la fonction split (taille 60%, 40%)
y = 'Survived' # target
X_train, X_test, Y_train, Y_test = split(df,y) # appel de la fonction split
print('Nb lignes échantillons d\'entraînement : ', len(X_train)) # échantilllon pour entrainer le modèle
print('Nb lignes échantillons de test : ', len(X_test)) # sert à évaluer la performance du modèle
```

    Nb lignes échantillons d'entraînement :  534
    Nb lignes échantillons de test :  357
    

# Entropie
Le but ici est de caculer l'entropie et de gain d'entropie comme pour la construction d'un arbre de décision.
* Ecrire une fonction entropie qui prend une liste (ou un array numpy) de valeur binaires et qui calcule l'entropie. Appliquer cette fonction sur la classe.
* Ecrire une fonction gain d'entropie qui prend une liste pour la classe (en binaire) et une liste pour les valeurs correspondantes d'un autre attribut (binaire aussi) 
* Calculer le gain d'entropie pour l'attribut pour le sexe et toutes les autres variables binaires que vous avez crées. Quelle serait le meilleur attribut pour démarrer un arbre de décision ?

## Fonction entropie


```python
from math import *

def log2(x):
    # fonction calcule du log
    return log(x)/log(2)

def entropie(n1,n2):
    # fonction entropie qui retourne la valeur de l'entropie
    # entrées :
        # - n1 : nombre de valeurs dans la classe 1
        # - n2 : nombre de valeurs dans la classe 2
    # sortie :
        # - valeur de l'entropie
    p1 = n1/(n1+n2)
    p2 = n2/(n1+n2)
    if n1 == 0 or n2 == 0:
        return 0
    else:
        return -p1*log2(p1)-p2*log2(p2)
    
# test de la fonction entropie sur la classe Survived
n1=len(df[df['Survived']==0])
n2=len(df[df['Survived']==1])
print('Entropie de la classe Survived : ',round(entropie(n1,n2),2))
```

    Entropie de la classe Survived :  0.96
    

La classe *Survived* est homogène, il y a une bonne répartition des valeurs binaires (équilibré). Il y a autant d'individus survivants que non survivants.

## Fonction gain entropie


```python
def gain(S,l):
    # fonction gain d'entropie qui retourne la valeur du gain d'entropie
    # entrées :
        # - S
        # - l
    # sorties : 
        # - g : valeur du gain d'entropie
    sp,sn = S
    nb = sp + sn 
    ent_s = entropie(sp,sn)
    g = ent_s
    for ap,an in l:
        g = g - (ap+an)/nb * entropie(ap,an)
    return g

# test de la fonction gain d'entropie sur le Sexe
n0=len(df[df['Survived']==0])
n1=len(df[df['Survived']==1])

sf0=len(df[df['Sex_female']==0]) # nombre de décès chez les femmes
sf1=len(df[df['Sex_female']==1]) # nombre de survivants chez les femmes

sm0=len(df[df['Sex_male']==0]) # nombre de décès chez les hommes
sm1=len(df[df['Sex_male']==1]) # nombre de survivants chez les hommes

print('Gain d\'entropie pour la variable Sexe : ',gain((n0,n1),[(sf0,sf1),(sm0,sm1)]))
```

    Gain d'entropie pour la variable Sexe :  -0.9117013846240573
    

Gain d\'entropie pour la variable *Pclass* (variable bianire) :


```python
# test de la fonction gain d'entropie sur Pclass
n0=len(df[df['Survived']==0])
n1=len(df[df['Survived']==1])

s10=len(df[df['Pclass_1']==0])
s11=len(df[df['Pclass_1']==1]) # nombre de passager de la classe 1

s20=len(df[df['Pclass_2']==0])
s21=len(df[df['Pclass_2']==1]) # nombre de passager de la classe 2

s30=len(df[df['Pclass_3']==0])
s31=len(df[df['Pclass_3']==1]) # nombre de passager de la classe 3

print('Gain d\'entropie pour la variable Pclass : ',gain((n0,n1),[(s10,s11),(s20,s21),(s30,s31)]))
```

    Gain d'entropie pour la variable Pclass :  -1.56556074408324
    

Le meilleur attribut pour démarrer un arbre de décision est la variable *Sexe* car le gain d'entropie est le plus élevé, la séparation n'est pas équilibrée (dissociation des deux classes, bon classifieur).

## Familiarisation avec quelques méthodes du package *scikit-learn*+
On utilisera les fonctions d'apprentissage bayésien naïf de *scikit-learn* pour prédire la classe de la base *test* créée précédemment lorsque le modèle prédictif est calculé à partir de la base d'apprentissage *train*. Pour cela, vous aurez besoin des fonctions:


```python
from sklearn.naive_bayes import GaussianNB # importation du module GaussianNB
clf = GaussianNB() # création du classifieur (bayésien naïf)
clf.fit(X_train,Y_train) # apprentissage du classfieur sur les données train
print(clf.predict(X_test)) # prédiction de la classe Survived avec la base de test
print(clf.predict_proba(X_test)) # probabilité d'appartenance à une classe
```

    [0 0 1 0 0 0 0 0 0 1 0 0 0 0 1 0 1 1 1 0 1 0 1 0 1 0 1 1 1 1 0 1 0 0 1 1 0
     1 1 1 0 0 0 1 0 1 0 1 1 1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1
     0 0 0 0 0 1 1 0 1 1 1 1 1 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
     0 0 0 1 1 0 1 0 0 0 0 0 0 0 1 0 0 1 1 0 0 0 1 0 0 0 0 1 1 1 0 0 0 1 0 1 0
     1 0 1 0 0 0 1 1 1 0 1 1 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 1 0 0 0 1 0 1 1
     1 0 1 1 0 0 0 0 1 1 0 0 0 1 1 1 1 1 1 0 0 1 0 0 1 0 0 1 1 1 0 1 0 0 1 0 0
     1 0 0 0 1 0 0 0 1 1 1 0 0 0 0 0 1 0 1 1 0 1 1 1 1 1 0 0 1 1 0 1 1 1 0 1 0
     1 0 0 0 0 0 0 1 1 0 1 1 1 1 1 0 1 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 1 0 1 0 1
     0 1 0 1 0 1 1 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 1 1 1
     0 1 1 0 1 1 0 0 0 0 0 1 1 0 0 0 1 0 0 1 0 0 0 1]
    [[9.58558306e-01 4.14416943e-02]
     [9.96300906e-01 3.69909397e-03]
     [2.51241490e-02 9.74875851e-01]
     [9.11517817e-01 8.84821828e-02]
     [7.40739796e-01 2.59260204e-01]
     [8.13123892e-01 1.86876108e-01]
     [9.96620971e-01 3.37902927e-03]
     [9.96401033e-01 3.59896674e-03]
     [7.32231876e-01 2.67768124e-01]
     [3.00694263e-01 6.99305737e-01]
     [9.94337514e-01 5.66248606e-03]
     [9.95681884e-01 4.31811589e-03]
     [9.96390956e-01 3.60904371e-03]
     [9.96528398e-01 3.47160228e-03]
     [2.09940009e-01 7.90059991e-01]
     [9.62951756e-01 3.70482439e-02]
     [2.44345772e-02 9.75565423e-01]
     [5.39829885e-04 9.99460170e-01]
     [1.51544880e-01 8.48455120e-01]
     [9.18521577e-01 8.14784227e-02]
     [3.98977830e-09 9.99999996e-01]
     [9.96449348e-01 3.55065182e-03]
     [2.60064859e-01 7.39935141e-01]
     [9.52174981e-01 4.78250194e-02]
     [3.01309299e-02 9.69869070e-01]
     [9.96275090e-01 3.72491025e-03]
     [2.84643608e-02 9.71535639e-01]
     [2.97433969e-02 9.70256603e-01]
     [1.83536120e-03 9.98164639e-01]
     [1.17953659e-03 9.98820463e-01]
     [9.95944120e-01 4.05588043e-03]
     [1.40479416e-01 8.59520584e-01]
     [9.95878603e-01 4.12139654e-03]
     [9.59256940e-01 4.07430597e-02]
     [4.38437760e-01 5.61562240e-01]
     [2.78864564e-02 9.72113544e-01]
     [9.60455515e-01 3.95444846e-02]
     [2.59293457e-04 9.99740707e-01]
     [3.92200011e-04 9.99607800e-01]
     [2.52162707e-02 9.74783729e-01]
     [9.96220771e-01 3.77922921e-03]
     [6.84317859e-01 3.15682141e-01]
     [7.36139007e-01 2.63860993e-01]
     [3.25073593e-02 9.67492641e-01]
     [9.96524328e-01 3.47567248e-03]
     [1.47793649e-01 8.52206351e-01]
     [9.61543562e-01 3.84564380e-02]
     [1.34411645e-01 8.65588355e-01]
     [3.00656230e-02 9.69934377e-01]
     [4.45897940e-04 9.99554102e-01]
     [9.50102722e-01 4.98972778e-02]
     [8.13171921e-04 9.99186828e-01]
     [9.95587384e-01 4.41261570e-03]
     [9.92280448e-01 7.71955168e-03]
     [9.96144549e-01 3.85545088e-03]
     [2.24171773e-01 7.75828227e-01]
     [9.62720338e-01 3.72796620e-02]
     [9.95960318e-01 4.03968170e-03]
     [9.95587384e-01 4.41261570e-03]
     [9.95676519e-01 4.32348120e-03]
     [9.96314391e-01 3.68560944e-03]
     [9.58823499e-01 4.11765007e-02]
     [6.50615851e-01 3.49384149e-01]
     [9.96524311e-01 3.47568938e-03]
     [9.96437299e-01 3.56270128e-03]
     [9.96344151e-01 3.65584860e-03]
     [7.34120260e-01 2.65879740e-01]
     [9.96131809e-01 3.86819117e-03]
     [2.74741745e-01 7.25258255e-01]
     [9.95959036e-01 4.04096406e-03]
     [9.95962392e-01 4.03760807e-03]
     [9.62666614e-01 3.73333857e-02]
     [9.96149449e-01 3.85055050e-03]
     [2.35450472e-01 7.64549528e-01]
     [7.29055048e-01 2.70944952e-01]
     [7.31925605e-01 2.68074395e-01]
     [9.96591900e-01 3.40810027e-03]
     [9.95780068e-01 4.21993202e-03]
     [9.94159381e-01 5.84061892e-03]
     [2.24199822e-01 7.75800178e-01]
     [2.41502244e-01 7.58497756e-01]
     [9.63735534e-01 3.62644662e-02]
     [3.11693257e-04 9.99688307e-01]
     [2.53365640e-01 7.46634360e-01]
     [3.13949600e-02 9.68605040e-01]
     [1.65840556e-03 9.98341594e-01]
     [3.09412610e-02 9.69058739e-01]
     [9.94126703e-01 5.87329673e-03]
     [9.55528953e-01 4.44710467e-02]
     [9.21953807e-01 7.80461927e-02]
     [2.84110188e-01 7.15889812e-01]
     [9.96452696e-01 3.54730429e-03]
     [9.96114878e-01 3.88512244e-03]
     [9.96527851e-01 3.47214865e-03]
     [9.96456155e-01 3.54384510e-03]
     [2.83407174e-02 9.71659283e-01]
     [9.96238695e-01 3.76130510e-03]
     [9.95673597e-01 4.32640330e-03]
     [9.58140830e-01 4.18591702e-02]
     [9.96438598e-01 3.56140224e-03]
     [9.93638979e-01 6.36102060e-03]
     [9.96445914e-01 3.55408632e-03]
     [9.95587384e-01 4.41261570e-03]
     [6.06968294e-01 3.93031706e-01]
     [9.54792530e-01 4.52074700e-02]
     [9.96452696e-01 3.54730429e-03]
     [9.96533145e-01 3.46685541e-03]
     [9.95297355e-01 4.70264534e-03]
     [9.96216525e-01 3.78347487e-03]
     [7.72255551e-02 9.22774445e-01]
     [7.33204751e-01 2.66795249e-01]
     [9.63010135e-01 3.69898649e-02]
     [9.96226138e-01 3.77386184e-03]
     [9.96549154e-01 3.45084581e-03]
     [2.78864564e-02 9.72113544e-01]
     [2.29072103e-54 1.00000000e+00]
     [9.95957389e-01 4.04261147e-03]
     [3.65356244e-11 1.00000000e+00]
     [9.96564369e-01 3.43563129e-03]
     [9.96389319e-01 3.61068087e-03]
     [9.96527239e-01 3.47276129e-03]
     [9.96436778e-01 3.56322173e-03]
     [9.95911714e-01 4.08828605e-03]
     [9.96484642e-01 3.51535817e-03]
     [9.96225475e-01 3.77452451e-03]
     [2.38042852e-01 7.61957148e-01]
     [6.04192584e-01 3.95807416e-01]
     [9.96452696e-01 3.54730429e-03]
     [2.58804529e-01 7.41195471e-01]
     [2.50422689e-01 7.49577311e-01]
     [9.54644871e-01 4.53551291e-02]
     [9.96436778e-01 3.56322173e-03]
     [9.62720338e-01 3.72796620e-02]
     [9.21100010e-07 9.99999079e-01]
     [9.96390283e-01 3.60971675e-03]
     [9.93364842e-01 6.63515795e-03]
     [9.96487942e-01 3.51205769e-03]
     [9.96460675e-01 3.53932539e-03]
     [2.52403698e-01 7.47596302e-01]
     [7.92343937e-08 9.99999921e-01]
     [4.14265049e-04 9.99585735e-01]
     [9.63056533e-01 3.69434671e-02]
     [9.96562892e-01 3.43710750e-03]
     [9.26245000e-01 7.37549995e-02]
     [2.55902780e-01 7.44097220e-01]
     [7.36296013e-01 2.63703987e-01]
     [2.60221830e-01 7.39778170e-01]
     [9.96537009e-01 3.46299094e-03]
     [2.60129321e-01 7.39870679e-01]
     [9.96452696e-01 3.54730429e-03]
     [2.65506711e-01 7.34493289e-01]
     [9.96550688e-01 3.44931184e-03]
     [5.59000561e-01 4.40999439e-01]
     [7.13671074e-01 2.86328926e-01]
     [2.56171168e-01 7.43828832e-01]
     [3.16746433e-04 9.99683254e-01]
     [3.11928616e-02 9.68807138e-01]
     [9.62623345e-01 3.73766547e-02]
     [8.99401144e-04 9.99100599e-01]
     [2.55804402e-01 7.44195598e-01]
     [9.95918369e-01 4.08163058e-03]
     [7.33623471e-01 2.66376529e-01]
     [9.96456155e-01 3.54384510e-03]
     [9.96361666e-01 3.63833391e-03]
     [9.91903331e-01 8.09666928e-03]
     [5.94457115e-01 4.05542885e-01]
     [2.63017580e-01 7.36982420e-01]
     [9.94836968e-01 5.16303193e-03]
     [4.39656999e-13 1.00000000e+00]
     [9.96422789e-01 3.57721058e-03]
     [3.11863302e-02 9.68813670e-01]
     [9.96533145e-01 3.46685541e-03]
     [9.63006252e-01 3.69937481e-02]
     [9.96480501e-01 3.51949941e-03]
     [7.24064268e-01 2.75935732e-01]
     [9.59184638e-01 4.08153624e-02]
     [9.96438598e-01 3.56140224e-03]
     [2.83605340e-02 9.71639466e-01]
     [9.85575584e-01 1.44244158e-02]
     [9.90012370e-01 9.98762992e-03]
     [9.59195803e-01 4.08041969e-02]
     [2.55804402e-01 7.44195598e-01]
     [9.59320827e-01 4.06791734e-02]
     [1.49661472e-03 9.98503385e-01]
     [1.26461866e-09 9.99999999e-01]
     [2.31707351e-01 7.68292649e-01]
     [5.35007926e-01 4.64992074e-01]
     [2.39019485e-01 7.60980515e-01]
     [1.78396319e-01 8.21603681e-01]
     [9.96436778e-01 3.56322173e-03]
     [9.54644871e-01 4.53551291e-02]
     [9.96499784e-01 3.50021587e-03]
     [9.33033426e-01 6.69665739e-02]
     [2.44420303e-02 9.75557970e-01]
     [3.14871955e-01 6.85128045e-01]
     [9.95707213e-01 4.29278697e-03]
     [9.95875352e-01 4.12464835e-03]
     [9.95875580e-01 4.12441996e-03]
     [9.93672146e-03 9.90063279e-01]
     [2.86587459e-02 9.71341254e-01]
     [3.99528687e-04 9.99600471e-01]
     [2.45168504e-01 7.54831496e-01]
     [1.42921356e-01 8.57078644e-01]
     [2.55804402e-01 7.44195598e-01]
     [9.36886754e-01 6.31132463e-02]
     [6.02337084e-01 3.97662916e-01]
     [3.01437767e-02 9.69856223e-01]
     [6.65239805e-01 3.34760195e-01]
     [9.96360406e-01 3.63959393e-03]
     [2.39183513e-02 9.76081649e-01]
     [9.96448960e-01 3.55104001e-03]
     [9.96099530e-01 3.90047017e-03]
     [3.15238526e-02 9.68476147e-01]
     [3.22605278e-03 9.96773947e-01]
     [3.06242559e-02 9.69375744e-01]
     [9.61582374e-01 3.84176257e-02]
     [2.86587459e-02 9.71341254e-01]
     [7.31530213e-01 2.68469787e-01]
     [9.96425433e-01 3.57456711e-03]
     [2.55804402e-01 7.44195598e-01]
     [9.92494385e-01 7.50561499e-03]
     [9.95451775e-01 4.54822477e-03]
     [2.59025012e-01 7.40974988e-01]
     [9.96507730e-01 3.49227015e-03]
     [9.58823499e-01 4.11765007e-02]
     [9.96259474e-01 3.74052631e-03]
     [3.52507347e-06 9.99996475e-01]
     [9.96297099e-01 3.70290093e-03]
     [9.59367969e-01 4.06320307e-02]
     [9.96471094e-01 3.52890602e-03]
     [1.57090246e-01 8.42909754e-01]
     [2.50070583e-02 9.74992942e-01]
     [2.55804402e-01 7.44195598e-01]
     [9.96398287e-01 3.60171272e-03]
     [9.96608807e-01 3.39119300e-03]
     [6.20625585e-01 3.79374415e-01]
     [9.96376300e-01 3.62370045e-03]
     [9.61417430e-01 3.85825703e-02]
     [2.44989041e-01 7.55010959e-01]
     [9.55820515e-01 4.41794855e-02]
     [2.19923673e-01 7.80076327e-01]
     [1.35027472e-01 8.64972528e-01]
     [6.02224225e-01 3.97775775e-01]
     [2.60064859e-01 7.39935141e-01]
     [2.57996460e-01 7.42003540e-01]
     [2.61179581e-01 7.38820419e-01]
     [4.52821096e-04 9.99547179e-01]
     [2.15759703e-01 7.84240297e-01]
     [9.96536009e-01 3.46399123e-03]
     [9.96449058e-01 3.55094229e-03]
     [2.43448560e-02 9.75655144e-01]
     [3.19731702e-03 9.96802683e-01]
     [9.25709738e-01 7.42902618e-02]
     [2.92011795e-02 9.70798820e-01]
     [1.59653637e-12 1.00000000e+00]
     [2.61981891e-02 9.73801811e-01]
     [7.03241275e-01 2.96758725e-01]
     [3.32173185e-03 9.96678268e-01]
     [9.96404234e-01 3.59576603e-03]
     [2.63868137e-01 7.36131863e-01]
     [9.92042704e-01 7.95729642e-03]
     [9.96530439e-01 3.46956106e-03]
     [9.63608181e-01 3.63918191e-02]
     [9.94950334e-01 5.04966622e-03]
     [9.96171648e-01 3.82835186e-03]
     [9.96449927e-01 3.55007251e-03]
     [1.96354778e-02 9.80364522e-01]
     [3.57698403e-05 9.99964230e-01]
     [5.47081998e-01 4.52918002e-01]
     [2.64452165e-01 7.35547835e-01]
     [2.55129839e-01 7.44870161e-01]
     [3.00656230e-02 9.69934377e-01]
     [2.19413132e-01 7.80586868e-01]
     [1.96206143e-01 8.03793857e-01]
     [9.96449348e-01 3.55065182e-03]
     [4.85210979e-01 5.14789021e-01]
     [9.96449348e-01 3.55065182e-03]
     [9.95850796e-01 4.14920362e-03]
     [9.96426016e-01 3.57398392e-03]
     [9.95443791e-01 4.55620905e-03]
     [9.96023867e-01 3.97613330e-03]
     [5.90951105e-01 4.09048895e-01]
     [2.57159083e-01 7.42840917e-01]
     [2.30075603e-01 7.69924397e-01]
     [9.96251631e-01 3.74836903e-03]
     [8.69488209e-03 9.91305118e-01]
     [9.96452696e-01 3.54730429e-03]
     [9.94425374e-01 5.57462584e-03]
     [9.96436673e-01 3.56332700e-03]
     [9.96363376e-01 3.63662412e-03]
     [9.95856082e-01 4.14391789e-03]
     [3.09364552e-02 9.69063545e-01]
     [9.61242446e-01 3.87575542e-02]
     [1.23487903e-03 9.98765121e-01]
     [9.96436673e-01 3.56332700e-03]
     [1.92273967e-03 9.98077260e-01]
     [9.96387955e-01 3.61204478e-03]
     [2.55788774e-01 7.44211226e-01]
     [9.96067213e-01 3.93278723e-03]
     [1.89017080e-04 9.99810983e-01]
     [9.95870091e-01 4.12990864e-03]
     [1.47641147e-03 9.98523589e-01]
     [1.84803139e-03 9.98151969e-01]
     [9.95906082e-01 4.09391795e-03]
     [9.96514940e-01 3.48506026e-03]
     [7.31925605e-01 2.68074395e-01]
     [9.96487271e-01 3.51272918e-03]
     [9.93615234e-01 6.38476572e-03]
     [9.96058991e-01 3.94100929e-03]
     [9.95960561e-01 4.03943884e-03]
     [7.27001525e-01 2.72998475e-01]
     [6.52341803e-16 1.00000000e+00]
     [4.99299787e-04 9.99500700e-01]
     [9.96424465e-01 3.57553501e-03]
     [2.68475693e-02 9.73152431e-01]
     [9.96487369e-01 3.51263081e-03]
     [7.36882150e-01 2.63117850e-01]
     [7.34953923e-01 2.65046077e-01]
     [9.96376314e-01 3.62368620e-03]
     [5.07291077e-01 4.92708923e-01]
     [9.60704094e-01 3.92959059e-02]
     [9.96449348e-01 3.55065182e-03]
     [9.96297296e-01 3.70270414e-03]
     [2.34809565e-01 7.65190435e-01]
     [9.92042704e-01 7.95729642e-03]
     [9.96159701e-01 3.84029880e-03]
     [2.64784466e-02 9.73521553e-01]
     [8.02532327e-01 1.97467673e-01]
     [9.61061449e-01 3.89385513e-02]
     [6.72734251e-01 3.27265749e-01]
     [4.56783967e-07 9.99999543e-01]
     [2.16574530e-01 7.83425470e-01]
     [2.78653401e-01 7.21346599e-01]
     [9.95929158e-01 4.07084196e-03]
     [2.89208178e-02 9.71079182e-01]
     [3.00479750e-02 9.69952025e-01]
     [9.58925560e-01 4.10744402e-02]
     [2.55804402e-01 7.44195598e-01]
     [1.67429616e-01 8.32570384e-01]
     [9.96290434e-01 3.70956611e-03]
     [7.32231876e-01 2.67768124e-01]
     [9.59195803e-01 4.08041969e-02]
     [9.96023749e-01 3.97625073e-03]
     [9.93733807e-01 6.26619328e-03]
     [4.54181518e-05 9.99954582e-01]
     [1.43776997e-01 8.56223003e-01]
     [9.96464913e-01 3.53508702e-03]
     [9.60813706e-01 3.91862943e-02]
     [9.55720192e-01 4.42798078e-02]
     [1.19763017e-02 9.88023698e-01]
     [9.96041621e-01 3.95837866e-03]
     [9.96556875e-01 3.44312478e-03]
     [8.85857523e-03 9.91141425e-01]
     [9.96294407e-01 3.70559318e-03]
     [9.96045480e-01 3.95452020e-03]
     [9.96452696e-01 3.54730429e-03]
     [4.69163319e-02 9.53083668e-01]]
    

S'aider de l'aide en ligne (http://scikit-learn.org) pour l'utilisation de la méthode.
* Quels sont les paramètres de réglage de la méthode ?


```python
clf.get_params() # affichage des paramètres de la méthode GaussianNB avec get_params
```




    {'priors': None, 'var_smoothing': 1e-09}



J'obtient les paramètres de la méthode **GaussianNB** avec **get_params**. Les paramètres de réglage sont donc **priors** (probabilités antérieures des classes) et **var_smoothing** (lissage de la variance). Ici par défaut, priors=None car les probabilités antérieures des classes ne sont pas ajustés en fonction des données et var_smoothing=1e-09.

Lors de la recherche des paramètres optimaux, je calculerais le taux de précision suivant différentes valeurs de **var_smoothing**.

Les paramètres par défaut dans la fonction **fit** sont **sample_weight=None** car on accorde le même poinds à chaque observations (=1).

L'entraînement du classifieur supervisé se construit à partir de la fonction **fit** sur la base d'entraînement (X_train, Y_train).

La prédiction de la classe utilise la fonction **predict** sur la base de test (X_test).

* Créez une fonction *prediction(labels,pred)* qui renverra le taux de prédiction (en %) pour une prédiction stockée dans *pred* et les valeurs souhaitées stockées dans *label*. On calculera le taux de prédiction de l'apprentissage bayésien naïf sur la base *test*.


```python
def prediction(label,pred):
    # fonction de calcul du taux de prédiction sur la base test (prédiction juste)
    # entrées :
        # - label : valeurs souhaitées (Y_test)
        # - pred : prédiction de X_test
    # sorties : 
        # - taux : taux de prédiction (en %)
    sum = 0
    n = len(label)
    for i,j in zip(label,pred): # boucle simultanée
        if i == j:
            sum += 1  
    taux = sum/n
    return taux
    
# test de la fonction prediction
label = Y_test
pred = clf.predict(X_test)
print('Taux de prédiction du classifieur bayésien naïf : {:.1%}'.format(prediction(label,pred)))
```

    Taux de prédiction du classifieur bayésien naïf : 77.6%
    

On peut également déterminer la précision avec la fonction **accuracy_score** :


```python
from sklearn.metrics import accuracy_score
print('Taux de prédiction du classifieur bayésien naïf : {:.1%}'.format(accuracy_score(Y_test,pred)))
```

    Taux de prédiction du classifieur bayésien naïf : 77.6%
    

Pour évaluer la qualité du prédicteur, on peut aussi calculer la **matrice de confusion** :


```python
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(label, pred)
conf
```




    array([[180,  40],
           [ 40,  97]], dtype=int64)



Avec la matrice de confusion, on voit que le classifieur se trompe peut.

* Répétez les questions précédentes avec la méthode des **arbres de décision**. Pour cela vous aurez besoin des fonctions:


```python
from sklearn import tree
clf = tree.DecisionTreeClassifier() # création du classifieur (arbre de décision)
clf.fit(X_train,Y_train) # apprentissage du classfieur sur les données train
clf.predict(X_test) # prédiction de la classe Survived avec la base de test
```




    array([0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0,
           1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1,
           1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0,
           1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
           1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
           0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
           0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
           0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
           1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1,
           0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
           1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
           0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1,
           0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0,
           1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0,
           1, 0, 0, 0, 1], dtype=int64)




```python
clf.get_params() # affichage des paramètres de la méthode des arbres de décision avec get_params
```




    {'ccp_alpha': 0.0,
     'class_weight': None,
     'criterion': 'gini',
     'max_depth': None,
     'max_features': None,
     'max_leaf_nodes': None,
     'min_impurity_decrease': 0.0,
     'min_impurity_split': None,
     'min_samples_leaf': 1,
     'min_samples_split': 2,
     'min_weight_fraction_leaf': 0.0,
     'presort': 'deprecated',
     'random_state': None,
     'splitter': 'best'}



Par défaut, le paramètre de séparation d'un noeud se calcule avec la fonction gini.

Lors de la recherche des paramètres optimaux, je calculerais le taux de précision suivant différentes valeurs de **criterion** (fonction pour mesurer la qualité d'une scission).


```python
tree.plot_tree(clf) # résultat de l'arbre de décision
plt.show() # affichage de l'arbre
```


![png](output_40_0.png)



```python
# test de la fonction prediction
label = Y_test
pred = clf.predict(X_test)
print('Taux de prédiction de l\'arbre de décision : {:.1%}'.format(prediction(label,pred)))
```

    Taux de prédiction de l'arbre de décision : 76.5%
    

* Mêmes questions ensuite avec la méthode des *k* plus proches voisins et la regression logistique. Voir l'aide en ligne pour un descriptif de l'utilisation des méthodes. A chaque fois, précisez bien quels sont les paramètres de la méthode.


Méthode des **k plus proches voisins** :


```python
from sklearn.neighbors import KNeighborsClassifier
k = 5 # paramètre de réglages pour le nombre de plus proches voisins
clf = KNeighborsClassifier(n_neighbors=k) # création du classifieur (k plus proches voisins)
clf.fit(X_train,Y_train) # apprentissage du classfieur sur les données train
clf.predict(X_test) # prédiction de la classe Survived avec la base de test
```




    array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0,
           1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1,
           1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0,
           1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
           1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0,
           1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
           1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1,
           0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0,
           1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1,
           0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1,
           0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0,
           0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1,
           0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0,
           1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0,
           1, 0, 0, 0, 1], dtype=int64)




```python
clf.get_params() # affichage des paramètres de la méthode des k plus proches voisins avec get_params
```




    {'algorithm': 'auto',
     'leaf_size': 30,
     'metric': 'minkowski',
     'metric_params': None,
     'n_jobs': None,
     'n_neighbors': 5,
     'p': 2,
     'weights': 'uniform'}



Les paramètres de réglage sont **algorithm** (calculer les voisins les plus proches), **leaf_size** (vitesse de construction), **metric** (méthode de calcul de la distance entre deux points), **metric_params** (arguments de mot-clé supplémentaires pour la fonction métrique), **n_jobs** (nombre de travaux parallèles à exécuter pour la recherche de voisins), **n_neighbors** (nombre de plus proches voisins), **p** (paramètre de puissance pour la métrique de minkowski) et **weights** (poids utilisée dans la prédiction). 

Ici, les paramètres sont réglés par défaut :

* algorithm=auto car il tentera de décider de l'algorithme le plus approprié
* leaf_size=30
* metric=minkowski
* metric_params=None
* n_jobs=None
* n_neighbors=5 car c'est la valeur par défaut
* p=2 car on utilise la distance euclidienne
* weights=uniform car on donne le même poids aux observations lors de la prédiction

Lors de la recherche des paramètres optimaux, je calculerais le taux de précision suivant différentes valeurs de **n_neighbors** (nombre de plus proches voisins).

Calcul du taux de prédiction :


```python
# test de la fonction prediction
label = Y_test
pred = clf.predict(X_test)
print('Taux de prédiction du classifieur k plus proches voisins : {:.1%}'.format(prediction(label,pred)))
```

    Taux de prédiction du classifieur k plus proches voisins : 68.1%
    

Méthode de la **regression logistique** :


```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression() # création du classifieur (regression logistique)
clf.fit(X_train,Y_train) # apprentissage du classfieur sur les données train
clf.predict(X_test) # prédiction de la classe Survived avec la base de test
```




    array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0,
           1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0,
           0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0,
           0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0,
           0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0,
           1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
           0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
           1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1,
           0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1,
           0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
           0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0,
           0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
           1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0,
           1, 0, 0, 0, 0], dtype=int64)




```python
clf.get_params() # affichage des paramètres de la méthode de la regression logistique avec get_params
```




    {'C': 1.0,
     'class_weight': None,
     'dual': False,
     'fit_intercept': True,
     'intercept_scaling': 1,
     'l1_ratio': None,
     'max_iter': 100,
     'multi_class': 'auto',
     'n_jobs': None,
     'penalty': 'l2',
     'random_state': None,
     'solver': 'lbfgs',
     'tol': 0.0001,
     'verbose': 0,
     'warm_start': False}



Les paramètres de réglage sont **class_weight**, **dual**, **fit_intercept**, **intercept_scaling**, **n_jobs**, **l1_ratio** , **max_iter** (nombre maximum d'itérations prises pour que les solveurs convergent), **multi_class**, **n_jobs** , **penalty** , **random_state**, **solver** (algorithme à utiliser dans le problème d'optimisation), **tol**, **verbose**  et **warm_start**.

Lors de la recherche des paramètres optimaux, je calculerais le taux de précision suivant **solver** (différents types d'algorithmes pour l'optimisation).

Calcul du taux de prédiction :


```python
# test de la fonction prediction
label = Y_test
pred = clf.predict(X_test)
print('Taux de prédiction du classifieur regression logistique : {:.1%}'.format(prediction(label,pred)))
```

    Taux de prédiction du classifieur regression logistique : 80.4%
    

## Recherche des paramètres optimaux
Pour chacune des méthodes, utilisez la fonction *split* pour séparer la base en deux DataFrames de taille équivalente (*train* et *test*). Recherchez ensuite les paramètres optimaux qui vous donneront le meilleur taux de prédiction sur *test* lorsque le modèle apprend les données de *train*. Pour cela, utilisez une grille de valeurs pour les paramètres et pour chaque valeur de la grille, calculez le taux de prédiction sur *test* lorsque vous apprenez *train*. Retenez les valeurs de paramètres donnant le meilleur taux. Les bornes de valeurs et le pas de la grille sont à déterminer de façon empirique.

**Bayésien naïf**

L'hyperparamètre d'optimisation pour cette algorithme est la valeur de **lissage de la variance** (var_smoothing).


```python
X_train, X_test, Y_train, Y_test = split(df,y,p=0.5) # appel de la fonction split (p=0.5 car 50% des données dans chaque split)
var_smoothing = [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12,1e-13,1e-14,1e-15] # tests de la valeur de lissage de la variance (var_smoothing)
liste_taux = []
for par in var_smoothing:
    clf = GaussianNB(var_smoothing=par) # création du classifieur (bayésien naïf)
    clf.fit(X_train,Y_train) # apprentissage du classfieur sur les données train
    pred = clf.predict(X_test) # prédiction de la classe Survived avec la base de test
    label = Y_test
    taux = prediction(label,pred)
    liste_taux.append(taux)

var_smoothing_max = var_smoothing[liste_taux.index(max(liste_taux))]
print('La valeur de lissage de la variance optimale est',var_smoothing_max)
print('Le meilleur taux de prédiction est : {:.1%}'.format(max(liste_taux)))
```

    La valeur de lissage de la variance optimale est 1e-05
    Le meilleur taux de prédiction est : 79.8%
    

**Arbre de décision**

L'hyperparamètre d'optimisation pour cette algorithme est la **fonction pour mesurer la qualité d'une d'un noeud** (criterion).


```python
X_train, X_test, Y_train, Y_test = split(df,y,p=0.5) # appel de la fonction split (p=0.5 car 50% des données dans chaque split)
criterion = ['gini','entropy'] # paramètres du classifieur à faire varier (différentes fonctions de séparation d'une variable)
liste_taux = []
for par in criterion:
    clf = tree.DecisionTreeClassifier(criterion=par) # création du classifieur (arbre de décision)
    clf.fit(X_train,Y_train) # apprentissage du classfieur sur les données train
    pred = clf.predict(X_test) # prédiction de la classe Survived avec la base de test
    label = Y_test
    taux = prediction(label,pred)
    liste_taux.append(taux)

criterion_max = criterion[liste_taux.index(max(liste_taux))]
print('Le fonction criterion optimale est',criterion_max)
print('Le meilleur taux de prédiction est : {:.1%}'.format(max(liste_taux)))
```

    Le fonction criterion optimale est entropy
    Le meilleur taux de prédiction est : 79.1%
    

**k plus proches voisins**

L'hyperparamètre d'optimisation pour cette algorithme est le nombre de **plus proches voisins** (n_neighbors).


```python
X_train, X_test, Y_train, Y_test = split(df,y,p=0.5) # appel de la fonction split (p=0.5 car 50% des données dans chaque split)
liste_taux = []
for par in range(1,40): # tests du nombre de plus proches voisins
    clf = KNeighborsClassifier(n_neighbors=par) # création du classifieur (k plus proches voisins)
    clf.fit(X_train,Y_train) # apprentissage du classfieur sur les données train
    pred = clf.predict(X_test) # prédiction de la classe Survived avec la base de test
    label = Y_test
    taux = prediction(label,pred)
    liste_taux.append(taux)

n_neighbors_max = liste_taux.index(max(liste_taux))
print('Le nombre de voisins optimal est',n_neighbors_max)
print('Le meilleur taux de prédiction est : {:.1%}'.format(max(liste_taux)))
```

    Le nombre de voisins optimal est 12
    Le meilleur taux de prédiction est : 72.6%
    

**Régression logistique**

L'hyperparamètre d'optimisation pour cette algorithme est l'**algorithme d'optimisation** (solver).


```python
X_train, X_test, Y_train, Y_test = split(df,y,p=0.5) # appel de la fonction split (p=0.5 car 50% des données dans chaque split)
solver = ['lbfgs','liblinear','sag','saga'] # paramètres du classifieur à faire varier (différents types d'algorithmes pour l'optimisation)
liste_taux = []
for par in solver:
    clf = LogisticRegression(solver=par) # création du classifieur (regression logistique)
    clf.fit(X_train,Y_train) # apprentissage du classfieur sur les données train
    pred = clf.predict(X_test) # prédiction de la classe Survived avec la base de test
    label = Y_test
    taux = prediction(label,pred)
    liste_taux.append(taux)

solver_max = solver[liste_taux.index(max(liste_taux))]
print('L\'algorithme optimal est',solver_max)
print('Le meilleur taux de prédiction est : {:.1%}'.format(max(liste_taux)))
```

    L'algorithme optimal est lbfgs
    Le meilleur taux de prédiction est : 78.7%
    

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn("The max_iter was reached which means "
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn("The max_iter was reached which means "
    

## Comparaison des méthodes par validation croisée
Pour chacune des méthodes et en utilisant les paramètres optimaux déterminés précédemment, réalisez une procédure de validation croisée pour calculer le taux de prédiction moyen et sa variance lorsque le nombre de folds est 5. Pour cela, vous aurez besoin de la fonction suivante :


```python
from sklearn.model_selection import cross_val_score
folds = 5 # nombre d'échantillons pour la validation croisée du modèle
```

**Bayésien naïf**


```python
clf = GaussianNB(var_smoothing=var_smoothing_max) # création du classifieur (bayésien naïf) avec le meilleur hyperparamètre
scores_gaussian = cross_val_score(clf,X_train,Y_train,cv=folds) # liste des taux de prédiction
print('Taux de prédiction pour chaque échantillon :',scores_gaussian)
print('Taux de prédiction moyen : {:.1%}'.format(scores_gaussian.mean()))
```

    Taux de prédiction pour chaque échantillon : [0.7752809  0.78651685 0.82022472 0.79775281 0.73033708]
    Taux de prédiction moyen : 78.2%
    

**Arbre de décision**


```python
clf = tree.DecisionTreeClassifier(criterion=criterion_max) # création du classifieur (arbre de décision) avec le meilleur hyperparamètre
scores_decisiontree = cross_val_score(clf,X_train,Y_train,cv=folds) # liste des taux de prédiction
print('Taux de prédiction pour chaque échantillon :',scores_decisiontree)
print('Taux de prédiction moyen : {:.1%}'.format(scores_decisiontree.mean()))
```

    Taux de prédiction pour chaque échantillon : [0.73033708 0.74157303 0.73033708 0.76404494 0.71910112]
    Taux de prédiction moyen : 73.7%
    

**k plus proches voisins**


```python
clf = KNeighborsClassifier(n_neighbors=n_neighbors_max) # création du classifieur (k plus proches voisins) avec le meilleur hyperparamètre
scores_knn = cross_val_score(clf,X_train,Y_train,cv=folds) # liste des taux de prédiction
print('Taux de prédiction pour chaque échantillon :',scores_knn)
print('Taux de prédiction moyen : {:.1%}'.format(scores_knn.mean()))
```

    Taux de prédiction pour chaque échantillon : [0.66292135 0.64044944 0.70786517 0.70786517 0.75280899]
    Taux de prédiction moyen : 69.4%
    

**Régression logistique**


```python
clf = LogisticRegression(solver=solver_max) # création du classifieur (régression logistique) avec le meilleur hyperparamètre
scores_logisticregression = cross_val_score(clf,X_train,Y_train,cv=folds) # liste des taux de prédiction
print('Taux de prédiction pour chaque échantillon :',scores_logisticregression)
print('Taux de prédiction moyen : {:.1%}'.format(scores_logisticregression.mean()))
```

    Taux de prédiction pour chaque échantillon : [0.76404494 0.79775281 0.80898876 0.79775281 0.74157303]
    Taux de prédiction moyen : 78.2%
    

Elle vous donnera les taux de prédiction pour chacun des folds de tests (voir procédure de validation croisée et l'aide en ligne http://scikit-learn.org).

## Affichage des résultats
On souhaite présenter graphiquement le résultat de la validation croisée ainsi que la comparaison des performances des méthodes. Pour cela on utilisera l'environnement *pyplot* de *matplotlib* :
* Pour chacune des 4 méthodes utilisées, affichez un *subplot* qui représentera un diagramme en barres des 5 taux de prédiction correspondants à chacun des folds de la validation croisée (vous pouvez utiliser les paramètres optimaux trouvés précédemment pour cela). Les 4 subplots seront affichés de manière à obtenir 2 niveaux de 2 subplots. Pour cela, vous aurez besoin des 2 fonctions suivantes:


```python
fig, ax = plt.subplots(2,2,figsize=(15, 10))
fig.tight_layout(pad = 3)
folds = [1,2,3,4,5]
ax[0,0].bar(folds,scores_gaussian,align='center',alpha=0.5)
ax[0,0].set_title('Bayésien naïf')
ax[0,0].set_xlabel('Folds')
ax[0,0].set_ylabel('Précision')
ax[0,0].set_ylim(0, 1)
ax[0,1].bar(folds,scores_decisiontree,align='center',alpha=0.5)
ax[0,1].set_title('Arbre de décision')
ax[0,1].set_xlabel('Folds')
ax[0,1].set_ylabel('Précision')
ax[0,1].set_ylim(0, 1)
ax[1,0].bar(folds,scores_knn,align='center',alpha=0.5)
ax[1,0].set_title('k plus proches voisins')
ax[1,0].set_xlabel('Folds')
ax[1,0].set_ylabel('Précision')
ax[1,0].set_ylim(0, 1)
ax[1,1].bar(folds,scores_logisticregression,align='center',alpha=0.5)
ax[1,1].set_title('Régression logistique')
ax[1,1].set_xlabel('Folds')
ax[1,1].set_ylabel('Précision')
ax[1,1].set_ylim(0, 1)
```




    (0.0, 1.0)




![png](output_76_1.png)


* Affichez ensuite un diagramme de 4 barres correspondant aux 4 méthodes où chaque barre représente le taux moyen de prédiction issu de la validation croisée pour une méthode (toujours avec les paramètres optimaux). Comparez.


```python
methodes = ['Bayésien naïf','Arbre de décision','k plus proches voisins','Régression logistique']
avg_taux = [scores_gaussian.mean(),scores_decisiontree.mean(),scores_knn.mean(),scores_logisticregression.mean()]
fig, ax = plt.subplots(figsize=(15, 10))
ax.bar(methodes,avg_taux,align='center',alpha=0.5)
ax.set_title('Diagramme en barre des taux de précision')
ax.set_xlabel('Méthodes')
ax.set_ylabel('Précision')
ax.set_ylim(0, 1)
```




    (0.0, 1.0)




![png](output_78_1.png)



```python
print('Taux de précision moyen pour bayésien naïf : {:.1%}'.format(scores_gaussian.mean()))
print('Taux de précision moyen pour arbre de décision : {:.1%}'.format(scores_decisiontree.mean()))
print('Taux de précision moyen pour k plus proches voisins : {:.1%}'.format(scores_knn.mean()))
print('Taux de précision moyen pour régression logistique : {:.1%}'.format(scores_logisticregression.mean()))
```

    Taux de précision moyen pour bayésien naïf : 78.2%
    Taux de précision moyen pour arbre de décision : 78.8%
    Taux de précision moyen pour k plus proches voisins : 69.4%
    Taux de précision moyen pour régression logistique : 78.2%
    

L'**arbre de décision** est plus performant que les autres méthodes de classification. Le prédicteur se trompe dans envirion 20% des cas.

La méthode des **k plus proches voisins** prédit moins bien la variable *Survived*.

## Bonus
Vous êtes encouragés à :
* créer de nouveaux attributs à partir des attributs nominaux que vous n'aurez pas utilisés
* tester d'autres algorithme
* faire la fonction de calcul de gain pour un attribut numérique. Cette fonction renverra le seuil qui maximise le gain.

Création d'un nouvel attribut *FamilySize* pour la taille d'une famille à partir de *SibSp* et *Parch* :


```python
df = pd.read_csv('titanic.csv') # rechargement du jeu de données
df = df.drop(['PassengerId','Name','Ticket','Cabin','Embarked'], axis=1) # je supprime les variables inutiles
df = pd.get_dummies(df, columns=['Pclass','Sex']) # convertion de la variable Pclass et Sex en type quantitatif binaire
df = df.fillna(df['Age'].mean()) # on remplace les valeurs nan par la moyenne
```


```python
df['FamilySize'] = df['SibSp'] +  df['Parch'] + 1 
print(df.head()) # affichage des 5 premières lignes
print(df.columns) # on affiche la nouvelle liste des variables d'apprentissage (features) et la classe (target)
```

       Survived   Age  SibSp  Parch     Fare  Pclass_1  Pclass_2  Pclass_3  \
    0         0  22.0      1      0   7.2500         0         0         1   
    1         1  38.0      1      0  71.2833         1         0         0   
    2         1  26.0      0      0   7.9250         0         0         1   
    3         1  35.0      1      0  53.1000         1         0         0   
    4         0  35.0      0      0   8.0500         0         0         1   
    
       Sex_female  Sex_male  FamilySize  
    0           0         1           2  
    1           1         0           2  
    2           1         0           1  
    3           1         0           2  
    4           0         1           1  
    Index(['Survived', 'Age', 'SibSp', 'Parch', 'Fare', 'Pclass_1', 'Pclass_2',
           'Pclass_3', 'Sex_female', 'Sex_male', 'FamilySize'],
          dtype='object')
    

Algorithme de l'**arbre de décision** avec le nouvel attribut *FamilySize*.


```python
y = 'Survived' # target
X_train, X_test, Y_train, Y_test = split(df,y,p=0.8) # appel de la fonction split
clf = tree.DecisionTreeClassifier(criterion=criterion_max) # création du classifieur (arbre de décision) avec le meilleur hyperparamètre
folds = 5 # nombre d'échantillons pour la validation croisée du modèle
scores_decisiontree = cross_val_score(clf,X_train,Y_train,cv=folds) # liste des taux de prédiction
print('Taux de prédiction moyen : {:.1%}'.format(scores_decisiontree.mean()))
```

    Taux de prédiction moyen : 78.8%
    