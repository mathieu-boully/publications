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
!pip install statistics
```

    Requirement already satisfied: nbconvert in c:\programdata\anaconda3\lib\site-packages (5.6.1)
    Requirement already satisfied: mistune<2,>=0.8.1 in c:\programdata\anaconda3\lib\site-packages (from nbconvert) (0.8.4)
    Requirement already satisfied: nbformat>=4.4 in c:\programdata\anaconda3\lib\site-packages (from nbconvert) (5.0.7)
    Requirement already satisfied: pandocfilters>=1.4.1 in c:\programdata\anaconda3\lib\site-packages (from nbconvert) (1.4.2)
    Requirement already satisfied: defusedxml in c:\programdata\anaconda3\lib\site-packages (from nbconvert) (0.6.0)
    Requirement already satisfied: entrypoints>=0.2.2 in c:\programdata\anaconda3\lib\site-packages (from nbconvert) (0.3)
    Requirement already satisfied: traitlets>=4.2 in c:\programdata\anaconda3\lib\site-packages (from nbconvert) (4.3.3)
    Requirement already satisfied: jupyter-core in c:\programdata\anaconda3\lib\site-packages (from nbconvert) (4.6.3)
    Requirement already satisfied: testpath in c:\programdata\anaconda3\lib\site-packages (from nbconvert) (0.4.4)
    Requirement already satisfied: bleach in c:\programdata\anaconda3\lib\site-packages (from nbconvert) (3.1.5)
    Requirement already satisfied: jinja2>=2.4 in c:\programdata\anaconda3\lib\site-packages (from nbconvert) (2.11.2)
    Requirement already satisfied: pygments in c:\programdata\anaconda3\lib\site-packages (from nbconvert) (2.6.1)
    Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in c:\programdata\anaconda3\lib\site-packages (from nbformat>=4.4->nbconvert) (3.2.0)
    Requirement already satisfied: ipython-genutils in c:\programdata\anaconda3\lib\site-packages (from nbformat>=4.4->nbconvert) (0.2.0)
    Requirement already satisfied: six in c:\programdata\anaconda3\lib\site-packages (from traitlets>=4.2->nbconvert) (1.15.0)
    Requirement already satisfied: decorator in c:\programdata\anaconda3\lib\site-packages (from traitlets>=4.2->nbconvert) (4.4.2)
    Requirement already satisfied: pywin32>=1.0; sys_platform == "win32" in c:\programdata\anaconda3\lib\site-packages (from jupyter-core->nbconvert) (227)
    Requirement already satisfied: packaging in c:\programdata\anaconda3\lib\site-packages (from bleach->nbconvert) (20.4)
    Requirement already satisfied: webencodings in c:\programdata\anaconda3\lib\site-packages (from bleach->nbconvert) (0.5.1)
    Requirement already satisfied: MarkupSafe>=0.23 in c:\programdata\anaconda3\lib\site-packages (from jinja2>=2.4->nbconvert) (1.1.1)
    Requirement already satisfied: pyrsistent>=0.14.0 in c:\programdata\anaconda3\lib\site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.4->nbconvert) (0.16.0)
    Requirement already satisfied: setuptools in c:\programdata\anaconda3\lib\site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.4->nbconvert) (49.2.0.post20200714)
    Requirement already satisfied: attrs>=17.4.0 in c:\programdata\anaconda3\lib\site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.4->nbconvert) (19.3.0)
    Requirement already satisfied: pyparsing>=2.0.2 in c:\programdata\anaconda3\lib\site-packages (from packaging->bleach->nbconvert) (2.4.7)
    Requirement already satisfied: pandas in c:\programdata\anaconda3\lib\site-packages (1.0.5)
    Requirement already satisfied: python-dateutil>=2.6.1 in c:\programdata\anaconda3\lib\site-packages (from pandas) (2.8.1)
    Requirement already satisfied: numpy>=1.13.3 in c:\programdata\anaconda3\lib\site-packages (from pandas) (1.18.5)
    Requirement already satisfied: pytz>=2017.2 in c:\programdata\anaconda3\lib\site-packages (from pandas) (2020.1)
    Requirement already satisfied: six>=1.5 in c:\programdata\anaconda3\lib\site-packages (from python-dateutil>=2.6.1->pandas) (1.15.0)
    Requirement already satisfied: numpy in c:\programdata\anaconda3\lib\site-packages (1.18.5)
    

    ERROR: Could not find a version that satisfies the requirement matplotlib.pyplot (from versions: none)
    ERROR: No matching distribution found for matplotlib.pyplot
    

    Requirement already satisfied: matplotlib in c:\programdata\anaconda3\lib\site-packages (3.2.2)
    Requirement already satisfied: cycler>=0.10 in c:\programdata\anaconda3\lib\site-packages (from matplotlib) (0.10.0)
    Requirement already satisfied: numpy>=1.11 in c:\programdata\anaconda3\lib\site-packages (from matplotlib) (1.18.5)
    Requirement already satisfied: python-dateutil>=2.1 in c:\programdata\anaconda3\lib\site-packages (from matplotlib) (2.8.1)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in c:\programdata\anaconda3\lib\site-packages (from matplotlib) (2.4.7)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\programdata\anaconda3\lib\site-packages (from matplotlib) (1.2.0)
    Requirement already satisfied: six in c:\programdata\anaconda3\lib\site-packages (from cycler>=0.10->matplotlib) (1.15.0)
    

    ERROR: Could not find a version that satisfies the requirement csv (from versions: none)
    ERROR: No matching distribution found for csv
    

    Requirement already satisfied: statistics in c:\programdata\anaconda3\lib\site-packages (1.0.3.5)
    Requirement already satisfied: docutils>=0.3 in c:\programdata\anaconda3\lib\site-packages (from statistics) (0.16)
    


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
import csv
import statistics
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
                                        Name   Sex  Ticket Cabin Embarked
    count                                891   891     891   204      889
    unique                               891     2     681   147        3
    top     Abbott, Mrs. Stanton (Rosa Hunt)  male  347082    G6        S
    freq                                   1   577       7     4      644
    




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



74% des personnes qui ont survécus sont des femmes contre 19% d'hommes.

## Nettoyage de la base

Les colonnes qui me semblent inutiles pour la base d'apprentissage sont *PassengerId*, *SibSp* (=nombre d’époux, de frères ou de soeurs présents à bord), *Parch* (=nombre de parents ou d’enfants présents à bord), *Name*, *Ticket* et *Cabin*.


```python
df = df.drop(['PassengerId','SibSp','Parch','Name','Ticket','Cabin'], axis=1) # je supprime les variables inutiles
print(df.columns) # on affiche la nouvelle liste des variables d'apprentissage (features) et la classe (target)
```

    Index(['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked'], dtype='object')
    

Nous retenons donc 4 variables explicatives (2 quantitatives et 2 qualitatives) ainsi que la classe à prédire *Survived*.


```python
df = pd.get_dummies(df, columns=['Pclass','Sex', 'Embarked']) # convertion de la variable Pclass, Sex et Embarked en type quantitatif binaire
```

On remarque des valeurs manquantes pour la variable *Age*


```python
print(df.isnull().sum())
df = df.fillna(df['Age'].mean()) # on remplace les valeurs nan par la moyenne
print(df.head()) # affichage du nouveau DataFrame nettoyé
df.info() # les valeurs nulles ont été remplacés par la moyenne
```

    Survived        0
    Age           177
    Fare            0
    Pclass_1        0
    Pclass_2        0
    Pclass_3        0
    Sex_female      0
    Sex_male        0
    Embarked_C      0
    Embarked_Q      0
    Embarked_S      0
    dtype: int64
       Survived   Age     Fare  Pclass_1  Pclass_2  Pclass_3  Sex_female  \
    0         0  22.0   7.2500         0         0         1           0   
    1         1  38.0  71.2833         1         0         0           1   
    2         1  26.0   7.9250         0         0         1           1   
    3         1  35.0  53.1000         1         0         0           1   
    4         0  35.0   8.0500         0         0         1           0   
    
       Sex_male  Embarked_C  Embarked_Q  Embarked_S  
    0         1           0           0           1  
    1         0           1           0           0  
    2         0           0           0           1  
    3         0           0           0           1  
    4         1           0           0           1  
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 11 columns):
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
     8   Embarked_C  891 non-null    uint8  
     9   Embarked_Q  891 non-null    uint8  
     10  Embarked_S  891 non-null    uint8  
    dtypes: float64(2), int64(1), uint8(8)
    memory usage: 28.0 KB
    

## Création d'une base d'apprentissage et d'une base de test

Avec la fonction **split** à la "main".


```python
def split(df,p=0.6): 
    # fonction split qui sépare et retourne deux DataFrames selon la valeur de p (=train_size)
    # entrées : 
        # - df : DataFrame complet
        # - p : taille de train_size (par défaut p=0.6)
    # sorties : 
        # - train : échantillon d'entraînement
        # - test : : échantillon de test
    train = df.sample(frac=p,replace=False)
    test = df.sample(frac=(1-p),replace=False)
    return train, test

# test de la fonction split (taille 60%, 40%)
train, test = split(df,0.6)
print('Nb lignes échantillon d\'entraînement : ', len(train)) # échantilllon pour entrainer le modèle
print('Nb lignes échantillon de test : ', len(test)) # sert à évaluer la prformance du modèle
```

    Nb lignes échantillon d'entraînement :  535
    Nb lignes échantillon de test :  356
    

En utilisant la fonction **train_test_split** de scikit-learn.


```python
from sklearn.model_selection import train_test_split # importation de la fonction train_test_split
def split(df,y,p=0.6): 
    # fonction split qui sépare et retourne 2 DataFrames selon la valeur de p (=train_size)
    # entrées : 
        # - df : DataFrame complet
        # - y : target
        # - p : taille de train_size (par défaut p=0.6)
    # sorties : 
        # - train_test_split : fonction scikit-learn qui renvoie X_train, X_test, Y_train et Y_test    
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
    # fonction pour calculer le log2 utilisé dans la fonction entropie
    return log(x)/log(2)

def entropie(n1,n2):
    # fonction entropie qui retourne la valeur de l'entropie
    # entrées :
        # - n1 : nombre de valeurs dans la classe 0 (Survived=0)
        # - n2 : nombre de valeurs dans la classe 1 (Survived=1)
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

## Fonction gain d'entropie


```python
def gain(S,l):
    # fonction gain d'entropie qui retourne la valeur du gain d'entropie d'un attribut
    # entrées :
        # - S : nb observations dans chaque classe
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
```

Gain d'entropie pour la variable *Sexe* (variable binaire) :


```python
# test de la fonction gain d'entropie sur le Sexe
#print(df.groupby(['Sex_male']).sum())
#print(df.groupby(['Sex_female']).sum())

n0=len(df[df['Survived']==0])
n1=len(df[df['Survived']==1])

sf0=109 # nombre de décès chez les femmes
sf1=233 # nombre de survivants chez les femmes

sm0=233 # nombre de décès chez les hommes
sm1=109 # nombre de survivants chez les hommes

print('Gain d\'entropie pour la variable Sexe : ',gain((n0,n1),[(sf0,sf1),(sm0,sm1)]))
```

    Gain d'entropie pour la variable Sexe :  0.26751352031370584
    

Gain d'entropie pour la variable *Pclass* (variable binaire) :


```python
# test de la fonction gain d'entropie sur Pclass
#print(df.groupby(['Pclass_1']).sum())
#print(df.groupby(['Pclass_2']).sum())
#print(df.groupby(['Pclass_3']).sum())

n0=len(df[df['Survived']==0])
n1=len(df[df['Survived']==1])

s10=206 # nombre de décès pour la classe 1
s11=136 # nombre de survivants pour la classe 1

s20=255 # nombre de décès pour la classe 2
s21=87 # nombre de survivants pour la classe 2

s30=223 # nombre de décès pour la classe 3
s31=119 # nombre de survivants pour la classe 3

print('Gain d\'entropie pour la variable Pclass : ',gain((n0,n1),[(s10,s11),(s20,s21),(s30,s31)]))
```

    Gain d'entropie pour la variable Pclass :  -0.0833127751270698
    

Gain d'entropie pour la variable *Embarked* (variable binaire) :


```python
# test de la fonction gain d'entropie sur Embarked
#print(df.groupby(['Embarked_C']).sum())
#print(df.groupby(['Embarked_Q']).sum())
#print(df.groupby(['Embarked_S']).sum())

n0=len(df[df['Survived']==0])
n1=len(df[df['Survived']==1])

s10=249 # nombre de décès pour le port Cherbourg
s11=93 # nombre de survivants pour la classe Cherbourg

s20=312 # nombre de décès pour la classe Queenstown
s21=30 # nombre de survivants pour la classe Queenstown

s30=125 # nombre de décès pour la classe Southampton
s31=217 # nombre de survivants pour la classe Southampton

print('Gain d\'entropie pour la variable Embarked : ',gain((n0,n1),[(s10,s11),(s20,s21),(s30,s31)]))
```

    Gain d'entropie pour la variable Embarked :  0.10851646912767599
    

Le meilleur attribut pour démarrer un arbre de décision est la variable *Sexe* car le gain d'entropie (=0.27) est le plus élevé, la séparation n'est pas équilibrée (dissociation des deux classes, c'est un bon classifieur).

## Familiarisation avec quelques méthodes du package *scikit-learn*+
On utilisera les fonctions d'apprentissage bayésien naïf de *scikit-learn* pour prédire la classe de la base *test* créée précédemment lorsque le modèle prédictif est calculé à partir de la base d'apprentissage *train*. Pour cela, vous aurez besoin des fonctions:


```python
from sklearn.naive_bayes import GaussianNB # importation du module GaussianNB
clf = GaussianNB() # création du classifieur (bayésien naïf)
clf.fit(X_train,Y_train) # apprentissage du classfieur sur les données train
print(clf.predict(X_test)) # prédiction de la classe Survived avec la base de test
print(clf.predict_proba(X_test)) # probabilité d'appartenance à une classe
```

    [1 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 1 1 0 1 0 1 0 0 1 0 0 1 1 0 1 0 1 0
     0 1 1 0 0 1 0 0 0 0 0 1 0 1 0 0 1 0 0 1 1 1 0 0 1 1 1 0 1 0 0 0 1 0 1 0 0
     1 0 1 0 1 0 1 0 0 0 0 1 0 1 0 1 0 0 0 0 1 1 0 0 0 1 0 1 0 0 1 1 0 0 0 0 0
     0 0 1 0 1 0 0 0 0 0 0 1 1 0 1 1 0 1 0 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
     0 0 0 0 1 0 0 0 1 0 0 0 0 0 1 0 1 0 0 1 0 0 1 0 0 0 1 1 0 0 0 0 1 0 1 1 0
     0 0 0 1 1 0 0 1 0 1 0 0 1 1 0 1 0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0
     0 1 1 1 0 0 0 1 0 0 0 0 1 1 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 1 1 1 1 0 0 1
     0 1 1 1 0 1 1 1 1 1 1 0 0 0 0 1 1 0 0 1 0 0 1 1 0 0 1 0 0 0 1 0 0 0 0 0 1
     0 1 1 0 0 0 1 1 0 1 0 0 1 0 0 1 1 0 0 0 0 0 0 0 1 1 0 1 0 1 0 0 1 0 1 0 0
     1 1 0 0 0 0 1 1 1 0 0 0 1 0 1 1 0 0 0 0 0 0 1 0]
    [[1.50432054e-02 9.84956795e-01]
     [8.84635408e-01 1.15364592e-01]
     [9.95123077e-01 4.87692275e-03]
     [8.69414859e-01 1.30585141e-01]
     [9.19043116e-01 8.09568837e-02]
     [8.16157121e-01 1.83842879e-01]
     [9.97392667e-01 2.60733311e-03]
     [9.97007519e-01 2.99248134e-03]
     [9.97391546e-01 2.60845423e-03]
     [9.45650305e-01 5.43496953e-02]
     [8.77499865e-01 1.22500135e-01]
     [3.78221198e-01 6.21778802e-01]
     [1.42831243e-01 8.57168757e-01]
     [6.04503433e-06 9.99993955e-01]
     [9.97389084e-01 2.61091552e-03]
     [9.44219086e-01 5.57809137e-02]
     [9.97351397e-01 2.64860251e-03]
     [8.75889981e-01 1.24110019e-01]
     [8.05239278e-01 1.94760722e-01]
     [1.98878963e-02 9.80112104e-01]
     [6.32969779e-03 9.93670302e-01]
     [9.97388036e-01 2.61196416e-03]
     [1.20716580e-02 9.87928342e-01]
     [9.96829876e-01 3.17012438e-03]
     [2.27433202e-01 7.72566798e-01]
     [9.67264722e-01 3.27352779e-02]
     [9.97096309e-01 2.90369144e-03]
     [1.63335590e-01 8.36664410e-01]
     [9.97408398e-01 2.59160202e-03]
     [8.74032069e-01 1.25967931e-01]
     [2.10067245e-01 7.89932755e-01]
     [2.24221571e-01 7.75778429e-01]
     [8.88846109e-01 1.11153891e-01]
     [1.24214677e-02 9.87578532e-01]
     [9.97498876e-01 2.50112410e-03]
     [2.71388737e-05 9.99972861e-01]
     [9.68252452e-01 3.17475481e-02]
     [9.96052367e-01 3.94763303e-03]
     [1.00895640e-01 8.99104360e-01]
     [1.00895640e-01 8.99104360e-01]
     [9.97113391e-01 2.88660948e-03]
     [9.68414655e-01 3.15853453e-02]
     [1.21131283e-02 9.87886872e-01]
     [9.46493612e-01 5.35063880e-02]
     [8.96603997e-01 1.03396003e-01]
     [8.89035706e-01 1.10964294e-01]
     [9.49193468e-01 5.08065319e-02]
     [5.80148352e-01 4.19851648e-01]
     [2.24779473e-01 7.75220527e-01]
     [9.96913126e-01 3.08687430e-03]
     [1.36637928e-02 9.86336207e-01]
     [9.97392052e-01 2.60794754e-03]
     [9.97270047e-01 2.72995324e-03]
     [1.53411545e-01 8.46588455e-01]
     [9.47876633e-01 5.21233674e-02]
     [9.97398719e-01 2.60128059e-03]
     [1.43810618e-02 9.85618938e-01]
     [2.53646794e-04 9.99746353e-01]
     [7.52721732e-03 9.92472783e-01]
     [9.43788817e-01 5.62111835e-02]
     [9.97501463e-01 2.49853733e-03]
     [1.40466261e-02 9.85953374e-01]
     [1.50214244e-02 9.84978576e-01]
     [4.17587483e-01 5.82412517e-01]
     [9.95029755e-01 4.97024549e-03]
     [5.71510473e-10 9.99999999e-01]
     [9.96962567e-01 3.03743321e-03]
     [6.36164314e-01 3.63835686e-01]
     [9.97354705e-01 2.64529506e-03]
     [2.26528122e-01 7.73471878e-01]
     [9.97052694e-01 2.94730640e-03]
     [2.02105177e-01 7.97894823e-01]
     [9.97173998e-01 2.82600179e-03]
     [9.96962567e-01 3.03743321e-03]
     [4.13652117e-04 9.99586348e-01]
     [9.97101319e-01 2.89868092e-03]
     [1.47540986e-02 9.85245901e-01]
     [9.35095969e-01 6.49040309e-02]
     [7.34908100e-33 1.00000000e+00]
     [9.44307994e-01 5.56920062e-02]
     [1.47120362e-02 9.85287964e-01]
     [9.97492511e-01 2.50748882e-03]
     [9.47556616e-01 5.24433844e-02]
     [9.97392667e-01 2.60733311e-03]
     [9.97175116e-01 2.82488362e-03]
     [1.00895640e-01 8.99104360e-01]
     [9.97046408e-01 2.95359163e-03]
     [1.42117186e-02 9.85788281e-01]
     [9.97348451e-01 2.65154925e-03]
     [2.38137892e-01 7.61862108e-01]
     [9.97214380e-01 2.78562027e-03]
     [6.81741221e-01 3.18258779e-01]
     [9.58511892e-01 4.14881081e-02]
     [9.95536871e-01 4.46312854e-03]
     [2.33138650e-01 7.66861350e-01]
     [1.43263114e-01 8.56736886e-01]
     [8.62036359e-01 1.37963641e-01]
     [9.95754207e-01 4.24579263e-03]
     [8.73230009e-01 1.26769991e-01]
     [2.10742151e-02 9.78925785e-01]
     [8.66953514e-01 1.33046486e-01]
     [2.25110023e-01 7.74889977e-01]
     [9.97243877e-01 2.75612252e-03]
     [9.97095985e-01 2.90401548e-03]
     [1.67547904e-01 8.32452096e-01]
     [1.80655049e-01 8.19344951e-01]
     [9.97392667e-01 2.60733311e-03]
     [9.94971911e-01 5.02808907e-03]
     [9.97091705e-01 2.90829504e-03]
     [9.65786655e-01 3.42133452e-02]
     [9.95732839e-01 4.26716121e-03]
     [9.67579862e-01 3.24201377e-02]
     [9.44848752e-01 5.51512477e-02]
     [1.72001824e-05 9.99982800e-01]
     [9.97385981e-01 2.61401875e-03]
     [1.53411545e-01 8.46588455e-01]
     [9.48811103e-01 5.11888970e-02]
     [9.97390543e-01 2.60945744e-03]
     [9.42908857e-01 5.70911432e-02]
     [9.97492894e-01 2.50710597e-03]
     [9.39317590e-01 6.06824098e-02]
     [9.69350312e-01 3.06496884e-02]
     [2.42142976e-01 7.57857024e-01]
     [4.89101396e-08 9.99999951e-01]
     [8.38627942e-01 1.61372058e-01]
     [5.63953846e-03 9.94360462e-01]
     [1.71981161e-01 8.28018839e-01]
     [9.96226608e-01 3.77339217e-03]
     [1.49566154e-02 9.85043385e-01]
     [9.97274244e-01 2.72575567e-03]
     [1.99047034e-01 8.00952966e-01]
     [3.54682948e-01 6.45317052e-01]
     [8.78805754e-01 1.21194246e-01]
     [9.95579974e-01 4.42002616e-03]
     [9.97463253e-01 2.53674694e-03]
     [9.97303102e-01 2.69689761e-03]
     [9.93701116e-01 6.29888435e-03]
     [9.97238983e-01 2.76101711e-03]
     [9.68381203e-01 3.16187975e-02]
     [9.97470885e-01 2.52911516e-03]
     [4.66765987e-06 9.99995332e-01]
     [8.84635408e-01 1.15364592e-01]
     [9.97303067e-01 2.69693287e-03]
     [9.97495023e-01 2.50497661e-03]
     [8.68594577e-01 1.31405423e-01]
     [9.69054807e-01 3.09451927e-02]
     [9.95549986e-01 4.45001409e-03]
     [9.95557576e-01 4.44242415e-03]
     [9.45437002e-01 5.45629981e-02]
     [9.97053960e-01 2.94604006e-03]
     [8.43714719e-01 1.56285281e-01]
     [9.96436038e-01 3.56396236e-03]
     [1.58227097e-01 8.41772903e-01]
     [9.97379941e-01 2.62005931e-03]
     [9.97178214e-01 2.82178611e-03]
     [9.97329436e-01 2.67056429e-03]
     [1.54924049e-01 8.45075951e-01]
     [9.97270047e-01 2.72995324e-03]
     [9.46493612e-01 5.35063880e-02]
     [9.97474900e-01 2.52510032e-03]
     [9.97392052e-01 2.60794754e-03]
     [9.66958278e-01 3.30417218e-02]
     [3.68058525e-01 6.31941475e-01]
     [9.97097729e-01 2.90227117e-03]
     [1.40924820e-02 9.85907518e-01]
     [9.95502681e-01 4.49731866e-03]
     [9.49518621e-01 5.04813786e-02]
     [1.61392091e-01 8.38607909e-01]
     [9.46924956e-01 5.30750439e-02]
     [9.47677690e-01 5.23223104e-02]
     [3.31171999e-03 9.96688280e-01]
     [9.97178351e-01 2.82164922e-03]
     [9.97392052e-01 2.60794754e-03]
     [9.95557576e-01 4.44242415e-03]
     [2.28311967e-01 7.71688033e-01]
     [1.29250855e-02 9.87074914e-01]
     [9.95308709e-01 4.69129131e-03]
     [9.69197113e-01 3.08028867e-02]
     [6.84153893e-01 3.15846107e-01]
     [9.96437977e-01 3.56202340e-03]
     [2.07640857e-02 9.79235914e-01]
     [9.97471505e-01 2.52849509e-03]
     [1.53438807e-01 8.46561193e-01]
     [1.47540986e-02 9.85245901e-01]
     [6.62888574e-01 3.37111426e-01]
     [9.95324626e-01 4.67537393e-03]
     [9.96214104e-01 3.78589580e-03]
     [9.97242786e-01 2.75721355e-03]
     [1.77805415e-05 9.99982219e-01]
     [1.53411545e-01 8.46588455e-01]
     [9.46071199e-01 5.39288011e-02]
     [9.97416497e-01 2.58350327e-03]
     [1.54639495e-03 9.98453605e-01]
     [9.97053266e-01 2.94673407e-03]
     [1.44675821e-02 9.85532418e-01]
     [9.97392052e-01 2.60794754e-03]
     [9.97445545e-01 2.55445488e-03]
     [8.51771857e-03 9.91482281e-01]
     [1.53438807e-01 8.46561193e-01]
     [9.96909008e-01 3.09099156e-03]
     [1.47198439e-01 8.52801561e-01]
     [9.97497732e-01 2.50226845e-03]
     [9.45437002e-01 5.45629981e-02]
     [9.97392667e-01 2.60733311e-03]
     [5.91568872e-01 4.08431128e-01]
     [9.97007237e-01 2.99276335e-03]
     [9.97485597e-01 2.51440281e-03]
     [3.54473957e-01 6.45526043e-01]
     [1.46818937e-02 9.85318106e-01]
     [9.41261075e-01 5.87389248e-02]
     [9.48066332e-01 5.19336675e-02]
     [9.97248111e-01 2.75188851e-03]
     [9.95124421e-01 4.87557948e-03]
     [9.97392052e-01 2.60794754e-03]
     [1.36916164e-02 9.86308384e-01]
     [9.97099178e-01 2.90082200e-03]
     [9.95557576e-01 4.44242415e-03]
     [9.45069374e-01 5.49306257e-02]
     [9.38637578e-01 6.13624217e-02]
     [9.97054423e-01 2.94557714e-03]
     [9.97007011e-01 2.99298870e-03]
     [9.97248361e-01 2.75163884e-03]
     [9.96028693e-01 3.97130686e-03]
     [8.82535585e-02 9.11746442e-01]
     [1.55416237e-01 8.44583763e-01]
     [2.19644178e-01 7.80355822e-01]
     [9.95557422e-01 4.44257763e-03]
     [9.97463253e-01 2.53674694e-03]
     [9.93209161e-01 6.79083934e-03]
     [1.53438807e-01 8.46561193e-01]
     [5.70047346e-01 4.29952654e-01]
     [9.97389084e-01 2.61091552e-03]
     [8.85286382e-01 1.14713618e-01]
     [9.97429806e-01 2.57019424e-03]
     [1.60279546e-02 9.83972045e-01]
     [4.20006576e-12 1.00000000e+00]
     [9.97392667e-01 2.60733311e-03]
     [9.68345016e-01 3.16549836e-02]
     [8.80850907e-01 1.19149093e-01]
     [9.34518797e-01 6.54812027e-02]
     [9.97434149e-01 2.56585133e-03]
     [8.84597545e-01 1.15402455e-01]
     [6.00638483e-01 3.99361517e-01]
     [1.41630456e-01 8.58369544e-01]
     [9.97417117e-01 2.58288341e-03]
     [1.31819744e-02 9.86818026e-01]
     [9.46113523e-01 5.38864769e-02]
     [5.91917499e-36 1.00000000e+00]
     [9.96907680e-01 3.09232001e-03]
     [9.97171236e-01 2.82876409e-03]
     [9.49891844e-01 5.01081562e-02]
     [8.84597545e-01 1.15402455e-01]
     [6.04156563e-06 9.99993958e-01]
     [1.00189252e-03 9.98998107e-01]
     [2.43710233e-01 7.56289767e-01]
     [1.41233351e-02 9.85876665e-01]
     [9.97398719e-01 2.60128059e-03]
     [9.97238544e-01 2.76145643e-03]
     [1.04681934e-03 9.98953181e-01]
     [9.47188141e-01 5.28118587e-02]
     [1.29641545e-04 9.99870358e-01]
     [2.24495878e-02 9.77550412e-01]
     [1.65854430e-03 9.98341456e-01]
     [9.95324626e-01 4.67537393e-03]
     [8.65309722e-04 9.99134690e-01]
     [2.32330444e-01 7.67669556e-01]
     [1.31694312e-01 8.68305688e-01]
     [4.90956085e-11 1.00000000e+00]
     [2.22352409e-01 7.77647591e-01]
     [4.07550684e-08 9.99999959e-01]
     [9.97171236e-01 2.82876409e-03]
     [9.97451854e-01 2.54814593e-03]
     [9.68344763e-01 3.16552370e-02]
     [9.97335727e-01 2.66427324e-03]
     [2.25110023e-01 7.74889977e-01]
     [1.47727251e-01 8.52272749e-01]
     [9.46254639e-01 5.37453609e-02]
     [9.40485970e-01 5.95140295e-02]
     [2.21677396e-01 7.78322604e-01]
     [9.44219086e-01 5.57809137e-02]
     [9.42960978e-01 5.70390217e-02]
     [5.72216495e-08 9.99999943e-01]
     [2.21816708e-01 7.78183292e-01]
     [9.95893437e-01 4.10656291e-03]
     [9.97174451e-01 2.82554916e-03]
     [2.41139355e-02 9.75886065e-01]
     [9.08231560e-01 9.17684398e-02]
     [8.34250095e-01 1.65749905e-01]
     [9.96909008e-01 3.09099156e-03]
     [3.17662288e-02 9.68233771e-01]
     [9.97273715e-01 2.72628520e-03]
     [8.67296439e-01 1.32703561e-01]
     [8.41053755e-01 1.58946245e-01]
     [9.95557576e-01 4.44242415e-03]
     [9.49809821e-01 5.01901792e-02]
     [5.51339912e-06 9.99994487e-01]
     [9.97385981e-01 2.61401875e-03]
     [2.12568928e-01 7.87431072e-01]
     [8.54947757e-04 9.99145052e-01]
     [9.97302578e-01 2.69742154e-03]
     [9.68345016e-01 3.16549836e-02]
     [9.45232481e-01 5.47675186e-02]
     [1.53411545e-01 8.46588455e-01]
     [4.01872306e-12 1.00000000e+00]
     [9.97354410e-01 2.64559034e-03]
     [1.80884064e-01 8.19115936e-01]
     [9.97428594e-01 2.57140555e-03]
     [9.94507802e-01 5.49219799e-03]
     [1.50902127e-02 9.84909787e-01]
     [9.11042916e-01 8.89570841e-02]
     [9.97271143e-01 2.72885736e-03]
     [5.95501779e-04 9.99404498e-01]
     [5.80832239e-03 9.94191678e-01]
     [9.43788817e-01 5.62111835e-02]
     [9.97325333e-01 2.67466742e-03]
     [9.46483752e-01 5.35162476e-02]
     [9.97406053e-01 2.59394699e-03]
     [9.68040396e-01 3.19596039e-02]
     [9.68207762e-01 3.17922380e-02]
     [9.97371873e-01 2.62812744e-03]
     [1.30619565e-02 9.86938043e-01]
     [1.40316386e-02 9.85968361e-01]
     [9.97179167e-01 2.82083256e-03]
     [1.07230179e-04 9.99892770e-01]
     [9.97453759e-01 2.54624064e-03]
     [2.09925553e-01 7.90074447e-01]
     [9.97175116e-01 2.82488362e-03]
     [9.44219086e-01 5.57809137e-02]
     [2.35846235e-03 9.97641538e-01]
     [9.97354584e-01 2.64541590e-03]
     [3.66097110e-03 9.96339029e-01]
     [9.97329864e-01 2.67013649e-03]
     [8.73298517e-01 1.26701483e-01]
     [1.78255231e-02 9.82174477e-01]
     [1.37208633e-02 9.86279137e-01]
     [9.97506766e-01 2.49323361e-03]
     [9.44936715e-01 5.50632852e-02]
     [9.46098744e-01 5.39012555e-02]
     [6.64572789e-01 3.35427211e-01]
     [1.37790169e-02 9.86220983e-01]
     [1.53411545e-01 8.46588455e-01]
     [2.43679108e-01 7.56320892e-01]
     [9.97137216e-01 2.86278427e-03]
     [9.97241365e-01 2.75863538e-03]
     [9.49809821e-01 5.01901792e-02]
     [3.04127207e-04 9.99695873e-01]
     [9.39189783e-01 6.08102172e-02]
     [1.39907384e-03 9.98600926e-01]
     [2.58024715e-01 7.41975285e-01]
     [8.82476853e-01 1.17523147e-01]
     [5.74471795e-01 4.25528205e-01]
     [9.97510524e-01 2.48947584e-03]
     [9.14479637e-01 8.55203631e-02]
     [9.29292136e-01 7.07078643e-02]
     [9.95380870e-01 4.61913030e-03]
     [1.52805928e-01 8.47194072e-01]
     [5.90920981e-01 4.09079019e-01]]
    

S'aider de l'aide en ligne (http://scikit-learn.org) pour l'utilisation de la méthode.
* Quels sont les paramètres de réglage de la méthode ?


```python
clf.get_params() # affichage des paramètres de la méthode GaussianNB avec get_params
```




    {'priors': None, 'var_smoothing': 1e-09}



J'obtient les paramètres de la méthode **GaussianNB** avec **get_params**. Les paramètres de réglage sont donc **priors** (probabilités antérieures des classes) et **var_smoothing** (lissage de la variance). Ici par défaut, priors=None car les probabilités antérieures des classes ne sont pas ajustés en fonction des données et var_smoothing=1e-09.

Lors de la recherche des paramètres optimaux, je calculerais le taux de précision suivant différentes valeurs de **var_smoothing**.

Le paramètre par défaut dans la fonction **fit** est **sample_weight=None** car on accorde le même poids à chaque observations (=1).

L'entraînement du classifieur supervisé se construit à partir de la fonction **fit** sur la base d'entraînement (X_train,Y_train).

La prédiction de la classe utilise la fonction **predict** sur la base de test (X_test).

* Créez une fonction *prediction(labels,pred)* qui renverra le taux de prédiction (en %) pour une prédiction stockée dans *pred* et les valeurs souhaitées stockées dans *label*. On calculera le taux de prédiction de l'apprentissage bayésien naïf sur la base *test*.


```python
def prediction(label,pred):
    # fonction pour calculer le taux de prédiction sur la base test (prédiction juste)
    # entrées :
        # - label : vecteurs des valeurs attendues (Y_test)
        # - pred : vecteurs des prédiction de X_test
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

    Taux de prédiction du classifieur bayésien naïf : 79.0%
    

On peut également déterminer la précision avec la fonction **accuracy_score** :


```python
from sklearn.metrics import accuracy_score
pred = clf.predict(X_test)
print('Taux de prédiction du classifieur bayésien naïf : {:.1%}'.format(accuracy_score(Y_test,pred)))
```

    Taux de prédiction du classifieur bayésien naïf : 79.0%
    

Pour évaluer la qualité du prédicteur, on peut aussi calculer la **matrice de confusion** :


```python
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(label, pred)
conf
```




    array([[195,  39],
           [ 36,  87]], dtype=int64)



Avec la matrice de confusion, on voit que le classifieur se trompe peut.
Si on effectue la somme de la diagonale sur le nombre d'observations, on retrouve la précision.

* Répétez les questions précédentes avec la méthode des **arbres de décision**. Pour cela vous aurez besoin des fonctions:


```python
from sklearn import tree
clf = tree.DecisionTreeClassifier() # création du classifieur (arbre de décision)
clf.fit(X_train,Y_train) # apprentissage du classfieur sur les données train
clf.predict(X_test) # prédiction de la classe Survived avec la base de test
```




    array([1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0,
           1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0,
           1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1,
           0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1,
           0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1,
           1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0,
           0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1,
           0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0,
           1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
           1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1,
           1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0,
           1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0,
           0, 0, 0, 1, 1], dtype=int64)




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

Lors de la recherche des paramètres optimaux, je calculerais le taux de précision suivant différentes valeurs de **criterion** (fonction pour mesurer la qualité d'une scission qui est soit gini ou entropy).


```python
tree.plot_tree(clf) # résultat de l'arbre de décision
plt.show() # affichage de l'arbre
```


![png](output_44_0.png)



```python
# test de la fonction prediction
label = Y_test
pred = clf.predict(X_test)
print('Taux de prédiction de l\'arbre de décision : {:.1%}'.format(prediction(label,pred)))
```

    Taux de prédiction de l'arbre de décision : 80.1%
    

* Mêmes questions ensuite avec la méthode des *k* plus proches voisins et la regression logistique. Voir l'aide en ligne pour un descriptif de l'utilisation des méthodes. A chaque fois, précisez bien quels sont les paramètres de la méthode.


Méthode des **k plus proches voisins** :


```python
from sklearn.neighbors import KNeighborsClassifier
k = 5 # paramètre de réglage pour le nombre de plus proches voisins
clf = KNeighborsClassifier(n_neighbors=k) # création du classifieur (k plus proches voisins)
clf.fit(X_train,Y_train) # apprentissage du classfieur sur les données train
clf.predict(X_test) # prédiction de la classe Survived avec la base de test
```




    array([0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0,
           1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1,
           0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1,
           0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1,
           0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
           0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1,
           1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
           0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1,
           0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0,
           1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0,
           0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0,
           0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0,
           1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0,
           0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0,
           0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0,
           1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0,
           0, 0, 0, 0, 0], dtype=int64)




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

Lors de la recherche des paramètres optimaux, je calculerais le taux de précision suivant différentes valeurs de **n_neighbors** (=nombre de plus proches voisins).

Calcul du taux de prédiction :


```python
# test de la fonction prediction
label = Y_test
pred = clf.predict(X_test)
print('Taux de prédiction du classifieur k plus proches voisins : {:.1%}'.format(prediction(label,pred)))
```

    Taux de prédiction du classifieur k plus proches voisins : 71.1%
    

Méthode de la **regression logistique** :


```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression() # création du classifieur (regression logistique)
clf.fit(X_train,Y_train) # apprentissage du classfieur sur les données train
clf.predict(X_test) # prédiction de la classe Survived avec la base de test
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    




    array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0,
           1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
           0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1,
           0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1,
           0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
           0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
           0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1,
           0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1,
           1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
           0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0,
           1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
           0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0,
           1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0,
           1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
           0, 0, 0, 1, 0], dtype=int64)




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

    Taux de prédiction du classifieur regression logistique : 80.1%
    

## Recherche des paramètres optimaux
Pour chacune des méthodes, utilisez la fonction *split* pour séparer la base en deux DataFrames de taille équivalente (*train* et *test*). Recherchez ensuite les paramètres optimaux qui vous donneront le meilleur taux de prédiction sur *test* lorsque le modèle apprend les données de *train*. Pour cela, utilisez une grille de valeurs pour les paramètres et pour chaque valeur de la grille, calculez le taux de prédiction sur *test* lorsque vous apprenez *train*. Retenez les valeurs de paramètres donnant le meilleur taux. Les bornes de valeurs et le pas de la grille sont à déterminer de façon empirique.

**Bayésien naïf**

L'hyperparamètre d'optimisation pour cette algorithme est la valeur de **lissage de la variance** (var_smoothing).


```python
X_train, X_test, Y_train, Y_test = split(df,y,p=0.5) # appel de la fonction split (p=0.5 car 50% des données dans train et test)
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

    La valeur de lissage de la variance optimale est 0.0001
    Le meilleur taux de prédiction est : 77.8%
    

**Arbre de décision**

L'hyperparamètre d'optimisation pour cette algorithme est la **fonction pour mesurer la qualité d'une d'un noeud** (criterion).


```python
X_train, X_test, Y_train, Y_test = split(df,y,p=0.5) # appel de la fonction split (p=0.5 car 50% des données dans train et test)
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

    Le fonction criterion optimale est gini
    Le meilleur taux de prédiction est : 79.4%
    

**k plus proches voisins**

L'hyperparamètre d'optimisation pour cette algorithme est le nombre de **plus proches voisins** (n_neighbors).


```python
X_train, X_test, Y_train, Y_test = split(df,y,p=0.5) # appel de la fonction split (p=0.5 car 50% des données dans train et test)
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

    Le nombre de voisins optimal est 7
    Le meilleur taux de prédiction est : 71.1%
    

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
    Le meilleur taux de prédiction est : 80.7%
    

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn("The max_iter was reached which means "
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn("The max_iter was reached which means "
    

## Comparaison des méthodes par validation croisée
Pour chacune des méthodes et en utilisant les paramètres optimaux déterminés précédemment, réalisez une procédure de validation croisée pour calculer le taux de prédiction moyen et sa variance lorsque le nombre de folds est 5. Pour cela, vous aurez besoin de la fonction suivante :

Elle vous donnera les taux de prédiction pour chacun des folds de tests (voir procédure de validation croisée et l'aide en ligne http://scikit-learn.org).


```python
from sklearn.model_selection import cross_val_score
folds = 5 # nombre d'échantillons/plis pour la validation croisée du modèle
```

**Bayésien naïf**


```python
clf = GaussianNB(var_smoothing=var_smoothing_max) # création du classifieur (bayésien naïf) avec le meilleur hyperparamètre
scores_gaussian = cross_val_score(clf,X_train,Y_train,cv=folds) # liste des taux de prédiction
print('Taux de prédiction pour chaque échantillon :',scores_gaussian)
print('Taux de prédiction moyen : {:.1%}'.format(scores_gaussian.mean()))
print('Taux de prédiction variance : {:.1%}'.format(scores_gaussian.var()))
```

    Taux de prédiction pour chaque échantillon : [0.70786517 0.76404494 0.79775281 0.84269663 0.83146067]
    Taux de prédiction moyen : 78.9%
    Taux de prédiction variance : 0.2%
    

**Arbre de décision**


```python
clf = tree.DecisionTreeClassifier(criterion=criterion_max) # création du classifieur (arbre de décision) avec le meilleur hyperparamètre
scores_decisiontree = cross_val_score(clf,X_train,Y_train,cv=folds) # liste des taux de prédiction
print('Taux de prédiction pour chaque échantillon :',scores_decisiontree)
print('Taux de prédiction moyen : {:.1%}'.format(scores_decisiontree.mean()))
print('Taux de prédiction variance : {:.1%}'.format(scores_decisiontree.var()))
```

    Taux de prédiction pour chaque échantillon : [0.69662921 0.70786517 0.79775281 0.76404494 0.85393258]
    Taux de prédiction moyen : 76.4%
    Taux de prédiction variance : 0.3%
    

**k plus proches voisins**


```python
clf = KNeighborsClassifier(n_neighbors=n_neighbors_max) # création du classifieur (k plus proches voisins) avec le meilleur hyperparamètre
scores_knn = cross_val_score(clf,X_train,Y_train,cv=folds) # liste des taux de prédiction
print('Taux de prédiction pour chaque échantillon :',scores_knn)
print('Taux de prédiction moyen : {:.1%}'.format(scores_knn.mean()))
print('Taux de prédiction vriance : {:.1%}'.format(scores_knn.var()))
```

    Taux de prédiction pour chaque échantillon : [0.68539326 0.64044944 0.73033708 0.70786517 0.73033708]
    Taux de prédiction moyen : 69.9%
    Taux de prédiction vriance : 0.1%
    

**Régression logistique**


```python
clf = LogisticRegression(solver=solver_max) # création du classifieur (régression logistique) avec le meilleur hyperparamètre
scores_logisticregression = cross_val_score(clf,X_train,Y_train,cv=folds) # liste des taux de prédiction
print('Taux de prédiction pour chaque échantillon :',scores_logisticregression)
print('Taux de prédiction moyen : {:.1%}'.format(scores_logisticregression.mean()))
print('Taux de prédiction variance : {:.1%}'.format(scores_logisticregression.var()))
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    

    Taux de prédiction pour chaque échantillon : [0.73033708 0.82022472 0.76404494 0.82022472 0.86516854]
    Taux de prédiction moyen : 80.0%
    Taux de prédiction variance : 0.2%
    

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




![png](output_80_1.png)


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




![png](output_82_1.png)



```python
print('Taux de précision moyen pour bayésien naïf : {:.1%}'.format(scores_gaussian.mean()))
print('Taux de précision moyen pour arbre de décision : {:.1%}'.format(scores_decisiontree.mean()))
print('Taux de précision moyen pour k plus proches voisins : {:.1%}'.format(scores_knn.mean()))
print('Taux de précision moyen pour régression logistique : {:.1%}'.format(scores_logisticregression.mean()))
```

    Taux de précision moyen pour bayésien naïf : 78.9%
    Taux de précision moyen pour arbre de décision : 76.4%
    Taux de précision moyen pour k plus proches voisins : 69.9%
    Taux de précision moyen pour régression logistique : 80.0%
    

Au vu des résultats des validation croisé k fols, la méthode **régression logistique** est la plus performante par rapport aux autres méthodes de classification. Le prédicteur se trompe dans 20% des cas. Donc pour prédire la classe d'appartenance, on choisira le classifieur **régression logistique**.

La méthode des **k plus proches voisins** prédit moins bien la variable *Survived*.

## Bonus
Vous êtes encouragés à :
* créer de nouveaux attributs à partir des attributs nominaux que vous n'aurez pas utilisés
* tester d'autres algorithme
* faire la fonction de calcul de gain pour un attribut numérique. Cette fonction renverra le seuil qui maximise le gain.

Création d'un nouvel attribut *FamilySize* pour la taille d'une famille à partir de *SibSp* et *Parch* :


```python
df = pd.read_csv('titanic.csv') # rechargement du jeu de données
df = df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1) # je supprime les variables inutiles
df = pd.get_dummies(df, columns=['Pclass','Sex','Embarked']) # convertion de la variable Pclass, Sex et Embarked en type quantitatif binaire
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
    
       Sex_female  Sex_male  Embarked_C  Embarked_Q  Embarked_S  FamilySize  
    0           0         1           0           0           1           2  
    1           1         0           1           0           0           2  
    2           1         0           0           0           1           1  
    3           1         0           0           0           1           2  
    4           0         1           0           0           1           1  
    Index(['Survived', 'Age', 'SibSp', 'Parch', 'Fare', 'Pclass_1', 'Pclass_2',
           'Pclass_3', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q',
           'Embarked_S', 'FamilySize'],
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

    Taux de prédiction moyen : 78.4%
    

Avec la nouvelle variable *FamilySize*, le taux de précision pour l'arbre de décision augmente. Donc, nous pouvons conserver cet attribut car il permet au classifieur d'être plus performant.

Un autre algorithme d'apprentissage : **Perceptron**

On répète 20 fois la procédure de validation croisée à 5 plis pour mesurer la dispersion du taux d’erreur :


```python
from sklearn.linear_model import Perceptron
from statistics import mean
liste_scores_ech = []
for i in range(20):
    X_train, X_test, Y_train, Y_test = split(df,y,p=0.6) # appel de la fonction split
    clf = Perceptron() # création du classifieur (Perceptron)
    scores_perceptron = cross_val_score(clf,X_train,Y_train,cv=5) # liste des taux de prédiction pour 5 plis
    liste_scores_ech.append(scores_perceptron.mean())

print('Taux de prédiction moyen du classifieur de Perceptron : {:.1%}'.format(mean(liste_scores_ech)))
```

    Taux de prédiction moyen du classifieur de Perceptron : 63.5%
    

Le taux de précision de cet algorithme est inférieur aux autres méthodes.

Visualisation du taux de précision :


```python
# boxplot
plt.boxplot(liste_scores_ech)
plt.title('Taux de précision de la méthode Perceptron')
plt.ylabel('Précision')
```




    Text(0, 0.5, 'Précision')




![png](output_96_1.png)

