#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Küçük ölçekli uygulamalar için:
def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data

df = load_application_train()
df.head()

# Büyük ölçekli uygulamalar için:
def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()
df.head()


#############################################
# OUTLIERS
#############################################

'''
    * Aykırı değişkenler sayısal verilerde oluşur.
    * Özellikle doğrusal problemlerde aykırı değerlerin etkisi daha şiddetlidir.
    * Ağaç yöntemlerinde aykırı değerlerin şiddeti daha düşüktür.
    * Aykırı değişkenler neye göre belirlenir?
        ** Tek değişkenli olarak boxplot (Interquartile range - IQR) yöntemi, çok değişkenli olarak ise LOF yöntemi yaygındır.
'''


#############################
# AYKIRI GOZLEMLERI YAKALAMAK
#############################

##############################################
# Grafik teknikle aykırılar nasıl gözlemlenir?
##############################################

sns.boxplot(x=df["Age"])
plt.show()


##################################
# Aykırı değerler nasıl yakalanır?
##################################

# 1 - çeyreklikleri belirle
q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)

# 2 - aralığı belirle
iqr = q3 - q1

# 3 - eşik değerleri belirle
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

# aykırı değerler getirir:
df[(df["Age"] < low) | (df["Age"] > up)]

# aykırı değerlerin indexleri:
df[(df["Age"] < low) | (df["Age"] > up)].index


#############################
# Aykırı değer var mı yok mu?
#############################

df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)

df[~((df["Age"] < low) | (df["Age"] > up))].any(axis=None)

# 1. eşik deger belirledik
# 2. aykırılara eriştik
# 3. aykırı var mı diye sorduk


##############################
# İşlemleri Fonksiyonlaştırmak - FUNCTIONAL DATA PRE-PROCESSING
##############################

def outlier_tresholds(dataframe, col_name, q1=.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

low, up = outlier_tresholds(df, "Fare")

# Aykırı değerleri getir:
df[(df["Fare"] < low) | (df["Fare"] > up)].head()

# Aykırı değerlerin indexleri:
df[(df["Fare"] < low) | (df["Fare"] > up)].index

# Aykırı değer olup olmadığını öğrenmek için kullanacağımız fonksiyon:
#q1 ve q3'ü değiştirmek istersek bu değişkenleri check_outliers'a parametre olarak göndermek zorundayız!!
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_tresholds(dataframe, col_name)
    if df[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Age")
check_outlier(df, "Fare")


################
# grab_col_names
################

dff = load_application_train()
dff.head()

# Dataframe'deki sayısal, kategorik vs. değişkenleri alabileceğimiz fonksiyon:
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ----------
        dataframe: dataframe
                Değişken isimleri alınmak istenen dataframe
        cat_th: int, optional
                numeric fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    --------
        import seaborn as sns
        df = sns.load_dataframe("iris")
        print(grab_col_names(df))

    Notes
    -----
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols =  [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col, check_outlier(df, col))

dff = load_application_train()
cat_cols, num_cols, cat_but_car = grab_col_names(dff)

for col in num_cols:
    print(col, check_outlier(dff, col))


#############################
# Aykırı Gözlemleri Yakalamak
#############################

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_tresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "Age")

age_index = grab_outliers(df, "Age", True)


#############################################
# AYKIRI GOZLEM PROBLEMINI COZME
#############################################

########
# Silme:
########

low, up = outlier_tresholds(df, "Fare")
df.shape

# Aykırı olmayan değerleri getir:
df[~((df["Fare"] < low) | (df["Fare"] > up))].shape

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_tresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

remove_outlier(df, "Fare").shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    new_df = remove_outlier(df, col)

# 116 adet değişken (outlier) dataframe'den çıkarılmış:
df.shape[0] - new_df.shape[0]


###################
# BASKILAMA YONTEMI - (re-assignment with thresholds)
###################

''' 
    * Silme yönteminde ortaya çıkabilecek veri kaybını önlemek için aykırı değerler
    silinmez, eşik değerlerle değiştirilir.
'''

low, up = outlier_tresholds(df, "Fare")

df[(df["Fare"] < low) | (df["Fare"] > up)]["Fare"]

# Yukardakinin loc kullanılmış versiyonu:
df.loc[((df["Fare"] < low) | (df["Fare"] > up)), "Fare"]

# Outlier değerleri üst limitle değiştirelim:
df.loc[(df["Fare"] > up), "Fare"] = up

# Outlier değerleri alt limitle değiştirelim:
df.loc[df["Fare"] < low, "Fare"] = low

# Yukarıda yaptıklarımızı fonksiyonel hale getirelim:
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_tresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))


#########################
# RECAP
########################

df = load()

outlier_thresholds(df, "Age")
check_outlier(df, "Age")
grab_outliers(df, "Age", index=True)

remove_outlier(df, "Age").shape
replace_with_thresholds(df, "Age")
check_outlier(df, "Age")


#####################################
# Çok Değişkenli Aykırı Değer Analizi - Local Outlier Factor ( LOF )
#####################################

'''
    * Çok değişkenli aykırı değer nedir?
        Tek başına aykırı olmayan bazı değerler birlikte ele alındıklarında bu durumun aykırılık oluşturmasıdır.
        Örneğin 3 evlilik yapmak aykırı değil. 17 yaşında olmak da aykırı değil. Fakat 17 yaşında 3 evlilik yapmış
        olmak aykırı bir durum oluşturur.
        
    * Local Outlier Factor (LOF):
        LOF helps determine if a data point is an outlier based on the density of data points around it. 
        If a point has a significantly lower density than its neighbors, it is considered an outlier.
        
    ** Elimizde 100 tane değişken var fakat ben bunları iki boyutta görselleştirmek istiyorum. 
       Nasıl görselleştirebilirim? (Mülakat Sorusu Olabilir) 
        Eğer bu 100 değişkenin temsil ettiği bilginin çoğunluğunu temsil eden iki değişken bulabilirsem
        bu 100 değişkenki iki değişken kullanarak iki boyutta görselleştirebilirim. İki değişkene indirgeme
        işlemini PCA (Principle Component Anaylsis - Temel bileşen Yöntemi) ile gerçekleştirebiliriz.
        
        * PCA helps reduce the number of features while retaining most of the important information, 
          making it easier to store and process data.
'''

df = sns.load_dataset("diamonds")
df = df.select_dtypes(include=["float64", "int64"])
df = df.dropna()
df.head()

low, up = outlier_tresholds(df, "carat")

df[((df["carat"] < low) | (df["carat"] > up))].shape

low, up = outlier_tresholds(df, "depth")

df[((df["depth"] < low) | (df["depth"] > up))].shape

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

# Değerlerin 1'e yakın olması inlier olması durumunu gösterir. Aşağıda negatif aldığımız için -1
# üzerinden düşüneceğiz.

df_scores = clf.negative_outlier_factor_
df_scores[0:5]
# df_scores = -df_scores
np.sort(df_scores)[0:5]

# PCA'de kullanılan "ELBOW METHOD"
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style=".-")
plt.show()

th = np.sort(df_scores)[3]

# Negatif değerlerle çalıştığımız için eşik değerden küçük olan aykırılara bakıyoruz
df[df_scores < th].shape

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].index

# hepsini silmek istersek;
df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)