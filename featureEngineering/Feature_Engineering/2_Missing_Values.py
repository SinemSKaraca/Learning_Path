#############################################
# MISSING VALUES
#############################################

'''
    * NaN olan değerler eksik değerlerdir.

    * Ağaç yöntemlerinde, aykırı değerlerde olduğu gibi, eksik değerlerin etkisi göz ardı edilebilir.

    * Eksik değer problemi 3 şekilde çözülebilir:
        ** Silme
        ** Değer Atama Yöntemleri
        ** Tahmine Dayalı Yöntemler (makine öğrenmesi vs. ile tahmin - daha gelişmiştir)

    * "The idea of imputation is both seductive and dangerous" - R.J.A Little & D.B. Rubin

    * Eksik veri ile çalışırken göz önünde bulundurulması gereken önemli konulardan birisi:
      Eksik verilerin rassallığı (Eksikliğin rastgele ortaya çıkıp çıkmadığı bilinmeli!)

    * "Eksik değere sahip gözlemlerin veri setinden direkt çıkarılması ve rassallığının incelenmemesi, yapılacak
      istatiksel çıkarımların ve modelleme çalışmalarının güvenilirliğini düşürecektir." - Reha Alpar

    * Eksik gözemlerin veri setinden direkt çıkarılabilmesi için veri setindeki eksikliğin bazı durumlarda kısmen
      bazı durumlarda tamamen rastlantısal olarak oluşmuş olması gerekmektedir.
      Eğer eksiklikler değişkenler ile ilişkili olarak ortaya çıkan yapısal problemler ile meydana gelmiş ise
      bu durumda yapılacak silme işlemleri ciddi yanlılıklara sebep olabilecektir.
'''

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
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

def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data

dff = load_application_train()
dff.head()

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()
df.head()

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if low_limit > 0:
        dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
    else:
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit

#############################################
# EKSIK DEGERLERIN YAKALANMASI
#############################################

df = load()
df.head()

# eksik gözlem var mı yok mu sorgusu:
df.isnull().values.any()

df.isnull().sum()

df.notnull().sum()

# Kendisinde en az bir tane eksik hücre olan satır sayısı:
df.isnull().sum().sum()

# En az bir tane eksik değere sahip olan gözlem birimleri:
df[df.isnull().any(axis=1)]

# Tam olan gözlem birimleri:
df[df.notnull().all(axis=1)]

# Azalan şekilde sıralama:
df.isnull().sum().sort_values(ascending=False)

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

# Eksik değer olan sütunları getirme:
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
# na_cols = [col for col in df.columns if df[col].isnull().any() == True]

# Yukarıda yaptıklarımızın fonksiyonlaştırılmış hali:
def missing_values_table(dataframe, na_name=False): # na_name - Eksik değerli sütun isimleri
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    number_of_missing = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([number_of_missing, np.round(ratio, 2)], axis=1, keys=["number_of_missing", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df)

###############################
# EKSIK DEGER PROBLEMINI COZME:
###############################

missing_values_table(df)

#########################
# Çözüm 1: Hızlıca silmek
#########################

df.dropna().shape

###############################################
# Çözüm 2: Basit Atama Yöntemleri ile Doldurmak
###############################################

df["Age"].fillna(df["Age"].mean()).isnull().sum()
df["Age"].fillna(df["Age"].median()).isnull().sum()
df["Age"].fillna(0).isnull().sum()

# Bu işlem sayısal olmayan değişkenlere de uygulanacağından HATA alırız!
# df.apply(lambda x: x.fillna(x.mean()), axis=0)

# Hata almamak için:
df.apply(lambda x: x.fillna(x.mean() if x.dtype != 'O' else x), axis=0).head()

'''
    NOTE -> In Pandas:
            axis=0 refers to operations along the rows (i.e., down the columns).
            axis=1 refers to operations along the columns (i.e., across the rows).
'''

dff = df.apply(lambda x: x.fillna(x.mean() if x.dtype != 'O' else x), axis=0)

dff.isnull().sum().sort_values(ascending=False)

df["Embarked"].mode()[0] # En çok tekrar eden değere bu şekilde ulaşırız
df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()

df["Embarked"].fillna("missing")

''' 
    NOTE:
        ** x.nunique() -> doesn't include NaN as a unique value.
        ** len(x.unique()) ->includes NaN as a unique value.
'''

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == 'O' and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

############################################
# Kategorik Değişken Kırılımında Değer Atama
############################################

df.groupby("Sex")["Age"].mean()

df["Age"].mean()

'''
    Buradaki amacımız şu: Örneğin yaş için düşünürsek, tüm eksik değerlere genel yaş ortalamasını atamak yerine
                          Cinsiyet kırılımında eksik yaş değeri bir kadına aitse kadınların yaş ortalamasını,
                          bir erkeğe aitse erkeklerin yaş ortalamasını atarız.
'''

# .transform("mean"): This calculates the mean age for each group (i.e., for males and females separately)
# and replaces the "Age" column with these mean values within each group.
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

# Yukarıdaki satırın daha anlaşılır ama düşük performanslı hali:
df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]
df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

df.isnull().sum()

#############################################
# Çözüm 3: Tahmine Dayalı Atama ile Doldurmak (Makine Öğrenmesi Tekninleri İle)
#############################################

# Eksikliğe sahip olan değişken bağımlı değişken, diğerleri bağımsız değişken kabul edilip
# bir modelleme işlemi gerçekleştireceğiz. Ve modelleme işlemine göre eksik değere sahip
# olan noktaları tahmin etmeye çalışacağız.

'''
    LABEL ENCOCING and ONE-HOT ENCODING:
        In machine learning, models typically work with numerical data. 
        However, real-world data often includes categorical variables (e.g., colors, cities, product names). 
        Since machine learning algorithms cannot directly process these categorical values, 
        we need techniques like label encoding and one-hot encoding to convert them into numerical representations.
    
        These techniques allow models to understand and learn patterns from categorical data. 
        Without such conversions, the models wouldn't be able to make use of important information 
        contained in categorical features, which could lead to poor performance.
        
        ** Label Encoding: Best used when the categorical variable has an ordinal relationship (e.g., Low, Medium, High).
        ** One-Hot Encoding: Best used when there’s no ordinal relationship, and all categories are equally important 
                             (e.g., Red, Blue, Green).
                             
        * The encoding process in label encoding is not random. Label encoding works by assigning a unique integer 
          to each unique category in the data. The order of these integers is typically based on the alphabetical 
          order of the categories by default.
          
        ~ Usage: Label encoding is typically used for binary categorical variables, where there are only two possible 
                 categories (e.g., "Male" and "Female").  
        ~ Usage: One-hot encoding is used for categorical variables with more than two categories. 
                It creates new binary columns for each category, where 1 indicates the presence of the category, 
                and 0 indicates its absence.
'''

df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

# cat_cols kısmını "LABEL ENCODER" ya da "ONE HOT ENCODING" işlemine tabi tutacağız.
# Burada özetle şunu yapıyoruz: iki veya daha fazla sınıfa sahip olan kategorik değişkenleri
# numeric bir şekilde ifade etmek.
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

dff.head()

# Değişkenlerin standartlaştırılması:
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

# Şu anda bir makine öğrenmesi tekniği kullanabilmek için verimiz uygun hale geldi.
# Devam edelim:

# KNN uygulaması:
# Bu bize, makine öğrenmesi yöntemiyletahmine dayalı eksik değer doldurma imkanı sağlayacak.

'''
    KNN (K-Nearest Neighbors):
        * Machine learning algorithm used for both classification and regression tasks.
        * Bana arkadaşını söyle sana kim olduğunu söyleyeyim mantığıyla çalışır:
            En yakın n adet komşusunun değerlerinin ortalaması eksik değere atanır.
        * Uzaklık temelli bir yöntemdir.
'''

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

# Standartlaştırma işlemini geri alıyoruz:
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

df["age_imputed_knn"] = dff[["Age"]]

df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]
df.loc[df["Age"].isnull()]

#######
# RECAP
#######

df = load()
# missing table
missing_values_table(df)
# sayısal değişkenleri median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine dayalı doldurma

#############################################
# GELISMIS ANALIZLER
#############################################

##################################
# Eksik Veri Yapısının İncelenmesi
##################################

# Veri setindeki tam olan gözlemlerin sayılarını verir:
msno.bar(df)
plt.show()

# Değişkenlerdeki eksiklikleirn birlikte çıkıp çıkmadığıyla ilgili bilgi verir:
msno.matrix(df)
plt.show()

# Eksik değerlere dayalı ısı haritası:
msno.heatmap(df)
plt.show()

###############################################################
# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin incelenmesi
###############################################################

missing_values_table(df, True)
na_cols = missing_values_table(df, True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "Survived", na_cols)

#######
# RECAP
#######

df = load()
na_cols = missing_values_table(df, True)
# sayısal değişkenleri direk median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Bağımlı değişken ile ilişkisini inceleme
missing_vs_target(df, "Survived", na_cols)








