#############################################
# ENCODING
#############################################

'''
    Encoding, değişkenlerin temsil şekillerinin değiştirilmesidir.
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

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

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

def missing_values_table(dataframe, na_name=False): # na_name - Eksik değerli sütun isimleri
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    number_of_missing = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([number_of_missing, np.round(ratio, 2)], axis=1, keys=["number_of_missing", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

#############################################
# 1) LABEL ENCODING - (Binary Encoding)
#############################################

'''
    * Sıralı (ordinal) değerler için kullanılır. (Örneğin 0 küçük 1 büyük değeri temsil ediyorsa)
    * Kategorik değişkenleri numerik olarak ifade etmek için kullanılır.
    * Değiştirme işlemi (default olarak) alfabetik sıraya göre gerçekleştirilir.
'''

df = load()
df.head()
df["Sex"].head()

label_encoder = LabelEncoder()

"""
    * fit_transform(): This method is a combination of two steps: fit() and transform().
        ** fit(): The encoder learns the mapping from the categorical labels to numerical values. 
        For example, if the "Sex" column contains "Male" and "Female," the encoder will assign 0 to 
        "Female" and 1 to "Male" (or vice versa, depending on the data).
        
        ** transform(): After learning the mapping, this method converts the categorical 
        labels into their corresponding numeric values.
"""
label_encoder.fit_transform(df["Sex"])[0:5]

"""
    * inverse_transform(): This method converts numeric labels back to their original categorical labels. 
    In this case, [0, 1] are the numeric labels you want to convert back.
"""
label_encoder.inverse_transform([0, 1])

# Yukardaki işlemleri fonksiyonlaştıralım:
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    # Aşağıdaki değişiklik kalıcı olacak
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

label_encoder(df, "Sex").head()

df = load()

# Problem: Yüzlerce değişkeni olan bir dataframe'de binary olan değişkenleri bulmamız gerek ki
#          label encoding işlemini sadece onlara uygulayabilelim.
binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"]
              and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df.head()

df = load_application_train()
df.shape

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"]
               and df[col].nunique() == 2]

df[binary_cols].head()

# DİKKAT!!! Burada eksik değerleri otomatik olarak kendi dolduruyor!
for col in binary_cols:
    label_encoder(df, col)

df.head()

#############################################
# 2) ONE-HOT ENCODING
#############################################

'''
    * Burada bir kategorik değişkenin unique değerlerini ayrı sınıflara dönüştürüyoruz.
      Oluşan bu değişkenlere "DUMMY VARIABLES" denir. 
      
    * Dummy değişkenler eğer birbiri üzerinden oluşturulabilir olursa bu durumda ortaya bir 
    ölçme problemi (multicollinearity (i.e., perfect correlation)) çıkmaktadır. 
    Bundan dolayı Dummy değişken oluşturulurken ilk sınıf drop edilir "drop_first=True".
'''

df = load()
df.head()
df["Embarked"].value_counts()

# Embarked değişkeninin dummy değişkenlerini olurşturduk:
pd.get_dummies(df, columns=["Embarked"]).head()

# Ölçme problemi olmaması için ilk dummy değişkeni df'den çıkarıyoruz:
pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()

# NaN değerler için de dummy değişken oluşturabiliriz:
pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()

# ÖNEMLİ!!!
# Tüm kategorik değişkenleri girdiğimizde binary değişkenler de dolaylı olarak
# label encode edilmiş gibi oluyor.
temp_df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True).head()

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = load()

# cat_cols, num_cols, cat_but_car = grab_col_names(df)

one_hot_cols = [col for col in df.columns if 2 < df[col].nunique() <= 10]

one_hot_encoder(df, one_hot_cols).head()

#############################################
# 3) RARE ENCODING
#############################################

# Bir değişkenin frekansı az olan değişkenlerini ayrı bir rare isimli değişkende toplarız.

# 1. Kategorik değişkenlerin azlık çoklık durumunu analiz edeceğiz
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkiyi analiz edeceğiz
# 3. Rare encoder yazacağız

####################################################################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi:
####################################################################

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#######################################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

################################################################################
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi:
################################################################################

df["NAME_INCOME_TYPE"].value_counts()

df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "TARGET", cat_cols)

###############################
# 3. Rare encoder'ın yazılması:
###############################

df = load_application_train()

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01)

rare_analyser(new_df, "TARGET", cat_cols)

df["OCCUPATION_TYPE"].value_counts()

#############################################
# FEATURE SCALING (Özellik Ölçeklendirme)
#############################################

'''
    ** Yaptığımız işlem basitçe standartlaştırma işlemidir.

    * Tüm değişkenleri "eşit şartlar altında değerlendirebilmek adına ölçeklendirme yapılar".
    
    * Özellikle gradient-descent yöntemi kullanıldığında "training süresi azalır". Çünkü her iterasyonda
      ölçek farklılıklarından kaynaklanan error'ların boyutları daha küçük olur.
    
    * Özellikle KNN, K-Means gibi uzaklık temelli bazı yöntemler kullanıldığında ölçeklerin birbirinden
      farklı olması durumu, uzaklık, benzemezlik gibi hesaplamalarda yanlılığa sebep olmaktadır. Scaling
      işlemi bu gibi durumlarda "yanlılığın önüne geçer".
'''

###############################
# 1-) StandardScaler
###############################

# Klasik standartlaştırma (Normalleştirme). Bütün değerlerden(x) ortalamayı(u) çıkar, standart sapmaya(s) böl.
# {z = (x - u) / s}

df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])

df.head()

'''
    NOTE:
        ** fit_transform() expects a 2D array or a DataFrame, not a 1D array.
        * df["Age"] -> Pandas Series (1D Structure)
        * df[["Age"]] -> Pandas Dataframe (2D Structure with rows and columns)
'''

'''
    NOTE:
        * Standart sapma da ortalama da aykırı değerlerden etkilenen metriklerdir.
          Medyan ve IQR etkilenmediği için bir sonraki scaler yöntemine robust denilmektedir.
      
        * Robust scaler aykırı değerlere karşı daha dayanıklı olsa da standard scaler
          daha yaygın bir kullanıma sahiptir.
'''

###############################
# 2-) RobustScaler
###############################

# Medyanı çıkar. IQR'a böl

rs = RobustScaler()
df["Age_robust_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

###############################
# 3-) MinMaxScaler
###############################

# Verilen iki değer arasında değişken dönüşümü

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

df.head()

'''
    NOTE:
        * Scaling işleminden sonra değişkenin dağılımı değişmiyor,
          Değişkenlerin ifade edildiği sayısal değerler (ölçek) değişiyor.
          
        * Yapılarını koruyacak şekilde ifade ediliş tarzlarını değiştiriyoruz!!!!!
          Yani değişkenin taşıdığı bilgiyi bozmuyoruz! Bu bilginin ifade ediliş tarzını değiştiriyoruz!
'''

age_cols = [col for col in df.columns if "Age" in col]

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in age_cols:
    num_summary(df, col, plot=True)

###################
# Numeric to Categorical: Sayısal Değişkenleri Kateorik Değişkenlere Çevirme
# Binning
###################

df["Age_qcut"] = pd.qcut(df["Age"], 5)

df.head()
