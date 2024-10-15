'''
    * Numpy'da eleman tipleri homojen olmalıydı fakat pandas'da böyle bir kısıtlama yok
'''

import pandas as pd
import numpy as np

# PANDAS SERIES:
# -------------

# data:
numbers = [20, 30, 40, 50]
pandas_series = pd.Series(numbers)
print(pandas_series)

letters = ["a", "b", "c", "d"]
pandas_series_2 = pd.Series(letters)
print(pandas_series_2)

datas = ["a", "b", "c", 20]
pandas_series_3 = pd.Series(datas)
print(pandas_series_3)

scalar = 5
pandas_series_4 = pd.Series(scalar, [0, 1, 2, 3])  # sağdaki index bilgisi
print(pandas_series_4)

# index değiştirme:
numbers = [10, 20, 30, 40]
pd_series = pd.Series(numbers, ["a", "b", "c", "d"])
print(pd_series)

dict = {'a': 10, 'b': 20, 'c': 30, 'd': 40}
pandas_series_5 = pd.Series(dict)
print(pandas_series_5)

# numpy ve pandası birlikte kullanma:
random_numbers = np.random.randint(10, 100, 6)
pandas_series_6 = pd.Series(random_numbers)
print(pandas_series_6)

pandas_series_7 = pd.Series([20, 30, 40, 50], ['a', 'b', 'c', 'd'])
print(pandas_series_7)
result = pandas_series_7[0]  # ACCESS BY POSITION
result = pandas_series_7[-1]
result = pandas_series_7[-2:]
result = pandas_series_7[:2]
result = pandas_series_7['a']
result = pandas_series_7['d']
result = pandas_series_7[['a', 'd']]
# result = pandas_series_7[['a', 'd', 'e']] - HATA

print(result)

print(pandas_series_7.ndim)
print(pandas_series_7.dtype)
print(pandas_series_7.shape)
print(pandas_series_7.sum())
print(pandas_series_7.max())
print(pandas_series_7.min())
print(pandas_series_7 + pandas_series_7)
print(pandas_series_7 + 50)
print(np.sqrt(pandas_series_7))
print(pandas_series_7 >= 50)
print(pandas_series_7 % 2 == 0)
print(pandas_series_7[pandas_series_7 % 2 == 0])

opel2018 = pd.Series([20, 30, 40, 10], ["astra", "corsa", "mokka", "insignia"])
opel2019 = pd.Series([40, 30, 20, 10], ["astra", "corsa", "granland", "insignia"])

total = opel2018 + opel2019
print(total)

print(total["astra"])

# -----------------------------------------------------------------------------------------------------
# DIFFERENCE BETWEEN DATAFRAMES AND SERIES:
# ** Series:
#       * A Series is one-dimensional. It's essentially a labeled list of values
#       (similar to a column in a table or a single array). Each value has an associated label (or index).
#       * You can access elements by using the index, e.g., series['index_value'].
# ** DataFrames:
#       * A DataFrame is two-dimensional. It's like a table with rows and columns,
#       where each column is a Series. You can think of a DataFrame as a collection of Series,
#       aligned along the same index.
#       * You can access columns using df['column_name'] and rows using .iloc[] for
#       positional indexing or .loc[] for label-based indexing.
# -----------------------------------------------------------------------------------------------------

# DATAFRAMES:
# ----------

import pandas as pd

s1 = pd.Series([3, 2, 0, 1])
s2 = pd.Series([0, 3, 7, 2])

data = {"apples": s1, "oranges": s2}

df = pd.DataFrame(data)

print(df)

# --------

df = pd.DataFrame()

print(df)

# --------

df = pd.DataFrame([1, 2, 3, 4])

print(df)

# --------

data = [["Ali", 50], ["Veli", 49], ["Yağmur", 70], ["Ayşe", 40]]

# columns ve indexi yazmazsan datadan sonra index bilgisini, sonra columns bilgisini girmelisin
df = pd.DataFrame(data, columns=["Name", "Grade"], index=[1, 2, 3, 4])

print(df)

# --------

dict = {"Name": ["Ali", "Veli", "Yağmur", "Ayşe"],
        "Grade": [50, 49, 70, 40]}

df = pd.DataFrame(dict, index=["212", "232", "236", "456"])

print(df)

# --------

dict_list = [
    {"Name": "Ali", "Grade": 50},
    {"Name": "Veli", "Grade": 49},
    {"Name": "Yagmur", "Grade": 70},
    {"Name": "Ayse", "Grade": 40}
]

df = pd.DataFrame(dict_list, index=["212", "232", "236", "456"])

print(df)

# --------

import pandas as pd
from numpy.random import randn

df = pd.DataFrame(randn(3, 3), index=['A', 'B', 'C'], columns=["Column1", "Column2", "Column3"])

# Sütuna göre seçme işlemleri
result = df["Column1"]
result = type(df["Column1"])
result = df[["Column1", "Column2"]]

# Satıra göre seçme işlemleri:
# LOC --> Used for label-based indexing
# LOC["ROW", "COLUMN"]
# LOC["ROW"] --> Sadece satır seçme
# LOC[:, "COLUMN"] --> Sadece sütun seçme
result = df.loc['A']
result = type(df.loc['A'])
result = df.loc[:, "Column1"]
result = df.loc[:, ["Column1", "Column2"]]
result = df.loc[:, "Column2":"Column3"]
result = df.loc[:, :"Column3"]
result = df.loc['A':'B', :"Column3"]
result = df.loc[:'B', :"Column2"]
result = df.loc['A', "Column3"]

print(result)

# ILOC --> You can access rows and columns by their integer positions
#          instead of their labels. (Integer Location)

# Yeni sütun ekleme
df["Column4"] = pd.Series(randn(3), ['A', 'B', 'C'])
df["Column5"] = df["Column1"] + df["Column3"]

print(df)

# Sütun Silme:
df = df.drop("Column5", axis=1) # atama yapmazsam değişim kalıcı değil

# Değişimin kalıcı olması için:
df.drop("Column5", axis=1, inplace=True)

print(df)

# --------

import pandas as pd
import numpy as np

data = np.random.randint(10, 100, 75).reshape(15, 5)
df = pd.DataFrame(data, columns=["Column1", "Column2", "Column3", "Column4", "Column5"])

result = df.columns
result = df.head()
result = df.head(10)
result = df.tail()
result = df["Column1"].head()
result = df[["Column1", "Column2"]].head()
result = df[5:15][["Column1", "Column2"]]
result = df[5:15][["Column1", "Column2"]].head()

print(result)

# FILTERING:
# ---------

result = df > 50
result = df[df > 50]
result = df[df % 2 == 0]
result = df[df["Column1"] > 50][["Column1", "Column2"]]
result = df[(df["Column1"] > 50) & (df["Column1"] < 70)]
result = df[(df["Column1"] > 50) & (df["Column2"] < 70)]
result = df[(df["Column1"] > 50) | (df["Column2"] < 70)]

print(result)

# --------

import pandas as pd
import numpy as np

personeller = {
    "Calisan": ["Ahmet Yilmaz", " Can Erturk", "Hasan Korkmaz", "Cenk Saymaz", "Ali Turan", "Riza Erturk", "Mustafa Can"],
    "Departman": ["IK", "Bilgi Islem", "Muhasebe", "IK", "Bilgi Islem", "Muhasebe", "Bilgi Islem"],
    "Yas": [30, 25, 45, 50, 23, 34, 42],
    "Semt": ["Kadikoy", "Tuzla", "Maltepe", "Tuzla", "Kadikoy", "Tuzla", "Maletepe"],
    "Maas": [5000, 3000, 4000, 3500, 2750, 6500, 4500]
}

df = pd.DataFrame(personeller)
result = df["Maas"].sum()
result = df.groupby("Departman").groups
result = df.groupby(["Departman", "Semt"]).groups

print(result)


semtler = df.groupby("Semt")

# Bu döngü groupby işleminde bir kalıp gibi
for name, group in semtler:
    print(name)
    print(group)


result = df.groupby("Semt").get_group("Kadikoy")
result = df.groupby("Departman").get_group("Muhasebe")
result = df.groupby("Departman")[["Yas", "Maas"]].sum()
result = df.groupby("Departman")["Maas"].mean()
result = df.groupby("Departman")["Maas"].max()["Muhasebe"]

print(result)

# -------------------------------------------------------------------------

import pandas as pd
import numpy as np

data = np.random.randint(10, 100, 15).reshape(5, 3)

df = pd.DataFrame(data, index=['a', 'c', 'e', 'f', 'h'], columns=["col1", "col2", "col3"])

df = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])

new_col = [np.nan, 30, np.nan, 51, np.nan, 30, np.nan, 10]
df["col4"] = new_col

result = df
result = df.drop("col1", axis=1)
result = df.drop(["col1", "col2"], axis=1)
result = df.drop('a', axis=0)
result = df.drop(['a', 'b', 'h'], axis=0)
result = df.isnull()
result = df.notnull()
result = df.isnull().sum()
result = df["col1"].isnull().sum()
result = df[df["col1"].isnull()]["col1"] # sonraki [col1] sadece o sütunu almak için
result = df[df["col1"].notnull()]["col1"]

print(result)

# DF.DROPNA() -> İçinde NAN değerler olan satır/sütun siler (default: axis=0)
result = df.dropna()
result = df.dropna(axis=1)
result = df.dropna(how="any") # içinde en az 1 tane NAN varsa siler
result = df.dropna(how="all") # satır veya sütundaki tüm değerler NANsa siler
result = df.dropna(subset=["col1", "col2"], how="all")
result = df.dropna(subset=["col1", "col2"], how="any")
result = df.dropna(thresh=2) # En az iki NaN olmayan değer bulundurmayanlar silinir
result = df.dropna(thresh=4)

print(result)

result = df.fillna(value="no input")
result = df.fillna(value=1)

print(result)

result = df.sum()
result = df.sum().sum()
result = df.isnull().sum()
result = df.isnull().sum().sum()

print(result)

def ortalama(dataframe):
    toplam = df.sum().sum()
    adet = df.size - df.isnull().sum().sum()
    return toplam / adet

result = df.fillna(value=ortalama(df))
print(result)

# --------------------------------------------------------------------------

import pandas as pd
import seaborn as sns

df = sns.load_dataset("Titanic")

df.dropna(inplace=True)
df["sex"] = df["sex"].str.upper()
df["sex"] = df["sex"].str.lower()

# Substringin görüldüğü ilk indexi döndürür.
df["sex"] = df["sex"].str.find('e')

df["class"] = df["class"].str.contains("first")



df.head()

# --------------------------------------------------------------------------

import pandas as pd

# A ve B kümeleri olsun:
# INNER JOIN -> Kümelerin kesişimi
# LEFT (OUTER) JOIN -> A kümesinin tamamı
# RIGHT (OUTER) JOIN -> B kümesinin tamamı
# FULL (OUTER) JOIN -> Kümelerin birleşimi

customers = {
    "CustomerId": [1, 2, 3, 4],
    "FirstName": ["Ahmet", "Ali", "Hasan", "Cinar"],
    "LastName": ["Yilmaz", "Korkmaz", "Celik", "Toprak"]
}

orders = {
    "OrderId": [10, 11, 12, 13],
    "CustomerId": [1, 2, 5, 7],
    "OrderDate": ["2010-07-04", "2010-08-04", "2010-07-07", "2012-07-04"]
}

df_customers = pd.DataFrame(customers, columns=["CustomerId", "FirstName", "LastName"])
df_orders = pd.DataFrame(orders, columns=["OrderId", "CustomerId", "OrderDate"])

print(df_customers)
print(df_orders)

# inner join - siparişi olan müşteriler gelsin:
result = pd.merge(df_customers, df_orders, how="inner")

# left join - siparişi olmasa bile bütün müşteriler gelsin:
result = pd.merge(df_customers, df_orders, how="left")

# right join
result = pd.merge(df_customers, df_orders, how="right")

# outer join - tüm kayıtlar getirilir:
result = pd.merge(df_customers, df_orders, how="outer")

print(result)

# ------------

customersA = {
    "CustomerId": [1, 2, 3, 4],
    "FirstName": ["Ahmet", "Ali", "Hasan", "Cinar"],
    "LastName": ["Yilmaz", "Korkmaz", "Celik", "Toprak"]
}

customersB = {
    "CustomerId": [4, 5, 6, 7],
    "FirstName": ["Yagmur", "Cinar", "Cengiz", "Canan"],
    "LastName": ["Bilge", "Turan", "Yilmaz", "Toprak"]
}

df_customersA = pd.DataFrame(customersA, columns=["CustomerId", "FirstName", "LastName"])
df_customersB = pd.DataFrame(customersA, columns=["CustomerId", "FirstName", "LastName"])

result = pd.concat([df_customersA, df_customersB])
result = pd.concat([df_customersA, df_customersB], axis=1)

print(result)

# --------------------------------------------------------------------------

import pandas as pd
import numpy as np

data = {
    "Col1": [1, 2, 3, 4, 5],
    "Col2": [10, 20, 13, 45, 25],
    "Col3": ["abc", "bca", "ade", "cba", "dea"]
}

df = pd.DataFrame(data)

result = df
result = df["Col2"].unique()
result = df["Col2"].nunique()
result = df["Col2"].value_counts()
result = df["Col1"] ** 2

def kare_al(x):
    return x * x

kare_al_2 = lambda x: x * x

result = df["Col1"].apply(kare_al)
result = df["Col1"].apply(kare_al_2)
result = df["Col1"].apply(lambda x: x ** 3)
df["Col4"] = df["Col3"].apply(len)

result = df.columns
result = len(df.columns)
result = df.index
result = len(df.index)
result = df.info

result = df.sort_values("Col2")
result = df.sort_values("Col3")
result = df.sort_values("Col3", ascending=False)

print(result)

# ------------

data = {
    "Ay": ["Mayis", "Haziran", "Nisan", "Mayis", "Haziran", "Nisan", "Mayis", "Haziran", "Nisan"],
    "Kategori": ["Elektronik", "Elektronik", "Elektronik", "Kitap", "Kitap", "Kitap", "Giyim", "Giyim", "Giyim"],
    "Gelir": [20, 30, 15, 14, 32, 42, 12, 36, 52]
}

df = pd.DataFrame(data)

print(df.pivot_table(index="Ay", columns="Kategori", values="Gelir"))

# --------------------------------------------------------------------------

# UYGULAMA 1 - NBA OYUNCULARININ VERİ ANALİZİ







# --------------------------------------------------------------------------

# UYGULAMA 2 - YOUTUBE İSTATİSTİK VERİLERİNİN ANALİZİ






