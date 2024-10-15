################################
# Unsupervised Learning
################################

# pip install yellowbrick

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder


''' Suç oranlarına göre clustering işlemi yapacağız '''


################################
# K-Means
################################

df = pd.read_csv("datasets/USArrests.csv", index_col=0)

df.head()
df.isnull().sum()
df.info()
df.describe().T

'''
    HATIRLATMA:
        Uzaklık ve Gradient Descent temelli yöntemlerde değişkenlerin standartlaştırılması önem arz ediyor!!
'''

# STANDARTLAŞTIRMA
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df) # Bu işlemin sonucu dataframe değil, numpy array
df[0:5]


# K-MEANS MODEL KURULUMU
kmeans = KMeans(n_clusters=4, random_state=17).fit(df)
kmeans.get_params()

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_    # Etiket isimleri 0,1,2,3.. diye gidiyor
kmeans.inertia_   # Sum of squared distances of samples to their closest cluster center. (SSE/SSD/SSR)


######################################
# Optimum Küme Sayısının Belirlenmesi
######################################

kmeans = KMeans()
ssd = []
K = range(1, 30)

# Her bir cluster sayısı için kmeans modeli tekrar tekrar kuruluyor ve sum of squared error listeye kaydediliyor.
# Bu şekilde en optimal cluster sayısını (hata oranı en düşük olanı) elde edeceğiz.
for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

# ELBOW YÖNTEMİ
plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme Sayısı İçin Elbow Yöntemi")
plt.show()


# KELBOW VISUALIZER
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_


####################################
# Final Cluster'ların Oluşturulması
####################################

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_

df[0:5]

clusters = kmeans.labels_

df = pd.read_csv("datasets/USArrests.csv", index_col=0)
df.head()

df["cluster"] = clusters

df["cluster"] = df["cluster"] + 1

df[df["cluster"] == 1]

df.groupby("cluster").agg(["count", "mean", "median"])

df.to_csv("clusters.csv")


################################
# Hierarchical Clustering
################################

df = pd.read_csv("datasets/USArrests.csv", index_col=0)

# Uzaklık temelli bir yöntem olduğundan standartlaştırma yapıyoruz.
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

# It computes the hierarchical clustering of your data, which means it creates a tree-like structure that
# represents how data points are grouped together into clusters.
hc_average = linkage(df, "average")

plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendrogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.show()


################################
# Kume Sayısını Belirlemek
################################

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axhline(y=0.6, color='b', linestyle='--')
plt.show()


################################
# Final Modeli Oluşturmak
################################

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, linkage="average")









################################
# Principal Component Analysis
################################

''' Hitters veri setindeki asıl amaç maaş tahminleri gerçekleştirmektir '''
df = pd.read_csv("datasets/hitters.csv")
df.head()

num_cols = [col for col in df.columns if df[col].dtypes != 'O' and "Salary" not in col]

df[num_cols].head()

df = df[num_cols]
df.dropna(inplace=True)
df.shape

df = StandardScaler().fit_transform(df)

pca = PCA()
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)


################################
# Optimum Bileşen Sayısı
################################




################################
# Final PCA'in Oluşturulması
################################




################################
# BONUS: Principal Component Regression
################################












################################
# BONUS: PCA ile Çok Boyutlu Veriyi 2 Boyutta Görselleştirme
################################

################################
# Breast Cancer
################################






################################
# Iris
################################






################################
# Diabetes
################################









