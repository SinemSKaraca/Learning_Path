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

clusters = cluster.fit_predict(df)

df = pd.read_csv("datasets/USArrests.csv", index_col=0)
df["hi_cluster_no"] = clusters

df["hi_cluster_no"] = df["hi_cluster_no"] + 1

df["k_means_cluster_no"] = df["k_means_cluster_no"] + 1
df["k_means_cluster_no"] = clusters


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

pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısı")
plt.ylabel("Kümülatif Varyans Oranı")
plt.show()


################################
# Final PCA'in Oluşturulması
################################

''' Başlangıçta 16 tane nümerik değişken vardı. Biz bunları 3 nümerik değişkene indirgedik '''

pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_ # Bileşenlerin tek başına açıkladıkları bilgi oranı
np.cumsum(pca.explained_variance_ratio_) # 1.bileşenin; 1 ve 2. bileşenin birlikte; 1,2 ve 3.bileşenlerin birlikte açıklama oranları


########################################
# BONUS: Principal Component Regression
########################################

'''
    Diyelim ki,
    Hitters veri seti doğrusal bir model ile modellenmek isteniyor ve değişkenler arasında çoklu
    doğrusal bağlantı problemi (Multicollinearity) var. (Regresyon problemlerinde değişkenler arasında yüksek korelasyon
    varsa bu bazı problemlere sebep olur)
'''

df = pd.read_csv("datasets/hitters.csv")
df.shape

len(pca_fit)

num_cols = [col for col in df.columns if df[col].dtypes != 'O' and "Salary" not in col]
len(num_cols)

others = [col for col in df.columns if col not in num_cols]

pd.DataFrame(pca_fit, columns=["PC1", "PC2", "PC3"]).head()

df[others].head()

final_df = pd.concat([pd.DataFrame(pca_fit, columns=["PC1", "PC2", "PC3"]),
                      df[others]], axis=1)
final_df.head()

''' BOYUT İNDİRGEMESİNİ TAMAMLADIK ŞİMDİ MODEL KURALIM '''
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# df'nin içinde kategorik değişkenler de var. Model kurmadan önce LabelEncoder, One-Hot Encoder vs
# kullanarak bu değişkenleri sayısallaştırmalıyız.
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in ["NewLeague", "Division", "League"]:
    label_encoder(final_df, col)

final_df.dropna(inplace=True)

y = final_df["Salary"]
X = final_df.drop(["Salary"], axis=1)

lm = LinearRegression()
rmse = np.mean(np.sqrt(-cross_val_score(lm, X, y, cv=5, scoring="neg_mean_squared_error")))
y.mean()

cart = DecisionTreeRegressor()
rmse = np.mean(np.sqrt(-cross_val_score(cart, X, y, cv=5, scoring="neg_mean_squared_error")))

cart_params = {"max_depth": range(1, 11),
               "min_samples_split": range(2, 20)}

# GridSearchCV
cart_best_grid = GridSearchCV(cart,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True,
                              error_score='raise').fit(X, y)

cart_final = DecisionTreeRegressor(**cart_best_grid.best_params_, random_state=17).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(cart_final, X, y, cv=5, scoring="neg_mean_squared_error")))


##############################################################
# BONUS: PCA ile Çok Boyutlu Veriyi 2 Boyutta Görselleştirme
##############################################################

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

################################
# Breast Cancer
################################

# Amacımız PCA kullanarak bu veri setini iki boyuta indirgeyerek görselleştirmek

df = pd.read_csv("datasets/breast_cancer.csv")

y = df["diagnosis"]
X = df.drop(["diagnosis", "id"], axis=1)

# Veri Setini iki boyuta indirgeyelim:
def create_pca_df(X, y):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(X)
    pca_df = pd.DataFrame(data=pca_fit, columns=["PC1", "PC2"])
    final_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1)
    return final_df

pca_df = create_pca_df(X, y)

''' Bu fonksiyona göndereceğimiz değişkenlerin sayısal değerlerden oluşması lazım '''
def plot_pca(dataframe, target):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("PC1", fontsize=15)
    ax.set_ylabel("PC2", fontsize=15)
    ax.set_title(f"{target.capitalize()}", fontsize=20)

    targets = list(dataframe[target].unique())
    colors = random.sample(['r', 'b', "g", "y"], len(targets))

    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, "PC1"], dataframe.loc[indices, "PC2"], c=color, s=50)
        ax.legend(targets)
        ax.grid()
        plt.show()


plot_pca(pca_df, "diagnosis")


################################
# Iris
################################

import seaborn as sns

df = sns.load_dataset("iris")

y = df["species"]
X = df.drop(["species"], axis=1)

pca_df = create_pca_df(X, y)
plot_pca(pca_df, "species")


################################
# Diabetes
################################

df = pd.read_csv("datasets/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "Outcome")



