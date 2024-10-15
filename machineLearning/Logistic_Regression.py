######################################################
# Diabetes Prediction with Logistic Regression
######################################################

# İş Problemi:

# Özellikleri belirtildiğinde kişilerin diyabet hastası olup
# olmadıklarını tahmin edebilecek bir makine öğrenmesi
# modeli geliştirebilir misiniz?

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin
# parçasıdır. ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan
# Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir. 768 gözlem ve 8 sayısal
# bağımsız değişkenden oluşmaktadır.
#
#
# Hedef değişken "outcome" olarak belirtilmiş olup;
#       1 diyabet test sonucunun pozitif oluşunu,
#       0 ise negatif oluşunu belirtmektedir.

# Değişkenler
# Pregnancies: Hamilelik sayısı
# Glucose: Glikoz.
# BloodPressure: Kan basıncı.
# SkinThickness: Cilt Kalınlığı
# Insulin: İnsülin.
# BMI: Beden kitle indeksi.
# DiabetesPedigreeFunction: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
# Age: Yaş (yıl)
# Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil (0)


# 1. Exploratory Data Analysis
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Model Validation: Holdout
# 6. Model Validation: 10-Fold Cross Validation
# 7. Prediction for A New Observation


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay, \
    roc_curve
from sklearn.model_selection import train_test_split, cross_validate


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + interquantile_range * 1.5
    low_limit = quartile1 - interquantile_range * 1.5
    return low_limit, up_limit

def check_outliers(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


######################################################
# Exploratory Data Analysis
######################################################

df = pd.read_csv("datasets/diabetes.csv")
df.head()
df.shape


##########################
# Target'ın Analizi -------- Bağımlı değişken analizi
##########################

df["Outcome"].value_counts()

sns.countplot(data=df, x="Outcome")
plt.show()

100 * df["Outcome"].value_counts() / len(df)


##########################
# Feature'ların Analizi ------ Bağımsız değişken analizi
##########################

df.describe().T

# Kutu Grafiği ve Histogram -> Sayısal verileri görselleştirmede kullanılır.
df["Glucose"].hist(bins=20)
plt.xlabel("Glucose")
plt.show()

# Yukarıdaki işlemi fpnksiyonlaştıralım
def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)

for col in df.columns:
    plot_numerical_col(df, col)

cols = [col for col in df.columns if col not in "Outcome"]

for col in cols:
    plot_numerical_col(df, col)


##########################
# Target vs Features
##########################

# Target'a göre groupby'a alıp sayısal değişkenlerin ortalamasını alalım
df.groupby("Outcome").agg({"Pregnancies": "mean"})

# Yukarıdaki işlemi fonksiyonlaştıralım
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in cols:
    target_summary_with_num(df, "Outcome", col)


######################################################
# Data Preprocessing (Veri Ön İşleme)
######################################################

df.isnull().sum()

df.describe().T # veri setinde NaN değerler 0 ile değiştirilmiş

for col in cols:
    print(col, check_outliers(df, col))

replace_with_thresholds(df, "Insulin")


'''
    Değişken Ölçeklendirme:
        - Doğrusal ve uzaklık temelli yöntemlerde ve gradient descent kullanan yöntemlerde genellikle
        standatlaştırma işlemlerleri büyük önem taşır.
        - Modellerin değişkenlere eşit yaklaşmasını sağlar. Örneğin, değerleri daha büyük olan değişkenin
        değerleri daha küçük olan değişkenlere bir üstünlüğü olmadığını ifade etmede kullanılır.
        - Kullanılan parametre tahmin yöntemlerinin daha hızlı ve daha doğru tahminlerde bulunması
        için kullanılır.
'''

# SCALING
for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

df.head()


######################################################
# Model & Prediction
######################################################

# Amaç, kişilerin verilen özelliklerine göre diyabet hastası olup olamayacaklarını bulmak

y = df["Outcome"]

X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)

log_model.intercept_
log_model.coef_

y_pred = log_model.predict(X)

y_pred[0:10]
y[0:10]

######################################################
# Model Evaluation - Model Başarı Değerlendirme
######################################################

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel("y_pred")
    plt.ylabel("y")
    plt.title("Accuracy Score: {0}".format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))

# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# ROC AUC
# predict_proba() returns the predicted probabilities for each class (e.g., class 0 and class 1) in a classification problem.
# This part extracts the second column of the 2D array returned by predict_proba(), which corresponds to the probability
# of the positive class (class 1).
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)
# 0.83939

'''
    Bu kısımda modeli, modeli eğittiğimiz verilerle test ettik.
'''

######################################################
# Model Validation: Holdout
######################################################

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=17)

log_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

# Başarı değerlendirme
print(classification_report(y_test, y_pred))


# EĞİTİLDİĞİ VERİ SETİYLE TEST EDİLDİ
# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# EĞİTİM SETİNDEN AYRILAN TEST SETİYLE TEST EDİLDİ
# Accuracy: 0.77
# Precision: 0.79
# Recall: 0.53
# F1-score: 0.63

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

# AUC
roc_auc_score(y_test, y_prob)

# ÖNCEKİ DEĞER -> 0.83
# ŞİMDİKİ DEĞER -> 0.87


######################################################
# Model Validation: 10-Fold Cross Validation
######################################################

# Veri setini böldüğümüzde hangi train ve test setlerini kullanacağımız, modelin doğruluğu açısından
# çok önemli. Modelin doğrulama sürecini en doğru şekilde ele alması için n-fold cross validation
# yöntemini kullanacağız.

y = df["Outcome"]
X = df.drop("Outcome", axis=1)

log_model = LogisticRegression().fit(X, y)

cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])


# EĞİTİLDİĞİ VERİ SETİYLE TEST EDİLDİ
# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# EĞİTİM SETİNDEN AYRILAN TEST SETİYLE TEST EDİLDİ
# Accuracy: 0.77
# Precision: 0.79
# Recall: 0.53
# F1-score: 0.63

# K-FOLD CROSS VALIDATION KULLANILDIĞINDA
cv_results['test_accuracy'].mean()
# Accuracy: 0.7721

cv_results['test_precision'].mean()
# Precision: 0.7192

cv_results['test_recall'].mean()
# Recall: 0.5747

cv_results['test_f1'].mean()
# F1-score: 0.6371

cv_results['test_roc_auc'].mean()
# AUC: 0.8327


######################################################
# Prediction for A New Observation
######################################################

X.columns

# 1: you want to select one row
random_user = X.sample(1, random_state=45)
log_model.predict(random_user)