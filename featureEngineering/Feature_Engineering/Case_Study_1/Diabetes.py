##############################
# Diabete Feature Engineering
##############################

# Problem : Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi
# istenmektedir. Modeli geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.

# BU PROBLEM BİR SINIFLANDIRMA (CLASSIFICATION) PROBLEMİDİR.

# # TARGET DEĞİŞKENİ -> OUTCOME
# # BAĞIMSIZ DEĞİŞKENLER -> OUTCOME HARİCİNDEKİLER

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır.
# ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian kadınları
# üzerinde yapılan diyabet araştırması için kullanılan verilerdir. 768 gözlem ve 8 sayısal bağımsız değişkenden oluşmaktadır.
# Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

# Pregnancies: Hamilelik sayısı
# Glucose: Glikoz
# BloodPressure: Kan basıncı (Diastolic(Küçük Tansiyon))
# SkinThickness: Cilt Kalınlığı
# Insulin: İnsülin.
# BMI: Beden kitle indeksi.
# DiabetesPedigreeFunction: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
# Age: Yaş (yıl)
# Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil (0)


# GÖREV 1: KEŞİFCİ VERİ ANALİZİ
           # Adım 1: Genel resmi inceleyiniz.
           # Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
           # Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
           # Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)
           # Adım 5: Aykırı gözlem analizi yapınız.
           # Adım 6: Eksik gözlem analizi yapınız.
           # Adım 7: Korelasyon analizi yapınız.

# GÖREV 2: FEATURE ENGINEERING
           # Adım 1:  Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
           # değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri
           # 0 olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik değerlere
           # işlemleri uygulayabilirsiniz.
           # Adım 2: Yeni değişkenler oluşturunuz.
           # Adım 3:  Encoding işlemlerini gerçekleştiriniz.
           # Adım 4: Numerik değişkenler için standartlaştırma yapınız.
           # Adım 5: Model oluşturunuz.


# Gerekli Kütüphane ve Fonksiyonlar:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
# from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
#pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


df = pd.read_csv("Feature_Engineering/Case_Study_1/diabetes.csv")
df.head()

##################################
# GÖREV 1: KEŞİFCİ VERİ ANALİZİ
##################################

##############
# GENEL RESİM
##############

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum()) #eksik deger var mı? varsa kac tane?
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T) # sayısal değişkenlerin ceyrekliklerinin incelenmesi

check_df(df)

# Glucose degeri sıfır olabilir mi?
# Insulin degeri sıfır olabilir mi ?
# Kan basıncı sıfır olabilir mi?
# Veri setinde eksik degereler vardı da sıfır basıldı?
# Insulin degerinde 95 ceyreklikten max degere buyuk bir sıcrayıs var bu da aykırı deger olabileceginin bir sinyali.

df.head()
df.dtypes

#################################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
#################################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtype != "O" and
                   dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtype == "O"
                   and dataframe[col].nunique() > car_th]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if dataframe[col] not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != "O"]
    # numeric görünümlü kategorikleri çıkaralım:
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations..: {dataframe.shape[0]}")
    print(f"Variables.....: {dataframe.shape[1]}")
    print(f"cat_cols......: {len(cat_cols)}")
    print(f"num_cols......: {len(num_cols)}")
    print(f"cat_but_car...: {len(cat_but_car)}")
    print(f"num_but_cat...: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################

# amacım değişkene dair degerlerin oranına göz atmak.
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(), #değişkende hangi degerden kacar adet var?
                        "Ratio": dataframe[col_name].value_counts() / len(dataframe)})) # # deger adetlerini toplam deger sayısına bölümü oran verir.
    print("------------------------------")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

# Hedef kategorik değişkende deneyelim:
cat_summary(df, "Outcome")

# Tüm datasette deneyelim:
for col in df.columns:
    cat_summary(df, col)

##################################
# NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################

def num_summary(dataframe, numerical_col, plot=False):  # plot:true olursa if çalışır.
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99] # hangi ceyreklikleri istiyorum?
    print(dataframe[numerical_col].describe(quantiles).T) # istedigim ceyreklikler bazında describe göz atıyorum.

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=False)

    
############################################
# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
############################################

# Burada target outcome (kişinin diyabetik olup olmaması durumu) değişkeni

def target_summary_with_num(dataframe, target, numerical_cols):
    print(dataframe.groupby(target).agg({numerical_cols: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

##################################
# KORELASYON
##################################

''' 
    * Korelasyon, olasılık kuramı ve istatistikte iki rassal değişken arasındaki doğrusal ilişkinin yönünü ve gücünü belirtir
    * In simpler terms, it tells you how one variable changes in relation to another.
    
    ** Positive correlation means that as one variable increases, the other also increases.
    ** Negative correlation means that as one variable increases, the other decreases.
    ** Zero correlation means there is no relationship between the two variables.
'''

df.corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()


##################################
# BASE MODEL KURULUMU
##################################
# Bu konular ileride görülecek... O yüzden şimdilik burayı ezbere geçelim.
# Amacımız herhangi bir işlem yapmadan başarımız ne durumda?
# Sonrasıyla karsılastıralım.

y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}") # basarı oranı
print(f"Recall: {round(recall_score(y_pred,y_test),3)}") # Gercekte diyabet olanların kacına diyabet dedigi
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}") # Recall'in tam tersi. Model tarafından tahmin edilen degerlerin kac tanesi diyabet
print(f"F1: {round(f1_score(y_pred,y_test), 2)}") # Recall ve precision ortalaması
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}") # farklı sınıflandırma esik degerlerine göre basarı

# Accuracy: 0.77
# Recall: 0.706 # pozitif sınıfın ne kadar başarılı tahmin edildiği
# Precision: 0.59 # Pozitif sınıf olarak tahmin edilen değerlerin başarısı
# F1: 0.64
# Auc: 0.75


# Model hangi değişkene daha cok önem varmiş?
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)


##################################
# GÖREV 2: FEATURE ENGINEERING
##################################

##################################
# EKSİK DEĞER ANALİZİ
##################################

# eksik degerler yoktu. Fakat sıfır olamayacak değişkenlere sıfır atanmıstı.
df.isnull().sum()
df.describe()


# Bir insanda Pregnancies ve Outcome dışındaki değişken değerleri 0 olamayacağı bilinmektedir.
# Bundan dolayı bu değerlerle ilgili aksiyon kararı alınmalıdır. 0 olan değerlere NaN atanabilir .

# minimum degeri sıfır olamayacak değişkenler yakalanıyor.
# kategorik değişkenler hariç bırakılıyor.
zero_columns = [col for col in df.columns if (df[col].min() == 0) and col not in ["Pregnancies", "Outcome"]]


# Gözlem birimlerinde 0 olan degiskenlerin her birisine gidip 0 iceren gozlem degerlerini NaN ile değiştiriyoruz.
# where ile eger ki şart saglanıyorsa NAN yazacagım, saglanmıyorsa oldugu gibi yazacagım.
'''
    np.where(condition, x, y): 
    This function checks the condition and returns x where the condition is true, and y where the condition is false.
'''
for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])


# Eksik değer gözlem analizi:
df.isnull().sum()


# artık eksik degerleri (NAN) incelebilirz.
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0] # eksik değerleri aldık
    n_miss =  dataframe[na_columns].isnull().sum().sort_values(ascending=False) # sıralamanın sebebi ilk olarak fazla eksik degere sahip eğişkenleri görmek istememiz.
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0]).sort_values(ascending=False) # eksik degerlerin tüm değerler içerisindeki oranı
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio']) # kac deger var ve oranını birleştiriyoruz.
    print(missing_df, end='\n')
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)


# Eksik Değerlerin Bağımlı Değişken (Target) ile İlişkisinin İncelenmesi
# Amacımız eksik degerler ile var olan degerlerin karsılastırmasını yapmak olacak.
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        # temp_df[col].isnull() degeri true false olarak döndürür. true ise 1: false ise 0 yazar.
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")]
    for col in na_flags:
        print({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
               "Count": temp_df.groupby(col)[target].mean()})

missing_vs_target(df, "Outcome", na_columns)

# Eskik değerlerin doldurulması:
for col in zero_columns:
    df.loc[df[col].isnull(), col] = df[col].median()

df.isnull().sum()

df.describe().T


##################################
# AYKIRI DEĞER ANALİZİ
##################################

# Aykırı degerler için limit belirleme:
def outlier_threshold(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr
    return low_limit, up_limit

low, up = outlier_threshold(df, "Insulin")

# Aykırı değer var mı yok mu?:
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_threshold(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Insulin")

'''
    NOTE -> For direct modifications in the original DataFrame, using loc is essential.
'''

# Aykırı değerleri baskılama:
def replace_with_threshold(dataframe, col_name, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_threshold(dataframe, col_name, q1, q3)
    df.loc[df[col_name] < low_limit, col_name] = low_limit
    df.loc[df[col_name] > up_limit, col_name] = up_limit

sns.boxplot(x=df["Insulin"])

# Aykırı değer analizi ve aykırı değer baskılama işlemi:
for col in df.columns:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_threshold(df, col)

for col in df.columns:
    print(col, check_outlier(df, col))


##################################
# ÖZELLİK ÇIKARIMI
##################################

df.loc[(df["Age"] >= 21) & (df["Age"] < 50), "NEW_AGE_CAT"] = "mature"
df.loc[(df["Age"] >= 50), "NEW_AGE_CAT"] = "senior"

df.head()

# Beden-Kitle indeksi (BMI)
# BMI 18,5 aşağısı underweight, 18.5 ile 24.9 arası normal, 24.9 ile 29.9 arası Overweight ve 30 üstü obez
df["NEW_BMI"] = pd.cut(x=df["BMI"], bins=[0, 18.5, 24.9, 29.9, 100], labels=["Underweight", "Normal", "Overweight", "Obese"])

df[["BMI", "NEW_BMI"]].head()

# Glukoz degerini kategorik değişkene çevirme:
df["NEW_GLUCOSE"] = pd.cut(x=df["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])

df[["Glucose", "NEW_GLUCOSE"]].head()

# Yaş ve beden kitle indeksini bir arada düşünerek kategorik değişken oluşturma (3 kırılım yakalandı):
df.loc[(df["BMI"] < 18.5) & (df["Age"] >=21) & (df["Age"] < 50), "NEW_AGE_BMI_NOM"] = "underweightmature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthymature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightmature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"
df.loc[(df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"


# Yaş ve Glikoz değerlerini bir arada düşünerek kategorik değişken oluşturma:
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 21) & (df["Age"] < 50), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"
df.loc[(df["Glucose"] > 125) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"


# İnsülin değeri ile kategorik değişken üretmek
def set_insulin(dataframe, col_name="Insulin"):
    if 16 <= dataframe[col_name] <= 166:
        return "Normal"
    else:
        return "Abnormal"

df["NEW_INSULIN_SCORE"] = df.apply(set_insulin, axis=1)

df.head()

df["NEW_GLUCOSE*INSULIN"] = df["Glucose"] * df["Insulin"]

# Pregnancies değişkeninden dolayı sıfır olan değişkenlere dikkat!!!
df["NEW_GLUCOSE*PREGNANCIES"] = df["Glucose"] * (1 + df["Pregnancies"])

# Kolonların isimlerinin büyütülmesi:
df.columns = [col.upper() for col in df.columns]

df.head()


##################################
# ENCODING
##################################

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# LABEL ENCODING:
# --------------

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == 'O' and df[col].nunique() == 2]

# Label encoderin tüm binary sütunlara uygulanması:
for col in df.columns:
    df = label_encoder(df, col)

df.head()

# ONE-HOT ENCODING:
# ----------------

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# cat_cols'tan binary col'ları ve target değişkenimi çıkarıyorum.
# bir de binary_cols, zaten daha öncesinde label encoder uygulamıstım.
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in "OUTCOME"]

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()


##################################
# STANDARTLAŞTIRMA
##################################

num_cols

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()
df.shape


##################################
# MODELLEME
##################################

# Feature Engineering ardından model basarısını degerlendirelim.

df.head()
df.columns


y = df["OUTCOME"]
X = df.drop("OUTCOME", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")



# Accuracy: 0.79
# Recall: 0.711
# Precision: 0.67
# F1: 0.69
# Auc: 0.77

# Base Model
# Accuracy: 0.77
# Recall: 0.706
# Precision: 0.59
# F1: 0.64
# Auc: 0.75



##################################
# FEATURE IMPORTANCE
##################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)