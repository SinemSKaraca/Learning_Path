################################################
# KNN
################################################

# 1. Exploratory Data Analysis ----------------------> Veriyi Tanıma
# 2. Data Preprocessing & Feature Engineering -------> Eksiklik, aykırılık vs inceleme & Yeni değişkenler üretme
# 3. Modeling & Prediction --------------------------> Modelleme ve tahmin yapma
# 4. Model Evaluation -------------------------------> Model başarısı değerlendirme
# 5. Hyperparameter Optimization --------------------> Optimizasyon yapma (Dışsal parametreye yani komşuluk sayısı hiperparametresine)
# 6. Final Model ------------------------------------> Optimizasyon sonrası final modelini kurma

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


################################################
# 1. Exploratory Data Analysis
################################################

df = pd.read_csv("datasets/diabetes.csv")

df.head()
df.shape
df.describe().T
df["Outcome"].value_counts()


################################################
# 2. Data Preprocessing & Feature Engineering
################################################

y = df["Outcome"]
X = df.drop("Outcome", axis=1)

X_scaled = StandardScaler().fit_transform(X)

X = pd.DataFrame(X_scaled, columns=X.columns)

################################################
# 3. Modeling & Prediction
################################################

knn_model = KNeighborsClassifier().fit(X, y)

random_user = X.sample(1, random_state=45)

knn_model.predict(random_user)


################################################
# 4. Model Evaluation
################################################

# Confusion matrix için y_pred:
y_pred = knn_model.predict(X)

# AUC için y_prob:
y_prob = knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))

# acc 0.83
# f1 0.74

# AUC
roc_auc_score(y, y_prob)
# 0.90

''' Bu kısımda modeli, modeli kurduğumuz/eğittiğimiz veriyle test ettik! '''

# Cross Validation:
cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

# 0.73
# 0.59
# 0.78

# BAŞARI SKORLARI NASIL ARTTIRILABİLİR?
# 1. Veri boyutu arttırılabilir
# 2. Veri ön işleme
# 3. Özellik mühendisliği
# 4. İlgili algoritma için optimizasyonlar yapılabilir.

knn_model.get_params()


################################################
# 5. Hyperparameter Optimization
################################################

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2,50)}

'''
    knn_model: This is the initialized KNN model for which you want to perform hyperparameter tuning.
    knn_params: This is the dictionary of hyperparameters and their corresponding ranges to test 
                (here, different values for n_neighbors).
    cv=5: This sets the cross-validation strategy to 5-fold cross-validation. It means that the data 
          will be split into 5 parts, and the model will train on 4 parts and validate on the remaining part, 
          iterating 5 times with different splits.
    n_jobs=-1: This means that all available CPU cores will be used for parallel processing, speeding up the grid search.
    verbose=1: This controls the verbosity of the output. 1 means that some information will be printed to the 
               console as the grid search progresses. (Raporlama yapar verbose=1 dersek)
'''
''' Her bir komşuluk değeri için knn modeli kurulur (fit) ve hataya bakılır '''
knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)

knn_gs_best.best_params_


################################################
# 6. Final Model
################################################

# En iyi n_neighbor değeri ile final modelini kuruyoruz. İlk kurduğumuz model default n = 5 ile kurulmuştu.
knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

random_user = X.sample(1)

knn_final.predict(random_user)

