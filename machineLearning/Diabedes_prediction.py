################################################
# End-to-End Diabetes Machine Learning Pipeline III
################################################

import joblib
import pandas as pd

'''
    * Model Deployement/Model Development süreçlerinden sonra bir modeli canlı sistemlere entegre etmek demek
    o model nesnesini çağırmak demektir.
'''

# Diyelim ki bu veriler yeni hasta verileri olsun.
df = pd.read_csv("datasets/diabetes.csv")

random_user = df.sample(1, random_state=45)

new_model = joblib.load("voting_clf.pkl")

new_model.predict(random_user)

'''
    - Yukarıdaki kodu çalıştırırsak hata alırız çünkü bizim oluşturduğumuz modelde yeni değişkenler var.
    Okuduğumuz yeni hasta verilerindeki değişkenlerle oluşturduğumuz model uyuşmuyor.
    - Bu sorunu oluşmaması için başta modelde yeni değişkenler oluştururken en basit ama en efektif modeli
    oluşturmaya özen göstermeliyiz.
    - Bu problemi çözmek için yeni veri setini Diabetes Data Prep. işleminden geçirmemiz lazım. 
'''

from Diabetes_Pipeline import diabetes_data_prep

X, y = diabetes_data_prep(df)

random_user = X.sample(1, random_state=45)

new_model = joblib.load("voting_clf.pkl")

new_model.predict(random_user)

df.head()