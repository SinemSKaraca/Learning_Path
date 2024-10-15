###############################################
# Sales Prediction with Linear Regression
###############################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.float_format", lambda x: "%.2f" % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


######################################################
# Simple Linear Regression with OLS Using Scikit-Learn
######################################################

df = pd.read_csv("datasets/advertising.csv")
df.head()

X = df[["TV"]]
y = df[["sales"]]


########
# Model
########

'''After initializing the linear regression model, we use the .fit() method to train it on the data.'''
reg_model = LinearRegression().fit(X, y)

# formül -> y(i) = b + w * x(i) :: kurduğumuz modelde x = TV
# "Bir model kurdum" dediğimizde elimizde olan değerler bias ve weight değerleridir

'''b -> This is the value of the dependent variable when all independent variables are 0.'''
# sabit / b / bias / intercept
reg_model.intercept_[0]

'''w -> These represent how much the dependent variable changes when the independent variable changes.'''
# TV'nin katsayısı / ağırlığı / coefficient (w1)
reg_model.coef_[0][0]

# Tahmin kısmında intercept ve coefficient değerlerini kullanarak tahmin işlemi gerçekleştireceğiz.


##########################
# Tahmin
##########################

# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir?

reg_model.intercept_[0] + reg_model.coef_[0][0] * 150

# 500 birimlik TV harcaması olsa ne kadar satış olması beklenir?

reg_model.intercept_[0] + reg_model.coef_[0][0] * 500

df.describe().T


# Modelin Görselleştirilmesi:
g = sns.regplot(x=X, y=y, scatter_kws={"color": 'b', 's': 9},
                ci=False, color='r')

g.set_title(f"Model Denklemi: Sales = "
            f"{round(reg_model.intercept_[0], 2)} + TV * {round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()


##########################
# Tahmin Başarısı
##########################

y_pred = reg_model.predict(X)

# MSE:
mean_squared_error(y, y_pred)
# 10.51
y.mean()
y.std()

# RMSE:
np.sqrt(mean_squared_error(y, y_pred))
# 3.24

# MAE:
mean_absolute_error(y, y_pred)
# 2.54

# R-KARE -> Veri setindeki bağımsız değişkenlerin bağımlı değişkeni açıklama yüzdesidir.
#           Örneğin burada TV değişkeninin sales değişkenini açıklama yüzdesine bakıyoruz.
reg_model.score(X, y)


# DİKKAT -> Veri setindeki değişken sayısı arttıkça R-KARE değerinde şişme gözlemlenir.
# DİKKAT -> Model anlamlılıkları, katsayı testi vs. yapmıyoruz. Konuya makine öğrenmesi yaklaşımıyla bakıyoruz.
#           Çünkü istatistiki problemler üzerinde çalışmıyoruz.


######################################################
# Multiple Linear Regression
######################################################

# Önceki bölümdeki gibi bir tane değil birden fazla bağımsız değişken ile çalışacağız ve
# bütün veri setini modellemiş olacağız

df = pd.read_csv("datasets/advertising.csv")

X = df.drop("sales", axis=1)

y = df[["sales"]]


##########################
# Model
##########################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

X_train.shape
X_test.shape

y_test.shape
y_train.shape

# Modeli kuralım:
reg_model = LinearRegression().fit(X_train, y_train)

# intercept (b - bias)
reg_model.intercept_

# coefficient (w - weight)
reg_model.coef_


##########################
# Tahmin
##########################

# Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?

# TV: 30
# radio: 10
# newspaper: 40

# 2.90794702                               -> Sabit
# 0.0468431 , 0.17854434, 0.00258619   -> Ağırlıklar

# MODEL DENKLEMİ -> Sales = 2.90  + TV * 0.04 + radio * 0.17 + newspaper * 0.002

# Yukarıdaki TV, Radio ve Newspaper değerlerine göre tahmin edilen satış:
2.90794702 + 0.0468431 * 30 + 0.17854434 * 10 + 0.00258619 * 40

# Yukarıdaki eşitliği fonksiyonlaştıralım:
yeni_veri = [[30], [10], [40]]
yeni_veri = pd.DataFrame(yeni_veri).T

reg_model.predict(yeni_veri)


##################################
# Tahmin Başarısını Değerlendirme
##################################

# Train seti üzerinden tahmin yapıyoruz çünkü modeli train seti üzerinden kurduk.

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# TRAIN R-KARE
reg_model.score(X_train, y_train)

# Test RMSE
y_pred  =reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# TEST R_KARE
reg_model.score(X_test, y_test)

# 10 Katlı CV (Cross Validation) RMSE:
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))
# 1.69 - Veri seti küçük olduğu için cv'yi tüm veri setine uyguladık

# 5 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))


##############################################################
# Simple Linear Regression with Gradient Descent from Scratch
##############################################################

'''
    BU KISIMDA GRADIENT DESCENT'İ SIFIRDAN KENDİMİZ YAZDIK
'''


'''
    * Gradient Descent is a common optimization algorithm used to train machine learning models
    in neural networks. 
    
    * By training on data that these models can learn over time, and because 
    they're learning over time, they can improve their accuracy. 
    
    ** TYPES OF GRADIENT DESCENT:
        * Batch Gradient Descent       -> Tamamını belleğe alır, hepsine birlikte bakar
        * Stochastic Gradient Descent  -> Tek tek bakar
        * Mini-Batch Gradient Descent  -> Küçük gruplar halinde belleğe alır, grup grup bakar
        
        ---------------------------------------------------------------------------------------------
        | PARAMETERS      | Batch GD Algorithms | Mini Batch GD Algorithm | Stochastic GD Algorithm |
        ---------------------------------------------------------------------------------------------
        | Accuracy        |         HIGH        |        MODERATE         |          LOW            | 
        ---------------------------------------------------------------------------------------------
        | Time Consuming  |         MORE        |        MODERATE         |          LESS           | 
        --------------------------------------------------------------------------------------------- 
    
    ** It can struggle to find the global minimum in non-convex problems. (FIGURE-1)    
        
'''

# Cost Function
def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0 # sum of squared error

    for i in range(0, m):
        y_hat = b + w * X[i] # y_hat = y_pred
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse

# update_weights
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y) # Tüm gözlem birimleri inceleneceği için len'e ihtiyacımız var
    b_deriv_sum = 0
    w_deriv_sum = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]

    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w

# train fonksiyonu
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                            cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)

        # 100 iterasyonda bir rapor ver
        if i % 100 == 0:
            print("iter={:d}   b={:.2f}   w={:.4f}   mse={:.2f}".format(i, b, w, mse))

    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w


df = pd.read_csv("datasets/advertising.csv")

X = df["radio"]
Y = df["sales"]

'''
    PARAMETRE -> Modelin veriyi kullanarak, veriden hareketle, bulduğu değerlerdir
    HİPERPARAMETRE -> Veri setinden bulunamayan, kullanıcı tarafından ayarlanması gereken parametrelerdir 
'''

# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 10000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)



