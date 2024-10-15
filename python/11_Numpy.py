'''
    NUMPY:
        NumPy is a powerful library that provides the foundation for numerical computation in Python.
        Performance: NumPy operations are implemented in C, making them much faster than equivalent operations performed with Python lists.
        Memory Efficiency: NumPy arrays use less memory than Python lists because they are stored as contiguous blocks of memory.
        Flexibility: NumPy's ability to handle different types of data and its extensive mathematical capabilities make it
                     essential for scientific computing, data analysis, and machine learning.
'''

import numpy as np

# pyhton list:
py_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# numpy array:
np_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

print(type(py_list))
print(type(np_array))

# çok boyutlu python list:
py_multi = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# çok boyutlu numpy array:
np_multi = np_array.reshape(3, 3)

print(py_multi)
print(np_multi)

print(np_array.ndim)
print(np_multi.ndim)

print(np_array.shape)
print(np_multi.shape)

# -------

import numpy as np

result = np.array([1, 3, 5, 7, 9])
result = np.arange(1, 10) # 10 dahil değil
result = np.arange(10, 100, 3)
result = np.zeros(10)
result = np.ones(10)

# NP.LINSPACE(): Verilen başlangıç ve bitiş değerlerini eşit aralıklarla böler
# LINSPACE - Linear Space
# 4.parametre olarak endpoint=False giderse stop değeri dahil olmaz
result = np.linspace(0, 100, 5)
result = np.random.randint(0, 10) # 10 dahil değil
result = np.random.randint(20)
result = np.random.randint(1, 10, 3) # son parametre eleman sayısını belirtir
result = np.random.rand(5)

print(result)

np_array = np.arange(50)
np_multi = np_array.reshape(5, 10)
print(np_multi.sum(axis=1)) # sütun
print(np_multi.sum(axis=0)) # satır

rnd_numbers = np.random.randint(1, 100, 10)
print(rnd_numbers)
max_ = rnd_numbers.max()
print(max_)
min_ = rnd_numbers.min()
print(min_)
mean_ = rnd_numbers.mean()
print(mean_)

# index of the max ve min value:
argmax_ = rnd_numbers.argmax()
print(argmax_)
argmin_ = rnd_numbers.argmin()
print(argmin_)

# -------

import numpy as np

numbers = np.array([0, 5, 10, 15, 20, 25, 50, 75])

result = numbers[5]
result = numbers[-1]
result = numbers[0:3] # numbers[:3] ile aynı
result = numbers[3:]
result = numbers[::]
result = numbers[::-1] # son parametre adım sayısıdır

numbers2 = np.array([[0, 5, 10], [15, 20, 25], [50, 75, 85]])
result = numbers2[0]
result = numbers2[2]
result = numbers2[0, 2]
result = numbers2[2, 1]
result = numbers2[:, 2] # Tüm satırların 2. sütunu
result = numbers2[:, 0:2]
result = numbers2[0:2, 0:2] # result = numbers2[:2, :2]
result = numbers2[-1, :]

print(result)

# Referans kopyalama:
arr1 = np.arange(0, 10)
arr2 = arr1

print(arr1)
print(arr2)

arr2[0] = 20

print(arr1)
print(arr2)

# Bir diziyi başka bir diziye kopyalama:
arr3 = np.arange(0, 10)
arr4 = arr3.copy() # arr3'ün içeriği farklı bir adresteki yeni diziye kopyalandı

print(arr3)
print(arr4)

arr3[0] = 99

print(arr3)
print(arr4)

# -------

import numpy as np

numbers1 = np.random.randint(10, 100, 6)
numbers2 = np.random.randint(10, 100, 6)

print(numbers1)
print(numbers2)

# listelerde eşit sayıda eleman olmalı
# Bu değişimler kalıcı değil
result = numbers1 + numbers2 # aynı indexteki elemanları toplar
result = numbers1 - numbers2
result = numbers1 - 10
result = numbers2 + 10
result = numbers1 * numbers2
result = numbers1 / numbers2
result = np.sin(numbers1)
result = np.cos(numbers1)
result = np.sqrt(numbers1)

print(result)

m_numbers1 = numbers1.reshape(2, 3)
m_numbers2 = numbers2.reshape(2, 3)

print(m_numbers1)
print(m_numbers2)

result = np.hstack((m_numbers1, m_numbers2))
result = np.vstack((m_numbers1, m_numbers2))

result = numbers1 >= 5

'''
    ** result = numbers1[result] filters the numbers1 array, keeping only the elements
    that correspond to "True" values in the "result" Boolean array.
    ** This line uses BOOLEAN INDEXING.
    ** NumPy will return only the elements of nımbers1 where the 
    corresponding value in result is True.
'''
result = numbers1 % 2 == 0
result = numbers1[result]

print(result)

# ----------------------------------------------------------------------

# UYGULAMA:

import numpy as np

n_array = np.array([10, 15, 30, 45, 60])
print(n_array)

n_array_2 = np.arange(5, 15)
print(n_array_2)

n_array_3 = np.arange(50, 100, 5)
print(n_array_3)

n_array_4 = np.zeros(10)
print(n_array_4)

n_array_5 = np.ones(10)
print(n_array_5)

n_array_6 = np.linspace(0, 100, 5)
print(n_array_6)

n_array_7 = np.random.randint(10, 30, 5)
print(n_array_7)

# RANDN(): Negatif sayılar da üretir
# -1 ile 1 arasında rastgele 10 tane sayı üretme
n_array_8 = np.random.randn(10)
print(n_array_8)

n_array_9 = np.random.randint(10, 50, 15).reshape(3,5)
print(n_array_9)

# Üretilen matrisin satır ve sütun sayıları toplamı:
matris = np.random.randint(10, 50, 15).reshape(3,5)
rowTotal = matris.sum(axis=0)
colTotal = matris.sum(axis=1)

print(rowTotal)
print(colTotal)

# Yukarıdaki matrisin max, min ve ortalama değerleri
max_ = n_array_9.max()
print(max_)

min_ = n_array_9.min()
print(min_)

mean_ = n_array_9.mean()
print(mean_)

argmax_ = n_array_9.argmax()
print(argmax_)

n_array_10 = np.arange(10, 20)
print(n_array_10[0:3])
print(n_array_10[::-1])

matris = np.random.randint(10, 50, 15).reshape(3,5)
print(matris[0])
print(matris[2, 3])
print(matris[:, 0])
print(matris ** 2)

even = matris[matris % 2 == 0]
pos_even = even[even > 0]
print(pos_even)
