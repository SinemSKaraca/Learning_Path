'''
    * Her modül .py uzantılı ayrı bir dosyadır.
    * They allow you to group related functionality in one place and access it across multiple programs.
    * Modules: A module is typically a single file containing code. It can be a part of a library.
    * Libraries: A library is a collection of modules, often organized to solve a particular problem or provide specific functionality.

    ** HAZIR MODÜLLER:
        1- Standart Kütüphane Modülleri
        2- Third Party Modülleri

    *** pypi.org --> Python kütüphane havuzu - pip install <paket_name> ile kurulur
'''

# Bu şekilde modül ismini kullanmadan modülün elemanlarına direkt erişebiliriz
from math import *

# You need to use the module name (math) as a prefix when calling any function or
# accessing any variable from the module.
import math

value = factorial(2)
print(value)

'''
    Bir modülün fonksiyonunu kullanacağın zaman aynı isimde bir de fonksiyonun varsa kod içerisinde
    hangisi sonra tanımlanmışsa o kullanılır. Yani alttaki üsttekini ezer.
'''
def sqrt(x):
    print("x..: " + str(x))

from math import sqrt # Bunu yukarı alırsam kendi fonksiyonum çalışır

value = sqrt(4)
print(value)

# ------------------------------------------------------------------------------------

import random

names = ["ali", "yagmur", "deniz", "cenk"]

result1 = names[random.randint(0, len(names) - 1)]

# Liste içinden rastgele bir eleman seçme: RANDOM.CHOICE()
result2 = random.choice(names)

# Liste elemanlarını karşıtırma: RANDOM.SHUFFLE()
liste = list(range(10))
random.shuffle(liste)
result3 = liste

print(result3)

# Liste içinden rastgele n adet bilgi alma: RANDOM.SAMPLE()
liste = range(100)
result4 = random.sample(liste, 3)
print(result4)
result5 = random.sample(names, 2)
print(result5)

