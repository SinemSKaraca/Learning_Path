# Nested Functions:
# ----------------

def greeting(name):
    print("Hello " + name)
    # Function doesn't have a return statement,
    # so by default, it returns "None"


print(greeting("sinem"))
print(greeting)

sayHello = greeting

print(sayHello)
print(greeting)

del sayHello

print(sayHello)
print(greeting)


# -------------

# ENCAPSULATION:
def outer(num1):
    print("outer")

    def inner_increment(num1):
        print("inner")
        return num1 + 1

    num2 = inner_increment(num1)
    print(num1, num2)


outer(10)
# inner_increment(10) -> Bu fonksiyonu tek başına kullanamam. Sadece outer fonksiyonu kapsamında kullanılabilir.

# -------------

def factorial(number):
    if not isinstance(number, int):
        raise TypeError("Number must be an integer!")

    if not number >= 0:
        raise ValueError("Number must be zero or positive")

    def inner_factorial(number):
        if number <= 1:
            return 1

        return number * inner_factorial(number - 1)

    return inner_factorial(number)

try:
    print(factorial(4))
except Exception as e:
    print(e)

# -------------

# FONKSİYONDAN FONKSİYON DÖNDÜRME:
def taban(number):

    def us(power):
        return number ** power

    return us

two = taban(2)
print(two(3))

three = taban(3)
print(three(4))

# ----

def yetki_sorgula(page):
    def inner(role):
        if role == "Admin":
            return "{} rolünün {} sayfasına ulaşabilir".format(role, page)
        else:
            return "{} rolünün {} sayfasına ulaşamaz".format(role, page)

    return inner

user1 = yetki_sorgula("Product Edit")
print(user1("Admin"))
print(user1("User"))

def islem(islem_adi):
    def toplama(*args):
        toplam = 0
        for i in args:
            toplam += i
        return toplam

    def carpma(*args):
        carpim = 1
        for i in args:
            carpim *= i
        return carpim

    if islem_adi == "toplama":
        return toplama
    else:
        return carpma

toplama = islem("toplama")
print(toplama(2, 4, 8))

carpma = islem("carpma")
print(carpma(2, 3, 12))

# -------------

# FUNCTIONS AS PARAMETERS:

def toplama(a, b):
    return a + b

def cikarma(a, b):
    return a - b

def carpma(a, b):
    return a * b

def bolme(a, b):
    return a / b

# f1, f2.. fonksiyonların referanslarını saklayacak parametreler
def islem(f1, f2, f3, f4, islem_adi):
    if islem_adi == "toplama":
        print(f1(2,3))
    elif islem_adi == "cikarma":
        print(f2(5, 3))
    elif islem_adi == "carpma":
        print(f3(3, 4))
    elif islem_adi == "bolme":
        print(f4(10, 2))
    else:
        print("Gecersiz ıslem!")

islem(toplama, cikarma, carpma, bolme, "toplama")
islem(toplama, cikarma, carpma, bolme, "cikarma")
islem(toplama, cikarma, carpma, bolme, "carpma")
islem(toplama, cikarma, carpma, bolme, "bolme")
islem(toplama, cikarma, carpma, bolme, "bolmee")

# -------------

# DECORATOR FUNCTIONS:
# Bir fonksiyona bir özellik eklemek istediğimizde kullanıyoruz

def my_decorator(func):
    def wrapper():
        print("Fonksiyondan önceki işlemler")
        func()
        print("Fonksiyondan sonraki işlemler")

    return wrapper

@my_decorator # sayHello = my_decoder(sayHello) ile eş değer
def sayHello():
    print("Hello!")

def sayGreeting():
    print("Greeting!")

sayHello()

# Bu işlemi fonksiyonun başına @decorator_name ekleyip fonksiyonu
# çağırarak da yapabiliriz. (Yukarıda yaptık)
sayHello = my_decoder(sayHello)
sayHello()

# ----

def my_decorator(func):
    def wrapper(name):
        print("Fonksiyondan önceki işlemler")
        func(name)
        print("Fonksiyondan sonraki işlemler")

    return wrapper

@my_decorator
def sayHello(name):
    print("Hello " + name)

sayHello("Ali")

# ----

import math
import time

def usalma(a, b):
    start = time.time()
    time.sleep(1)

    print(math.pow(a, b))

    finish = time.time()
    print("Fonksiyon " + str(finish - start) + " saniye sürdü")

def faktoriyel(num):
    start = time.time()
    time.sleep(1)

    print(math.factorial(num))

    finish = time.time()
    print("Fonksiyon " + str(finish - start) + " saniye sürdü")

usalma(2, 3)
faktoriyel(4)

# Yukarıdaki örneği decorator kullanarak tekrar yapalım:
def calculate_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        time.sleep(1)

        func(*args, **kwargs)

        finish = time.time()
        print("Fonksiyon " + func.__name__ + " " + str(finish - start) + " saniye sürdü.")

    return wrapper

@calculate_time
def usalma(a, b):
    print(math.pow(2, 3))

@calculate_time
def faktoriyel(num):
    print(math.factorial(num))

usalma(2, 3)
faktoriyel(4)