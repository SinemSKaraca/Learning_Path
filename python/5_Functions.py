'''
    Methods vs functions:
        * Functions can exist independently, meaning they are not tied to any specific object or class.
        * A method is a function that is associated with an object or a class. It is called on an object,
        meaning it operates on data that belongs to that object.
        * Methods are specialized for OOP
'''

# parametre olarak gönderilen name bilgisi değişmedi!!
# dışardaki değişkenin farklı bir kopyası oluşturulur.
# Value Type
def changeName(n):
    n = "Ada"

name = "Yigit"

print("Before the function..: ", name)
changeName(name)
print("After the function..: ", name)

# Parametre olarak gönderilen liste kalıcı olarak değişti!
# çünkü buradaki n listenin adresi. Adresin içindeki değer değişiyor.
# Listenin kopyası değil direkt kendi adresi parametre olarak gönderilir.
# Reference Type
def change(n):
    n[0] = "Istanbul"

cities = ["Ankara", "Sakarya"]

print("Before..: ", cities)
change(cities)
print("After..: ", cities)

# Yukarıdaki ile benzer bir işlem şu şekildedir:
cities = ["Ankara", "Sakarya"]
n = cities
n[0] = "Sinop"

print(cities)

# atama yapmadan listenin kopyasını çıkarma (slicing ile):
# n'deki değişiklikler cities dizisini değiştirmez.
cities = ["Ankara", "Sakarya"]
n = cities[:]
n[0] = "Sinop"

print(cities)
print(n)

# Aynı şekilde metot içine de liste kopyası gönderebiliriz:
def change(n):
    n[0] = "Istanbul"

cities = ["Ankara", "Sakarya"]

print("Before..: ", cities)
change(cities[:]) # !!!!!!!!!
print("After..: ", cities)

# ------------------------------------------------------------------------------
# PARAMETRE OLARAK TUPLE GÖNDERME: (*params)

# NOTE: In python * symbol is used to unpack a collection into individual
# elements when passing arguments to a function.

# Fonksiyona İstediğimiz Kadar Parametre Gönderebiliriz:
def add(*params):
    print(params) # tuple
    return sum(params)

print("2 parameters..: ", add(20, 50))
print("6 parameters..: ", add(10,2, 45, 23, 2, 1))

# ------------------------------------------------------------------------------
# PARAMETRE OLARAK DICTIONARY GÖNDERME: (**params)

# Yukarıdaki gibi istediğim kadar parametre göndereceğim fakat bu parametrelerin
# hangi değerleri temsil ettiği farklı ve önemliyse:
def displayUser(**params): # **: Bir dictionay geleceğini bildirir
    for key, value in params.items():
        print("{} is {}".format(key, value))

displayUser(name = "Ali", age = 23, city = "Sakarya")
displayUser(name = "Veli", age = 47, city = "Istanbul", phone = "12364")

# ------------------------------------------------------------------------------

def myFunc(a, b, *args, **kwargs):
    print(a)
    print(b)
    print("Type..: ", type(args), args)
    print("Type..: ", type(kwargs), kwargs)

myFunc(10,20, 30, 40, 50, key1 = "val_1", key2 = "val_2")

# --------------------------------------------------------------------------------------

# 1: Gönderilen bir kelimeyi belirtilen sayı adedince ekranda gösteren fonksiyon:
times = int(input("How many times do you want..: "))
word = input("Enter a word..: ")

def show(times, word):
    for i in range(0, times):
        print(word)

show(times, word)

# 2: Kendine gönderilen sınırsız sayıdaki parametreyi bir listeye çeviren fonks:
def toList(*args):
    result = list(args)
    print(type(result))
    return result

numbers = (12, 90, 56, 73)
converted_list = toList(*numbers)
print(converted_list)

# 2.YOL:
def toList(*params):
    list = []

    for param in params:
        list.append(param)

    return list

result = toList(10, 23, 30, "Merhaba")
print(result)

# 3: Gönderilen iki sayı arasındaki tüm asal sayıları bulan fonksiyon:
num_1 = int(input("Enter number 1..: "))
num_2 = int(input("Enter number 2..: "))


def prime_number_finder(num1, num2):
    i = 2
    prime = []
    for num in range(num1, num2 + 1):
        while i < num - 1:
            if num % i == 0:
                break
            i += 1
        else:
            prime.append(num)
        i = 2
    return prime

print(prime_number_finder(num_1, num_2))

# 2.YOL:
def prime_number_finder(num1, num2):
    prime = []
    for num in range(num1, num2 + 1):
        if num > 1:  # Prime numbers are greater than 1
            is_prime = True
            for i in range(2, int(num ** 0.5) + 1):  # Check divisibility up to sqrt(num)
                if num % i == 0:
                    is_prime = False
                    break
            if is_prime:
                prime.append(num)
    return prime

print(prime_number_finder(num_1, num_2))

# 4: Kendisine gönderilen bir sayının tam bölenlerini bir liste şeklinde döndüren fonks:
num = int(input("Enter a number..: "))

def divisors(num):
    divisors = []
    for i in range(1, num + 1):
        if num % i == 0:
            divisors.append(i)
        i += 1
    return divisors

print(divisors(num))

# --------------------------------------------------------------------------------------

# LAMBDA, MAP and FILTER:
#------------------------

# MAP:

# Map, listenin tüm elemanları üzerinde fonksiyonun işlemini yapmak için kullanılır.
def square(num): return num ** 2

numbers = [1, 3, 5, 9]

result = list(map(square, numbers))

print(result)

# Yazdırma için 2.yol:
for item in map(square, numbers):
    print(item)

# LAMBDA:

result = list(map(lambda num: num ** 3, numbers))

# ---------------------

square_ = lambda x: x ** 2
numbers = [1, 2, 5, 6]

result = list(map(square_, numbers))
print(result)

result = square_(3)
print(result)

# FILTER:

_numbers = [1, 3, 5, 9, 10, 4]

def check_even(num):
    return num % 2 == 0

# true olarak belirtilen değerler listeye eklenir.
print(list(filter(check_even, _numbers)))

# Şu şekilde de yapabilirsin ama asıl kullanım yukardaki
def check_even(num):
    if num % 2 == 0:
        return num

# ---------------------

result = list(filter(lambda num: num % 2 == 0, _numbers))
print(result)

# --------------------------------------------------------------------------------------

# Fonksiyon Kapsamları:

x = 50
def test():
    global x # dışardaki x manipüle edilebilir
    print(f"x: {x}")

    x = 100
    print(f"changed x to {x}")

print("Global x before..: ", x)
test()
print("Global x after..: ", x)

# --------------------------------------------------------------------------------------

# BANKAMATİK UYGULAMASI:
#-----------------------

'''
    Burada verileri (veliHesap vs) bir dictionary yapısında tuttuğumuzdan aslında bir referans
    üzerinde oynamalar yapıyoruz. Bu nedenle fonksiyonda bir değişiklik yapıldığında bu değişiklikler
    kalıcı oluyor.
'''

VeliHesap = {
    "ad": "Veli Vel",
    "hesapNo": "13245678",
    "bakiye": 3000,
    "ekBakiye": 2000
}

AliHesap = {
    "ad": "Ali Al",
    "hesapNo": "12345678",
    "bakiye": 2000,
    "ekBakiye": 1000
}

def paraCek(hesap, miktar):
    print(f"Merhaba {hesap['ad']}!")

    if miktar <= hesap['bakiye']:
        hesap["bakiye"] -= miktar
        print("Paranizi alabilirsiniz.")
    else:
        toplam = hesap["bakiye"] + hesap["ekBakiye"]
        if toplam >= miktar:
            ekHesapKullanimi = input("Ek hesap kullanılsın mı? (e/h)")
            if ekHesapKullanimi == "e":
                ekHesapKullanilacakMiktar = miktar - hesap["bakiye"]
                hesap["bakiye"] = 0
                hesap["ekBakiye"] -= ekHesapKullanilacakMiktar
                print("Paranizi alabilirsiniz.")
            else:
                print(f"{hesap['hesapNo']} numaralı hesabinizda {hesap['bakiye']} TL bulunmaktadir")
        else:
            print("Bakiye yetersiz!")

def bakiyeSorgula(hesap):
    print(f"{hesap['hesapNo']} nolu hesabınızda {hesap['bakiye']} TL bulunmaktadır. Ek hesap limitiniz ise {hesap['ekBakiye']} TL'dir.")

paraCek(VeliHesap, 4000)
bakiyeSorgula(VeliHesap)