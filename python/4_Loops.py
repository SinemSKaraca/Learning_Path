numbers = [1, 3, 5, 7, 9, 12 ,19, 21]

# 1: 3'ün katı olanlar:
for num in numbers:
    if(num % 3 == 0):
        print(num);

# 2: Listedeki sayıların toplamı:
total = 0
for num in numbers:
    total = total + num

print(total)

# 3: Listedeki tek sayıların karesi:
for num in numbers:
    if(num % 2 != 0):
        print(num ** 2)

# ----------------------------------------------------------------

cities = ["kocaeli", "istanbul", "ankara", "izmir", "rize"]

# 4: Şehirlerden en fazla 5 karakterli olanlar:
for city in cities:
    if(len(city) <= 5):
        print(city)

# ----------------------------------------------------------------

products = [
    {"name": "samsung  S6", "price": "3000"},
    {"name": "samsung  S7", "price": "4000"},
    {"name": "samsung  S8", "price": "5000"},
    {"name": "samsung  S9", "price": "6000"},
    {"name": "samsung  S10", "price": "7000"}
]

# 5: ürünlerin fiyatları toplamı:
total = 0

for p in products:
    price = int(p["price"])
    total += price

print(total)

# 6: Ürünlerden fiyatı en fazla 5000 olan ürünler:
for p in products:
    if(int(p["price"]) <= 5000):
        print(p)

# ----------------------------------------------------------------

# NOT --> " " (space) karakteri FALSE olarak değerlendirilir!

numbers = [1, 3, 5, 7, 9, 12 ,19, 21]

# 1: Listeyi while ile ekrana yazdır:
i = 0
while(i < len(numbers)):
    print(numbers[i])
    i += 1

# 2: Başlangıç ve bitiş değerlerini kullanıcıdan alıp aradaki tüm
#    tek sayıları ekrana yazdır:
start = int(input("start..: "))
end = int(input("end..: "))

while(start < end):
    if(start % 2 != 0):
        print(start)
    start += 1

# 3: 1-100 arasındaki sayıları azalan şekilde ekrana yazdır:
i = 0
while(i < 100):
    print(100 - i)
    i += 1

# 4: Kullanıcıdan alacağınız 5 sayıyı ekranda sıralı bir şekilde yazdır:
numbers = []

i = 0
while(i < 5):
    num = int(input("Enter a number..: "))
    numbers.append(num)
    i += 1

numbers.sort()
print(numbers)

# 5: Kullanıcıdan alınan sınırsız ürün bilgisini ürünler listesi içinde sakla:
#    ** ürün sayısını kullanıcıya sor
#    ** dictionary listesi yapısı (name, price) şeklinde olsun
#    ** ürün ekleme işlemi bittiğinde ürünleri ekranda while ile listele

products = []

amount = int(input("How many products will you add..: "))
i = 0

while(i < amount):
    name = input("Name of the product..: ")
    price = input("Price of the product..: ")
    products.append({"name": name, "price": price})
    i += 1

for p in products:
    print(f"Name of the product: {p['name']}\nPrice of the product: {p['price']}")
    print("----------------------------------------------")

# ----------------------------------------------------------------

# RANGE:

# içerden dışarı doğru: 10'dan 100' onar onar git.(100 dahil değil)
# İlgili sayıları bir listeye at
print(list(range(10, 110, 10)))

# ENUMERATE:

greeting = "Hello World"
index = 0
for letter in greeting:
    print(f"index: {index} letter: {letter}")
    index += 1

# greeting bir stringdir ve bir string üzerinde döngü yapıldığında,
# her iterasyon yalnızca bir karakter döner. Ancak, döngüde iki
# değişkene (index ve letter) atama yapmaya çalışıyorsunuz, bu da bu hataya neden olur.
# Eğer hem indeks hem de harfi almak istiyorsanız, enumerate() fonksiyonunu kullanmalısınız.
for index, letter in greeting:
    print(f"index: {index} letter: {letter}")
    index += 1

# enumerate ile:
for item in enumerate(greeting, 1):
    print(item)

# enumerate kullanarak _ 2:
for index, letter in enumerate(greeting):
    print(f"index: {index} letter: {letter}")

# ZIP:

list1 = [1, 2, 3, 4, 5]
list2 = ["a", "b", "c", "d", "e"]

print(list(zip(list1, list2)))

for item in zip(list1, list2):
    print(item)

for a, b in zip(list1, list2):
    print(a)

# ----------------------------------------------------------------

# LIST COMPREHENSION:

# list comp. ile:
numbers = [x ** 2 for x in range(10)]
print(numbers)

# normal döngü ile:
for x in range(10):
    print(x ** 2)

numbers = [x*x for x in range(10) if x%3==0]
print(numbers)

resuls = [x if x % 2 == 0 else 'TEK' for x in range(1, 10)]

# iç içe döngü _ normal döngü ile:
result = []
for x in range(3):
    for y in range(3):
        result.append((x,y)) # parantez kullanarak tuple şeklinde argüman olarak gönderdik

# list comp. ile:
numbers = [(x, y) for x in range(3) for y in range(3)]
print(numbers)

# ----------------------------------------------------------------

# Sayı Tahmin Uygulaması:
#------------------------

'''
    1-100 arasında rastgele üretilecek bir sayıyı aşağı-yukarı
    ifadeleri ile buldurmaya çalışı.
    ** "random" modülü kullanın.
    ** 100 üzerinden puanlama yapın. Her tur 20 puan.
    ** Hak bilgisini kullanıcıdan alın ve her soru belirtilen can sayısı
       üzerinden hesaplansın.
'''

import random

# 1 ve 100 dahil
num = random.randint(1, 100) # randint: tam sayı üretir
chance = int(input("Have many chance do you need..: "))
punishment = 100 / chance
guess = int(input("Guess the number..: "))

point = 100
i = 1
while (num != guess) and (chance - 1 > 0):
    i += 1
    chance -= 1
    point -= punishment
    if guess < num:
        print("Higher!")
    elif guess > num:
        print("Lower!")

    guess = int(input("Try again..: "))

if chance == 1 and guess == num:
    print("You got it on your {}. time! Your point is {}.".format(i, point))
else:
    print("You run out of chance. Want to play again?")

# ----------------------------------------------------------------

# Asal Sayı Uygulaması:
#----------------------

'''
    Girilen bir sayının asallığını kontrol edin.
    ** Asal sayı: 1 ve kendisi haricindeki sayılara bölünemeyen sayıdır.
'''

'''
    NOT!!!!!
        Python'da else ifadesini döngülerle birlikte kullanabilirsin.
        Genelde break olan döngülerde daha kullanışlı. Eğer döngüde break ile
        hiç karşılaşılmadan döngü biterse else ifadesine girer!
        Break ile karşılaşılırsa else ifadesine girilmez!
'''

num = int(input("Enter a number..: "))

i = num - 1
while i > 1:
    if num % i == 0:
        print("Number is not a prime number!")
        break
    i -= 1

else:
    print("Number is a prime number!")

# veya

num = int(input("Enter a number..: "))

is_prime = True
i = num - 1

while i > 1:
    if num % i == 0:
        is_prime = False  # Mark as not prime
    i -= 1
else:
    if is_prime:
        print("Number is a prime number!")
    else:
        print("Number is not a prime number!")





