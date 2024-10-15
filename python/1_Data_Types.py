x = input('1st num: ')
y = input('2nd num: ')

total = int(x) + int(y)

print(total)

# --------------------------------------------------------

pi = 3.14
r = input('Enter the radius: ') # str bilgi doner

area = pi * (int(r) ** 2)
circumference = 2 * pi * int(r)

print("area", area)
print("circumference", circumference)

# --------------------------------------------------------

name = 'Sinem'
surname = 'Karaca'
age = '23'

# Sadece str verileri bu şekilde birleştirebilirsin
# int kullanacaksan str ile tür dönüşümü yapmalısın
print("My name is " + name + " " + surname + " and\nI am " + age + " years old.")

# NOT: indexler sağdan sola -1, -2.. olarak gider!!

greeting = "My name is " + name + " " + surname + " and I am " + age + " years old."

print(len(greeting))
print(greeting[-1])
print(greeting[0:-5]) # 0'dan sondan 5. karaktere kadar
print(greeting[2:40:2]) # 2'den 40.indexe kadar ikişerli

# --------------------------------------------------------
# STRING FORMATTING
# --------------------------------------------------------

name = "Sinem"
surname = "Karaca"
age = 23

print("My name is {} {}".format(name, surname))
print("My name is {1} {0}".format(name, surname))
print("My name is {n} {s}".format(n=name, s=surname))
print("My name is {} {} and I am {} years old.".format(name, surname, age))

result = 200 / 700
print("The result is {}".format(result))
# sağ-virgülden önceki digit sayısı; sol-virgülden sonraki digit sayısı
print("The result is {r:1.4}".format(r=result))

# f-string
#---------

print(f"My name is {name} {surname} and I am {age} years old.")

# --------------------------------------------------------

website = "http://www.sadikturan.com"
course = "Python Kursu: Bastan Sona Python Programlama Rehberiniz (40 Saat)"

# 1: String uzunluğu yazdırma:
print(len(course))

# 2: Website içinden www stringini alma:
website[7:10]

# 3: Website içinden com stringini alma:
website[22:25]
website[len(website)-3:]

# NOT: Slicing işleminde başlangıç indexi bitiş indexinden küçük olmalı!!
# Bu nedenle aşağıdaki kod yanlış:
# course[-1:-15]

# 4: İlk 15 karakteri yazdırma:
course[0:15]

# 5: Son 15 karakteri yazdırma:
course[-15:] # sondan 15.karakterden son karaktere kadar

# 6: course stringini tersten yazdırma:
course[::-1]

# --------------------------------------------------------

s = "12345" * 5 # string ifadeyi 5 kez alır
print(s[::5])

# --------------------------------------------------------

name, surname, age, job = "Bora", "Yilmaz", 32, "Engineer"

# 6: Ekrana şu ifadeyi yazdır: My name is Bora Yilmaz, I'm 32 years old and I'm an engineer
print("My name is " + name + " " + surname + ", I'm " + str(age) + " years old and I'm an " + job)
print(f"My name is {name} {surname}, I'm {age} years old and I'm an {job}")
print("My name is {} {}, I'm {} and I'm an {}".format(name, surname, age, job))

# 7: Hello world ifadesindeki w'yi W ile değiştir:
str = "Hello world"
str = str[0:6] + "W" + str[-4:]
str = str.replace("w", "W")
# Yukarıda atama yapmazsan değişiklik kalıcı olmaz!!

# 8: abc ifadesini yan yana 3 defa yazdır:
s = "abc" * 3

# --------------------------------------------------------
# STRING METHODS
# --------------------------------------------------------

message = "Hello there. My name is Sinem Karaca"

print(message.upper())
print(message.lower())
print(message.title()) # her kelimenin baş harfi büyük harf olur
print(message.capitalize()) # stringin sadece ilk karakteri büyük olur

message = " Hello there. My name is Sinem Karaca"

print(message.strip()) # baştaki whitespace silinir
print(message.split()[0]) #defaultu boşluk karakteri
print(message.split("."))
print("*".join(message))

index = message.find("Sinem") # Kelimenin ilk harfinin indexi
# Kelime stringde yoksa -1 döner

isFound = message.startswith("H")
isFound = message.endswith("K")

print(message.replace("Sinem", "*****").replace("Karaca", "---"))

print(message.center(100)) # stringin sağ ve solundan ortalar

# --------------------------------------------------------
# UYGULAMA
# --------------------------------------------------------

website = "http://www.sadikturan.com"
course = "Python Kursu: Bastan Sona Python Programlama Rehberiniz (40 Saat)"

# 1: ' Hello World ' karakter dizisinin baş ve sondaki boşluk karakterlerini sil:
" Hello World ".strip()
" Hello World ".lstrip() # soldan siler
" Hello World ".rstrip() # sağdan siler

# 2: 'www.sadikturan.com' içindeki sadikturan bilgisi haricindeki karakterleri sil:
website.strip("http://www.com")

# 3: 'course' stringinin tüm karakterlerini küçük harf yap:
print(course.lower())

# 4: 'website' içinde kaç tane a karakteri var bul:
print(website.count("a"))
print(website.count("a", 0, 15))

# 5: 'website' www ile başlayıp com ile bitiyor mu bak:
print(website.startswith("www") and website.endswith("com"))

# 6: 'website' içinde .com ifadesi var mı bak:
# NOT: find() alt stringi bulamazsa -1 döndürür
#      index() alt stringi bulamazsa 'ValueError' hatası fırlatır
print(website.find(".com"))
print(website.find(".com", 0, 10))
print(course.find("Python"))
print(course.rfind("Python")) # aramaya sağdan başlar

print(website.index(".com"))

# 7: 'course' içindeki karakterlerin hepsi harf mi bak:
print(course.isalpha())

# 8: 'contents' ifadesini satırda 50 karakter içine yerleştirip sağına ve soluna * ekle:
'contents'.center(50, "*")
'contents'.ljust(50, "*")
'contents'.rjust(50, "*")

# 9: 'course' içindeki tüm boşluk karakterlerini '-' ile değiştir:
print(course.replace(" ", "-"))
print("Hello World".replace("World", "Dunya"))

# 10: 'course' karakter dizisini boşluk karakterlerinden ayır:
print(course.split())

# --------------------------------------------------------

brands = ["BMW", "Mercedes", "Opel", "Mazda"]

len(brands)

print(brands[0])

print(brands[len(brands)-1])

brands[-1] = "Toyota"

print("Mercedes" in brands)

brands[-2]

brands[0:3]

brands[2:] = ["Toyota", "Renault"]

brands + ["Cherry"]
brands.append("Audi")
brands.append("Nissan")

brands.remove(brands[len(brands)-1])

brands[::-1]
brands.reverse()

# --------------------------------------------------------

studentA = ["Yigit", "Bilgi", 2010, [70, 60, 70]]
studentB = ["Sena", "1999", 2010, [80, 80, 70]]
studentC = ["Ahmet", "1998", 2010, [80, 70, 90]]

print(f"{studentA[0]} {studentA[1]} is {2019-studentA[2]} years old and his grade average is {sum(studentA[3]) / len(studentA[3])}")

# --------------------------------------------------------

# Dictionary kullanmazsak:
cities = ["Sakarya", "Istanbul"]
plates = [54, 34]

print(plates[cities.index("Sakarya")])

# Dictionary kullanarak:
plates = {"Istanbul": 34, "Sakarya": 54}

print(plates["Sakarya"])

plates["Ankara"] = 6

users = {
    "sadikturan": {
        "age": 36,
        "email": "sadik@gmail.com",
        "address": "kocaeli",
        "phone": "123123"
    },
    "cinarturan": {
        "age": 2,
        "roles": ["admin", "user"],
        "email": "cinar@gmail.com",
        "address": "kocaeli",
        "phone": "123123"
    }
}

print(users["cinarturan"]["age"])

"""
students = { 
    "120": {
        "name": "Ali",
        "surname": "Yilmaz",
        "phone": "532 000 00 01"
    },
    "125": {
        "name": "Can",
        "surname": "Korkmaz",
        "phone": "532 000 00 02"
    },
    "128": {
        "name": "Volkan",
        "surname": "Yukselen",
        "phone": "532 000 00 03"
    }
} """

# students sözlüğünü kullanıcıdan aldığımız bilgilerle yukarıdaki gibi yapacağız
students = {}

number = input("student number: ")
name = input("student name: ")
surname = input("student surname: ")
phone = input("student phone: ")

students[number] = {
    "name": name,
    "surname": surname,
    "phone": phone
}

# yukarıdakini yapmanın farklı bir yolu:
students.update({
    number: {
        "name": name,
        "surname": surname,
        "phone": phone
    }
})

print(students)

student_id = input("Enter the student ID: ")
print("Student info..: ", students[student_id])

# --------------------------------------------------------

fruits = {"apple", "melon", "banana"}

# Set veri yapısı indexlenemez
# fruits[0] --> bu yanlış bir kullanım

# Set içinde bir eleman yalnızca bir kez bulunabilir.

# --------------------------------------------------------
# VALUE AND REFERENCE TYPES
# --------------------------------------------------------

# VALUE TYPES --> string, number
x = 5
y = 5

x = y

y = 10

print(x, y)

# REFERENCE TYPES --> list
a = ["apple", "banana"] # Burada a ve b listelerinin adreslerini taşıyan referanslardır.
b = ["apple", "banana"]

a = b # burada b'nin içindeki adresle a'nın içindeki adres eşitleniyor. Yani aynı alanı gösteriyorlar

b[0] = "avacado"

print(a, b)








