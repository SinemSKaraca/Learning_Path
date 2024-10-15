values = 1, 2, 3, 4, 5

x, y, z = values # bu haliyle hata alırız

x, y, *z = values # böyle yaparak z'yi liste haline getirebiliriz
type(z)

print(x, y, z)
print(x, y, z[0])

# --------------------------------------------------------------------------

x, y, z = 2, 5, 107

numbers = 1, 5, 7, 10, 6

# 1: Kullanıcıdan alınan iki sayının çarpımı ile x, y, z toplamının farkı nedir?
a = input("1st num: ")
b = input("2nd num: ")

result = int(a) * int(b) - (x + y + z)

# 2: y'nin x'e kalansız bölümünü hesaplayınız
result = y // x

# 3: (x, y, z) toplamının mod 3'ü nedir?
result = (x + y + z) % 3

# 4: y'nin x. kuvvetini hesaplayınız
result = y ** x

# 5: x, *y, z = numbers işlemine göre z'nin küpü kaçtır?
x, *y, z = numbers
z ** 3

# 6: x, *y, z = numbers işlemine göre y'nin değerleri toplamı kaçtır?
x, *y, z = numbers
result = y[0] + y[1] + y[2]

# --------------------------------------------------------------------------

# 1: Girilen iki sayıdan hangisi büyük:
a = int(input("a: "))
b = int(input("b: "))

result = a > b
print(f"a: {a} is greater than b: {b} -- {result}")

# 2: Kullanıcıdan iki vize (%60) ve final (%40) notunu alıp ortalama hesaplama:
#    Eğer ortalama 50'den büyükse geçti, değilse kaldı yazdırın
midterm_1 = float(input("1st midterm..: "))
midterm_2 = float(input("2ns midterm..: "))
final = float(input("final..: "))

average = (midterm_1 + midterm_2) * 0.6 + final * 0.4

print(f"Your average is..: {average} and the state of your class (passed=true)..: {average>=50}")

# 3: Girilen bir sayının tek mi çift mi olduğunu yazdırma:
num = int(input("a..: "))
isEven = num % 2 == 0

print(f"Number {num} is an even number..: {isEven}")

# 4: Girilen bir sayının negatif mi pozitif mi olduğunu yazdırma:
a = int(input("a..: "))
isPositive = a > 0

print(f"Number {a} is a positive number..: {isPositive}")

# 5: Parola ile email bilgisini isteyip doğruluğunu kontrol ediniz.
#    (email: email@sadikturan.com parola: abc123)
email = input("Enter the email..: ")
password = input("Enter the passward..: ")

isEmail = (email == "email@sadikturan.com".strip())
isPassword = (password == "abc123")

print(f"Your email is {isEmail} and password is {isPassword}")

# --------------------------------------------------------------------------

# 1: Girilen bir sayının 0-100 arasında olup olmadığını kontol ediniz.
a = int(input("a..: "))
isInBetween = (a > 0) and (a < 100)

print(f"Number {a} is in between 0 and 100..: {isInBetween}")

# 2: Girilen bir sayının pozitif çift sayı olup olmadığını kontrol ediniz.
a = int(input("a..: "))
isPosEven = (a > 0) and (a % 2 == 0)

print(f"Number {a} is a positive and even number..: {isPosEven}")

# 3: Email ve parola bilgileri ile giriş kontrolü yapınız.
_email = "email@sadikturan"
_passw = "abc123"

email = input("Email..: ")
passw = input("Password..: ")

isTrue = (_email == email) and (_passw == passw)

print(f"Email {email} and password {passw} are {isTrue}")

# 4: Girilen 3 sayıyı büyüklük olarak karşılaştırınız.
a = int(input("a..: "))
b = int(input("b..: "))
c = int(input("c..: "))

isA = (a > b) and (a > c)
print(f"a is the biggest number..: {isA}")

isB = (b > a) and (b > c)
print(f"b is the biggest number..: {isB}")

isC = (c > a) and (c > b)
print(f"c is the biggest number..: {isC}")

# 5: Kullanıcıdan 2 vize (%60) ve final (%40) notunu alıp ortalama hesaplayınız
#    Eğer ortalama 50'den yüksekse geçti değilse kaldı yazın
#    a-) Ortalama 50 olsa bile final notu en az 50 olmalıdır.
#    b-) Finalden 70 alındığında ortalamanın önemi olmasın.

midterm_1 = float(input("Mideterm_1..: "))
midterm_2 = float(input("Mideterm_2..: "))
final = float(input("Final..: "))

average = (midterm_1 + midterm_2) * 0.4 + final * 0.6
isPass = (average>=50 and final>=50) or final>=70

print(f"The Student Passed the Class..: {isPass}")

# 6: Kişinin ad, kilo ve boy bilgilerini alıp kilo indekslerini hesaplayınız.
#    Formül: (Kilo / boy uzunluğunun karesi)
#    Aşağıdaki tabloya göre kişi hangi gruba girmektedir?
#    0-18.4    => Zayıf
#    18.5-24.9 => Normal
#    25.0-29.9 => Fazla Kilolu
#    30.0-34.9 => Şişman (Obez)

name = input("Name..: ")
weight = float(input("Weight..: "))
height = float(input("Height..: "))

proportion = (weight / height ** 2)
isThin = (proportion > 0) and (proportion <= 18.4)
isFit = (proportion > 18.5) and (proportion <= 24.9)
isFat = (proportion > 25.0) and (proportion <= 29.9)
isObese = (proportion > 30.0) and (proportion <= 34.9)

print(f"{name} with the weight of {weight} and height of {height} is Thin...: {isThin}")
print(f"{name} with the weight of {weight} and height of {height} is Fit....: {isFit}")
print(f"{name} with the weight of {weight} and height of {height} is Fat....: {isFat}")
print(f"{name} with the weight of {weight} and height of {height} is Obese..: {isObese}")

# --------------------------------------------------------------------------

# IDENTITY OPERATOR -- is
x = y = [1, 2, 3]
z = [1, 2, 3]

# listelerin kendileri kıyaslanır:
print("x == y :", x == y)
print("x == z :", x == z)

# referansların gösterdikleri adresler kıyaslanır:
print("x is y :", x is y)
print("x is z :", x is z)

x = [1, 2, 3]
y = [2, 4]

del x[2]
y[1] = 1
y.reverse()

print("x == y :", x == y)
print("x is y :", x is y)

# MEMBERSHIP OPERATOR -- in
x = ["apple", "banana"]
print("banana" in x)

name = "Sinem"
print("e" in name)
print("n" not in name)