# 1: Kullanıcıdan isim, yaş, eğitim bilgilerini isteyip ehliyet alabilme
#    durumunu kontrol ediniz Ehliyet alma koşulu en az 18 yaş ve eğitim durumu
#    lise ya da üniversite olmalıdır.

name = input("Name..: ")
age = int(input("Age..: "))
degree = input("Degree..: ")

if age >= 18 and (degree == "high school" or degree == "collage"):
    print("You can get your licance.")
else:
    print("You are not fit for the licance.")

# 2: Bir öğrencinin 2 yazılı bir sözlü notunu alıp hesaplanan ortalamaya göre
#    not aralığına karşılık gelen not bilgisini yazdırınız.
#    0-24   => 0
#    25-44  => 1
#    45-54  => 2
#    55-69  => 3
#    70-84  => 4
#    85-100 => 5

grade_1 = float(input("Enter your 1st grade..: "))
grade_2 = float(input("Enter your 2nd grade..: "))
grade_3 = float(input("Enter your 3rd grade..: "))

gpa = (grade_1 + grade_2 + grade_3) / 3
print("Your gpa is {}".format(gpa))

if gpa > 0 and gpa < 25:
    print("You get 0.")
elif gpa >= 25 and gpa < 45:
    print("You get 1.")
elif gpa >= 45 and gpa < 55:
    print("You get 2.")
elif gpa >= 55 and gpa < 70:
    print("You get 3.")
elif gpa >= 70 and gpa < 85:
    print("You get 4.")
else:
    print("You get 5.")

# 3: Trafiğe çıkış tarihi alınan bir aracın servis zamanını
#    aşağıdaki bilgilere göre hesaplayınız:
#    1.Bakım => 1. yıl
#    2.Bakım => 2. yıl
#    3.Bakım => 3. yıl
#    ** Süre hesabını alınan gün, ay, yıl bilgisine göre gün bazlı
#    hesaplayınız.
#    *** datetime modülünü kullanmanız gerekiyor
#    (simdi) - (2018/8/1) => gün

days = int(input("How many days have you been in traffic for..: "))

if days <= 365:
    print("1.service period.")
elif days > 365 and days <= 730:
    print("2.service period.")
elif days > 730 and days <= 365*3:
    print("3.service period.")
else:
    print("Wrong entry!")

# datetime kullanarak:
import datetime

date = input("When did your vehicle hit the road (Y/M/D)..: ")
date = date.split("/")
# print(date)

# date listesini datetime objesine çevirdik.
_datetime = datetime.datetime(int(date[0]), int(date[1]), int(date[2]))
now = datetime.datetime.now()
difference = now - _datetime
days = difference.days

if days <= 365:
    print("1.service period.")
elif days > 365 and days <= 730:
    print("2.service period.")
elif days > 730 and days <= 365*3:
    print("3.service period.")
else:
    print("Wrong entry!")