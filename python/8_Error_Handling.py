# ERROR:
# ------

# print(a) => NameError
# int("1a2") => ValueError
# print(10/0) => ZeroDivisionError
# print("denem"e) => SyntaxError

# ERROR HANDLING:
# ---------------

# Hata gelme ihtimali olan kodlar "try" bloğu içine alınır.
# Burada 0'a bölme hatası alma ihtimalimiz var.
try:
    x = int(input("x..: "))
    y = int(input("y..: "))
    print(x/y)
# öngördüğümüz hata türü
except ZeroDivisionError:
    print("y can't be equal to zero!")
except ValueError:
    print("Enter a numerical value for x and y!")

# -----------------------------

'''
    * Ayrı ayrı except yazmak yerine gelen hataları gruplayabiliriz:
        except (ZeroDivisionError, ValueError):
            print("Wrong value!")
'''

'''
    except: yazıp tür belirtmezsek e objesi kullanarak hata türünü almayız!
'''

try:
    x = int(input("x..: "))
    y = int(input("y..: "))
    print(x/y)
# öngördüğümüz hata türü
except (ZeroDivisionError, ValueError) as e: # e objesi
    print("Wrong value!")
    print(e) # Hatanın türünü alabiliriz

# -----------------------------

try:
    x = int(input("x..: "))
    y = int(input("y..: "))
    print(x/y)
except:
    print("Wrong value!")
# except çalışmazsa else çalışır
else:
    print("Everything is all right!")

# Bu özelliği şu tür uygulamalarda kullanabiliriz:
# Hata olduğu sürece döngü döner, hata alınmadığı zaman döngüden çıkar
while True:
    try:
        x = int(input("x..: "))
        y = int(input("y..: "))
        print(x / y)
    # Exception base class'tır ve alt class'ların yerine kullanılabilir
    except Exception as ex:
        print("Wrong value!", ex)
    else:
        break

# -----------------------------
# FINALLY: This block runs no matter what—whether an exception occurred or not.
# It's typically used for cleanup code, like closing files or releasing resources.

while True:
    try:
        x = int(input("x..: "))
        y = int(input("y..: "))
        print(x / y)
    # Exception base class'tır ve alt class'ların yerine kullanılabilir
    except Exception as ex:
        print("Wrong value!", ex)
    else:
        break
    finally:
        print("Error Handling ended.")

# -----------------------------------------------------------------------------

# RAISING AN EXCEPTION:
# ---------------------

'''
    
    Raising an exception in Python means intentionally triggering an error in 
    your code to indicate that something went wrong or to signal that a particular 
    condition has been met that requires special handling.

    When you raise an exception, you're telling Python to stop executing the current block 
    of code and jump to the nearest exception handler (try-except block) to manage the situation.
'''

x = 10

if x > 5:
    raise Exception("x can't be greater than 5!")

# -----------------------------

def check_password(psw):
    import re # regular expression
    if len(psw) < 8:
        raise Exception("Password must contain at least 7 letters!")
    elif not re.search("[a-z]", psw):
        raise Exception("Password must contain lowercase letter!")
    elif not re.search("[A-Z]", psw):
        raise Exception("Password must contain uppercase letter!")
    elif not re.search("[0-9]", psw):
        raise Exception("Password must contain numerical character!")
    elif not re.search("[_@$]", psw):
        raise Exception("Password must contain at least one of these: _@$!")
    elif re.search("\s", psw):
        raise Exception("Password must not contain space character!")

password = input("Enter your password..: ")

try:
    check_password(password)
except Exception as ex:
    print(ex)
else:
    print("Password is valid!")
finally:
    print("Validation is finished!")

# -----------------------------

'''
    NOTLAR:
        * "init" metotu constructor'dur. The constructor is automatically called 
          when you create a new object (or instance) of the class.
        * The "self" parameter refers to the instance of the class. When you create 
          a Person object, self represents that specific object.
        *  
'''
class Person:
    def __init__(self, name, year):
        if len(name) > 10:
            raise Exception("name contains more character than needed!")
        else:
            self.name = name

p = Person("Aliiiiiiiiiiii", 1998)

# -----------------------------------------------------------------------------

# UYGULAMA:
# ---------

liste = ["1", "2", "5a", "10b", "abc"]

# 1: Liste elemanları içindeki sayısal değerleri bulunuz
numeric_list = []

for item in liste:
    if item.isnumeric():
        numeric_list.append(item)

print(numeric_list)

# try-except kullanarak:
for x in liste:
    try:
        # gelen değerleri int'e çevirmeeye çalışıyoruz
        # gelen değer int'e çevrilemeyebilir. Bu durumda catch'e gidilir.
        result = int(x)
        print(result)
    except ValueError:
        continue

# ------------------

# 2: Kullanıcı "q" değerini girmedikçe aldığınız her inputun sayı
#    olduğundan emin olunuz. Aksi halde hata mesajı yazın.

''' 
    NOTE --> Python doesnt have built-in do-while loop
'''
while True:
    number = input("Enter a number")
    if number == "q":
        break
    try:
        int(number)
        print("The number you entered..: ", number)
    except ValueError:
        print("It is not a number!")


# ------------------

# 3: Girilen parola içinde türkçe karakter hatası veriniz
password = input("Enter your password (without Turkish characters!)..: ")
def check_password(psw):
    turkish_char = "çÇğĞıİöÖşŞüÜ"
    for letter in psw:
        if letter in turkish_char:
            raise Exception("Your password can't contain Turkish characters!")
        else:
            pass # This does nothing, the program just continues after this block

try:
    check_password(password)
except TypeError as err:
    print(err)

# ------------------

# 4: Faktöriyel fonksiyonu oluşturup fonksiyona gelen değer için
#    hata mesajı verin.
def factorial(x):
    x = int(x)
    if x < 0:
        raise Exception("Negative value!")
    result = 1
    for i in range(1, x + 1):
        result *= i
    return result

for x in [5, 10, 20, -3, '10a']:
    try:
        y = factorial(x)
    except Exception as err:
        print(err)
        continue # hata aldıktan sonra da devam etmemizi sağlar
    print(y)
