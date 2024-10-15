'''
    ITERATORS:
        * An iterator in Python is an object that allows you to traverse through all
        the elements of a collection (like a list or tuple) one element at a time,
        without needing to know the underlying structure of the collection.
'''

liste = [1, 2, 3, 4, 5]

# list yapısındaki __iter__ metodu sayesinde bunu yapabiliyoruz.
# Burada iteratör oluşturma işini for döngüsü kendi yapıyor.
for i in liste:
    print(i)

print(dir(liste))

# Yukarıdaki for döngüsünün çalşıma mantığı_1:
iterator = iter(liste)

# iteratörü next metotu ile her çapırdığımızda listenin bir elemanı gelir
print(next(iterator))
print(next(iterator))
print(next(iterator))

# Yukarıdaki for döngüsünün çalşıma mantığı_2:
liste = [1, 2, 3, 4, 5]
iterator = iter(liste)

while True:
    try:
        element = next(iterator)
        print(element)
    except StopIteration:
        break

# --------

class MyNumbers:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def __iter__(self):
        return self

    def __next__(self):
        if self.start <= self.stop:
            x = self.start
            self.start += 1
            return x
        else:
            raise StopIteration

liste = MyNumbers(10, 20)

# class'ımızda iter metotu olduğundan iterable bir sınıftır.
# Bu nedenle next vs. kullanmadan for otomatik olarak listeyi dolaşır.
for x in liste:
    print(x)

# for kullanmadan:
liste = MyNumbers(10, 15)

myIter = iter(liste)

while True:
    try:
        element = next(myIter)
        print(element)
    except StopIteration as e:
        break

# ---------------------------------------------------------------

'''
    GENERETORS:
        In Python, a generator is a special type of iterator that allows you to 
        iterate over a sequence of values without having to store them all in memory at once. 
        Generators provide a way to produce values on-the-fly and are particularly useful 
        when working with large data sets or streams of data.
'''

def cube():
    for i in range(5):
        # Değer üretildiği an kullanılır ve işi biter. Bellekte depolamaz.
        # Yani bu değere ikinci defa ulaşamam.
        yield i ** 3

for i in cube():
    print(i)

# --------

generator = (i ** 3 for i in range(5))
print(generator)

for i in generator:
    print(i)