# CLASS:
class Person:
    pass
    # class attributes:
    address = "no information"
    # constructure (kurucu metot):
    '''
         The self parameter in the method refers to the instance of the object itself, 
         allowing you to define and modify the object's attributes.
    '''
    def __init__(self, _name, _year):
        # object attributes:
        self.name = _name
        self.year = _year

    # Instance methods:
    '''
        ** "self" is necessary in all instance methods, 
        even if the method doesn't explicitly take any additional arguments.
        ** In Python, when you define a method inside a class, theh first
        parameter is always "self".
    '''
    def intro(self):
        print("Hello There. I am " + self.name)

    def calculate(self):
        return 2024 - self.year

# OBJECT:
p1 = Person(_name="Ali", _year=1999)
p2 = Person("Veli", 1998)


# accessing objecet attributes:
print("p1 - name: {} year: {} address: {}".format(p1.name, p1.year, p1.address))
print("p2 - name: {} year: {}".format(p2.name, p2.year))

# Updating attributes:
p1.name = "Ayse"
p1.address = "Sakarya"

# Using methods:
p1.intro()
p1.calculate()

print(f"Hello. My name is {p1.name} and I am {p1.calculate()} years old.")

# ----------------------

class Circle:
    pi = 3.14
    def __init__(self, _radius = 1):
        self.radius = _radius

    def area(self):
        return self.pi * self.radius ** 2

    def circumference(self):
        return 2 * self.pi * self.radius

c = Circle(5)
print(f"The Area is..: {c.area()} - The Circumference is..: {c.circumference()}")

# ----------------------------------------------------------------------------------

# INHERITANCE:
# ------------

class Person:
    def __init__(self, fname, lname):
        self.firstName = fname
        self.lastName = lname
        print("Person is Created")

    def who_am_i(self):
        print("I'm a person")

    def eat(self):
        print("I'm eating")

class Student(Person):
    def __init__(self, fname, lname, number):
        Person.__init__(self, fname, lname) # super().__init__() - this use is more common
        self.studentNumber = number
        print("Student is Created")

    # Method Overriding:
    def who_am_i(self):
        print("I'm a student")

    def sayHello(self):
        print("Hello. I'm a student")

class Teacher(Person):
    def __init__(self, fname, lname, branch):
        super().__init__(fname, lname)
        self.branch = branch
        print("Teacher is Created")

    def who_am_i(self):
        print("I'm a teacher")


p1 = Person("Ali", "Al")
s1 = Student("Veli", "Vel", 561)

print(p1.firstName + " " + p1.lastName)
print(s1.firstName + " " + s1.lastName + " " + str(s1.studentNumber))

p1.who_am_i()
s1.who_am_i()

p1.eat()
s1.eat()

s1.sayHello()

t1 = Teacher("Ayse", "Ay", "science")
print(t1.firstName + " " + t1.lastName + " " + t1.branch)

# ----------------------------------------------------------------------------------

my_list = [1, 2, 3]
# Special methods for classes:
class Movie:
    def __init__(self, title, director, duration):
        self.title = title
        self.director = director
        self.duration = duration

        print("Movie instance is created")

    def __str__(self):
        return f"{self.title} by {self.director}"

    def __len__(self):
        return self.duration

    def __del__(self):
        print("Object is Deleted")

m = Movie("Avengers", "Russo Bros", 120)

print(type(my_list))
print(type(m))

print(str(my_list))
print(str(m))

print(len(my_list))
print(len(m))

del m

print(m)

# ----------------------------------------------------------------------------------

# QUIZ UYGULAMASI:
# ----------------

class Question:
    def __init__(self, text, choices, answer):
        self.text = text
        self.choices = choices
        self.answer = answer

    def check_answer(self, answer):
        return self.answer == answer

class Quiz:
    def __init__(self, questions):
        self.questions = questions
        self.score = 0
        self.question_index = 0

    def get_question(self):
        return self.questions[self.question_index]

q1 = Question("En iyi programlama dili hangisidir?", ["C#", "Python", "Javascript", "Java"], "python")
q2 = Question("En popüler programlama dili hangisidir?", ["Javascript", "Python", "C#", "Java"], "python")
q3 = Question("En çok kazandıran programlama dili hangisidir?", ["C#", "Javascript", "Java", "Python"], "python")
questions = [q1, q2, q3]

quiz = Quiz(questions)
question = quiz.questions[quiz.question_index]
print(question.text)








