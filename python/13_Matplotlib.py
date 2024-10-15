import matplotlib.pyplot as plt
import numpy as np

x = [1, 2, 3, 4]
y = [1, 4, 9, 16]

plt.plot(x, y, "o--r")
plt.axis([0, 6, 0, 20]) # ilk iki x'in sınırları son iki y'nin sınırları

plt.title("Grafik Basligi")
plt.xlabel("x label")
plt.ylabel("y label")

plt.show()

# ---------

x = np.linspace(0, 2, 100)

plt.plot(x, x, label="linear", color="red")
plt.plot(x, x**2, label="quadratic")
plt.plot(x, x**3, label="cubic")

plt.xlabel("x label")
plt.ylabel("y label")

plt.title("Simple Plot")

plt.legend() # description for lines

plt.show()

# ---------

# Bir düzleme birden fazla grafik yerleştirme:
x = np.linspace(0, 2, 100)
fig, axs = plt.subplots(3)

axs[0].plot(x, x, color="red")
axs[0].set_title("linear")

axs[1].plot(x, x**2, color="green")
axs[1].plot("quadratic")

axs[2].plot(x, x**3, color="blue")
axs[2].plot("cubic")

plt.tight_layout()

plt.show()

# ---------

x = np.linspace(0, 2, 100)
fig, axs = plt.subplots(2, 2)
fig.suptitle("Grafik Başlığı")

axs[0, 0].plot(x, x, color="red")
axs[0, 1].plot(x, x**2, color="blue")
axs[1, 0].plot(x, x**3, color="green")
axs[1, 1].plot(x, x**4, color="orange")

plt.show()

# ---------

'''
    A figure in matplotlib is like a blank canvas on which you 
    can plot graphs, add subplots, titles, labels, and other elements.
'''

# Figure oluşturma:
x = np.linspace(-10, 9, 20)
y = x ** 3
z = x ** 2

figure = plt.figure()

# grafiğin düzlemdeki konumunu belirleme:
# sol, alt, genişlik, yükseklik
axes_cube = figure.add_axes([0.1, 0.1, 0.8, 0.8])
axes_cube.plot(x, y, 'b')
axes_cube.set_xlabel("X Axis")
axes_cube.set_ylabel("Y Axis")
axes_cube.set_title("Cube")

axes_square = figure.add_axes([0.15, 0.6, 0.25, 0.25])
axes_square.plot(x, z, 'r')
axes_square.set_xlabel("X Axis")
axes_square.set_ylabel("Y Axis")
axes_square.set_title("Square")

plt.show()

# ---------

import matplotlib.pyplot as plt

yil = [2011, 2012, 2013, 2014, 2015]

oyuncu1 = [8, 10, 12, 7, 9]
oyuncu2 = [7, 12, 5, 15, 21]
oyuncu3 = [18, 20, 22, 25, 19]

# Stack Plot:

plt.plot([], [], color="g", label="oyuncu1")
plt.plot([], [], color="r", label="oyuncu2")
plt.plot([], [], color="b", label="oyuncu3")

plt.stackplot(yil, oyuncu1, oyuncu2, oyuncu3, colors=['g', 'r', 'b'])
plt.title("Yıllara Göre Atılan Gol Sayıları")
plt.xlabel("Yıl")
plt.ylabel("Gol Sayısı")
plt.legend()
plt.show()

# Pie Grafiği:
goal_types = "Penaltı", "Kaleye Atılan Şut", "Serbest Vuruş"

goals = [12, 35, 7]
colors = ['g', 'r', 'b']

plt.pie(goals, labels=goal_types, colors=colors, shadow=True, explode=(0.05, 0.05, 0.05))
plt.show()

# Bar  Grafiği:
plt.bar([0.25, 1.25, 2.25, 3.25, 4.25], [50, 40, 70, 80, 20], label="BMW")
plt.bar([0.75, 1.75, 2.75, 3.75, 4.75], [80, 20, 20, 50, 60], label="Audi")

plt.legend()
plt.xlabel("Gün")
plt.ylabel("Mesafe (km)")
plt.title("Araç Bilgileri")

plt.show()

# Histogram Grafiği:
yaslar = [22, 55, 62, 45, 21, 22, 34, 42, 32, 4, 2, 102, 95, 85, 55, 110, 120, 70, 65, 55, 111, 115]
yas_grupları = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

plt.hist(yaslar, yas_grupları, histtype="bar", rwidth=0.8)
plt.xlabel("Yaş Grupları")
plt.ylabel("Kişi Sayısı")
plt.title("Histogram Grafiği")

plt.show()

