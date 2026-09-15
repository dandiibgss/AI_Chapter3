# Modifikasi Example 3.15 Linear Regression
import matplotlib.pyplot as plt
from scipy import stats

# Data diperbanyak (lebih banyak titik x dan y)
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
y = [3, 5, 5, 6, 7, 8, 8, 9, 11, 12, 13, 12, 15]

slope, intercept, r, p, std_err = stats.linregress(x, y)
print("slope: ", slope)
print("intercept: ", intercept)
print("r-squared: ", r**2)

def myfunc(x):
    return slope * x + intercept

mymodel = list(map(myfunc, x))

# Plot data asli
plt.scatter(x, y, color="blue", label="Data asli")

# Plot garis regresi
plt.plot(x, mymodel, color="red", label="Garis regresi linear")

# Tambahan label, judul, legenda, dan grid
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")
plt.legend()
plt.grid(True)

plt.show()