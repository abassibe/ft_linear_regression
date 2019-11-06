import numpy as np

def estimatePrice(theta_0, theta_1, x):
    return theta_0 + theta_1 * x


theta = np.loadtxt("theta.csv", dtype = np.longdouble, delimiter = ',')
try:
    x = np.longdouble(input("Enter a mileage: "))
except:
    print("Error")
    exit()
print(estimatePrice(theta[0], theta[1], x))
