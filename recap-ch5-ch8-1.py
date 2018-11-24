# 5.1.13

from math import tan, pi, sqrt, atan, exp
from numpy import array

a = 500 # m
alphas = array([54.80, 54.06, 53.34])*pi/180
betas = array([65.59, 64.59, 63.62])*pi/180

def f(alpha, beta): # expression of x
    return a*tan(beta)/(tan(beta)-tan(alpha))

def g(alpha, beta):
    return a*tan(alpha)*tan(beta)/(tan(beta)-tan(alpha))

xdot10 = (f(alphas[2], betas[2]) - f(alphas[0], betas[0]))/2
ydot10 = (g(alphas[2], betas[2]) - g(alphas[0], betas[0]))/2

v10 = sqrt(xdot10**2+ydot10**2)
gamma = atan(ydot10/xdot10)

print("v(10)={} m/s".format(v10))
print("gamma={} deg".format(gamma*180/pi))

# 6.2.12

def erf(x):
    A0 = 8/9
    A1 = 5/9
    x1 = sqrt(3/5)
    
    def f(u):
        return exp(-(x/2+x/2*u)**2)

    I = A0*f(0) + A1*f(x1) + A1*f(-x1)
    
    return I*(2/sqrt(pi)*x/2)

print("erf(1.0)={}".format(erf(1.0)))

# 7.1.13

# FROM CHAPTER 7
from numpy import array

def runge_kutta_4(F, x0, y0, x, h):
    '''
    Return y(x) given the following initial value problem:
    y' = F(x, y)
    y(x0) = y0 # initial conditions
    h is the increment of x used in integration
    F = [y'[0], y'[1], ..., y'[n-1]]
    y = [y[0], y[1], ..., y[n-1]]
    '''
    X = []
    Y = []
    X.append(x0)
    Y.append(y0)
    while x0 < x:
        k0 = F(x0, y0)
        k1 = F(x0+h/2.0, y0 + h/2.0*k0)
        k2 = F(x0 + h/2.0, y0 + h/2*k1)
        k3 = F(x0+h, y0+h*k2)
        y0 = y0 + h/6.0*(k0+2*k1+2.0*k2+k3)
        x0 += h
        X.append(x0)
        Y.append(y0)
    return array(X), array(Y)

def F(t, y):
    c = 0.03 # kg/(m.s)^(1/2)
    g = 9.8 # m/s^2
    m = 0.25 # kg
    return array([
                    y[1],
                    -c/m*y[1]*(y[1]**2+y[3]**2)**(1/4),
                    y[3],
                    -c/m*y[3]*(y[1]**2+y[3]**2)**(1/4)-g
                ])

v0 = 50 # m/s
gamma = 30*pi/180 # rad
y0 = array([
            0,
            v0/sqrt(1+tan(gamma)**2),
            0,
            v0*tan(gamma)/sqrt(1+tan(gamma)**2)
         ])
T, Y =runge_kutta_4(F, 0, y0, 5, 10E-4)

from matplotlib import pyplot as plt
plt.plot(Y[:,0], Y[:,2])
#plt.show()

for i in range(len(T)):
    if Y[i, 2] < 0:
        time_of_flight = T[i]
        range_x = Y[i, 0]
        break

print("Time of flight: {}s".format(time_of_flight))
print("Range: {}m".format(range_x))














