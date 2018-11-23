# 5.1.13
from numpy import array
from math import pi, tan, sqrt, atan, sin

alpha = array([54.80, 54.06, 53.34])*pi/180
beta = array([65.59, 64.59, 63.62])*pi/180
a = 500

def x(alp , bet):
    return a*tan(bet)/(tan(bet)-tan(alp))

def y(alp, bet):
    return a*(tan(alp)*tan(bet))/(tan(bet)-tan(alp))

x_dot = (x(alpha[2], beta[2]) - x(alpha[0], beta[0]))/2
y_dot = (y(alpha[2], beta[2]) - y(alpha[0], beta[0]))/2

print("==== 5.1.13 === ")
print("xdot= {} km/h".format(x_dot*3.6))
print("ydot= {} km/h".format(y_dot*3.6))
print("Plane velocity: {} km/h".format(
        sqrt(x_dot**2 + y_dot**2)*3.6
    ))
print("Gamma = {} deg".format(atan(y_dot/x_dot)*180/pi))

# 6.2.12
print("==== 6.2.12 ====")
from math import exp
def erf(x):
    def f(u, x):
        return exp(-(x/2+x/2*u)**2)
    
    A0 = 8/9
    A1 = 5/9
    x0 = 0
    x1 = sqrt(3/5)
    
    I = 2/sqrt(pi)*x/2*(
        A0*f(x0, x) + A1*f(x1, x) + A1*f(-x1, x)
    )
    return I
    
print("erf(1.0)={}".format(erf(1.0)))

# 6.2.4
print("==== 6.2.4 ====")

A0 = (18+sqrt(30))/36
A1 = (18-sqrt(30))/36
x0 = sqrt(3/7-2/7*sqrt(6/5))
x1 = sqrt(3/7+2/7*sqrt(6/5))
I = pi/2*(A0*( sin(pi/2+pi/2*x0) + sin(pi/2-pi/2*x0) ) +
     A1*( sin(pi/2+pi/2*x1) + sin(pi/2-pi/2*x1)))
print("I = {}".format(I))

# 7.1.4
print("==== 7.1.4 ====")
from numpy import array
def euler(F, x0, y0, x, h):
    '''
    Return y(x) given the following initial value problem:
    y' = F(x, y)
    y(x0) = y0 # initial conditions
    h is the increment of x used in integration
    F = [y'[0], y'[1], ..., y'[n-1]]
    y = [y[0], y[1], ..., y[n-1]]
    '''
    X = [] # will store the value of x0 at each iteration
    Y = [] # will store the value of y0 at each iteration
    while x0 < x:
        h = min(h, x-x0)
        y0 = y0 + h*F(x0, y0)
        x0 += h
        X.append(x0)
        Y.append(y0)
    return array(X), array(Y)
def F(x, y):
    return y**(1/3)
x, y = euler(F, 0, 10**(-16), 1, 0.01)
x, y1 = euler(F, 0, 0, 1, 0.01)
from matplotlib import pyplot as plt
plt.plot(x, y, '+', x, y1, 'o', x, (2*x/3)**(3/2))
# plt.show()


# 7.1.13
print("==== 7.1.13 ====")
c = 0.03 # kg/(m.s)**(1/2)
m = 0.25 # kg
g = 9.8 # m/s**2
def F(x, y):
    return array([
        y[1],
        -c/m*y[1]*(y[1]**2+y[3]**2)**(1/4),
        y[3],
        -c/m*y[3]*(y[1]**2+y[3]**2)**(1/4)-g
    ])
x0 = 0
y0 = 0
xdot0 = sqrt(2500/(1+tan(30*pi/180)**2))
ydot0 = xdot0*tan(30*pi/180)**2
t, y = euler(F, 0, array([x0, xdot0, y0, ydot0]), 3, 0.1)
plt.clf()
plt.plot(y[:,0], y[:,2])
#plt.show()

for i in range(len(t)):
    if y[i, 2] < 0:
        print("Range = {} m".format(y[i, 0]))
        print("Time of flight = {} s".format(t[i]))
        break 
        

print("==== 8.1.19 ====")


## FROM CHAPTER 4

from numpy import zeros

def jacobian(f, x):
    '''
    Returns the Jacobian matrix of f taken in x J(x)
    '''
    n = len(x)
    jac = zeros((n, n))
    h = 10E-4
    fx = f(x)
    # go through the columns of J
    for j in range(n):
        # compute x + h ej
        old_xj = x[j]
        x[j] += h
        # update the Jacobian matrix (eq 3)
        # Now x is x + h*ej
        jac[:, j] = (f(x)-fx) / h 
        # restore x[j]
        x[j] = old_xj
    return jac

from numpy.linalg import solve
from numpy import sqrt

def newton_raphson_system(f, init_x, epsilon=10E-4, max_iterations=100):
    '''
    Return a solution of f(x)=0 by Newton-Raphson method.
    init_x is the initial guess of the solution
    '''
    x = init_x
    for i in range(max_iterations):
        J = jacobian(f, x)
        delta_x = solve(J, -f(x)) # we could also use our functions from Chapter 2!
        x = x + delta_x
        if sqrt(sum(delta_x**2)) <= epsilon:
            print("Converged in {} iterations".format(i))
            return x
    raise Exception("Could not find root!")

# FROM CHAPTER 7

from numpy import array
def runge_kutta_2(F, x0, y0, x, h):
    X = []
    Y = []
    X.append(x0)
    Y.append(y0)
    while x0 < x:
        k0 = F(x0, y0)
        k1 = F(x0+h/2, y0 + h/2*k0)
        y0 = y0 + h*k1
        x0 += h
        X.append(x0)
        Y.append(y0)
    return array(X), array(Y)

c = 3.2 * 10**(-4) # kg/m
g = 9.8 # m/s**2
m = 20 # kg

def F(x, y): # problem definition
    return array([
        y[1],
        -c/m*sqrt(y[1]**2+y[3]**2)*y[1],
        y[3],
        -c/m*sqrt(y[1]**2+y[3]**2)*y[3]-g
    ])

def r(u): # residuals, u = [xdot0, ydot0]
    # solve IVP
    y0 = array([0, u[0], 0, u[1]])
    t, y = runge_kutta_2(F, 0, y0, 9.9, 0.1)
    ru =  array([
                  abs(y[-1,0]-8000),
                  abs(y[-1,2])
                  ])   # residual has 2 dims...
    return ru
    
u = newton_raphson_system(r,
                      array([565.0, 565.0]))
#t, y = runge_kutta_2(F, 0, u, 10, 0.1)
print("x0, y0 = {}".format(u))
x0 = u[0]
y0 = u[1]
print("theta={} deg, v={} m/s".format(atan(y0/x0)*180/pi, sqrt(x0**2+y0**2)))
