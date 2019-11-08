from matplotlib import pyplot as plt
from numpy import zeros, ones, arange, tan
from math import pi
   

# Chapter 3

def plot_lagrange_poly(lagrange_poly):
    '''
    Plot Lagrange polynomials as defined in the notebook
    '''

    # Generate and plot points
    n = 3
    x_data = range(n)
    x_data_fine = arange(0, n, 0.1)
    plt.ylim((-0.25, 1.25))
    plt.plot(x_data, ones(n), 'o', x_data, zeros(n), 'o')
    legend = ['Ones', 'Zeros']
    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params) # to use latex math expressions in labels

    # Plot Lagrange polynomials
    for i in range(n):
        plt.plot(x_data_fine, lagrange_poly(x_data, i, x_data_fine), '-')
        legend.append("$l_"+str(i)+"$")
    plt.legend(legend)
    plt.title("The first {} Lagrange polynomials".format(n))
    plt.show()


def plot_interpolator(poly_func, x_data, y_data, title="Interpolation", ylim=None):
    '''
    Plots (x_data, y_data) and their interpolator poly_func
    '''
    n = len(x_data)

    # Points where to plot the polynomial
    step = 0.1
    xfine = arange(0, n-1+step, step)
    yfine = zeros(xfine.size)
    for i in range(xfine.size):
        yfine[i] = poly_func(x_data, y_data, xfine[i])

    # Plot
    from matplotlib import pyplot as plt
    plt.plot(xfine, yfine, '-', x_data, y_data, 'x')
    if ylim:
        plt.ylim((-ylim, ylim))
    plt.title(title)
    plt.show()


def plot_spline(x, y):
    '''
    plots the cubic spline interpolating x and y
    '''
    from scipy.interpolate import CubicSpline # that's a class      
    step = 0.1
    n = len(x)
    xfine = arange(0, n-1+step, step)
    
    # Interpolate them with a cubic spline
    spline = CubicSpline(x, y, 0, 'natural')
    plt.plot(xfine, spline(xfine), "-", x, y, "x")
    
    # Add lines and labels
    # Matplotlib tricks, skip if you're not interested
    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params) # to use latex math expressions in labels
    for i in range(n):
        xl = x[i]+0.5
        yl = spline(xl) + 0.5
        if yl >= 1.5:
            yl = spline(xl) - 0.5
        plt.annotate('$x_'+str(i)+'$',xy=(x[i]-0.1, -1), fontsize='x-large')
        plt.plot([x[i], x[i]], [-0.9, spline(x[i])], '--', color='gray')
        plt.annotate('$y_'+str(i)+'$',xy=(-1, y[i]), fontsize='x-large')
        plt.plot([-0.6, x[i]], [y[i], y[i]], '--', color='gray')
        if i < n-1:
            plt.annotate('$f_'+str(i)+'$(x)',xy=(xl, yl), fontsize='x-large')
            plt.plot([xl, xl-0.2], [yl, spline(xl-0.2)], '-', color='black')
    plt.xlim((-1, n))
    plt.ylim((-1, 1.5))
    plt.axis('off')
    plt.title('A natural cubic spline with {} knots (n={})'.format(n, n-1))
    plt.show()

# Chapter 4

def enlarge_plot_size():
    # Enlarge the plot size, everywhere in the notebook
    def set_size(preset='screen'):
        import matplotlib
        if preset == 'screen':
            matplotlib.rcParams['figure.figsize'] = (7, 4)
            matplotlib.rcParams['axes.titlesize'] = 10
            matplotlib.rcParams['axes.labelsize'] = 10
        if preset == 'presentation':
            matplotlib.rcParams['figure.figsize'] = (14, 8)
            matplotlib.rcParams['axes.titlesize'] = 30
            matplotlib.rcParams['axes.labelsize'] = 30
    
    set_size('presentation')

def intro_plot():
        xmin = 0
        xmax = 6
        x = arange(xmin, xmax, 0.01)
        plt.plot(x, tan(x), '-')
        plt.plot(x, zeros(x.size), '-', color='black')
        miny = -5
        maxy = 5
        plt.ylim(miny, maxy)
        plt.xlim(xmin, xmax)
    
        # Annotation
        label_size = 'x-large'
        plt.annotate("f(x)", xy=(1.1, tan(1)), fontsize=label_size, color='royalblue')
    
        # Bracket
        minx = 2
        maxx = 4
        bracket_color = 'green'
        plt.plot([minx, minx], [miny, maxy], '--', color=bracket_color)
        plt.plot([maxx, maxx], [miny, maxy], '--', color=bracket_color)
        ybracket = 2/3*(maxy)
        plt.annotate("Bracket [{}, {}]".format(minx, maxx), xy=((minx+maxx)/2-0.4, ybracket), color=bracket_color, fontsize=label_size)
        plt.annotate(s='', xy=(minx, ybracket-0.5), xytext=(maxx,ybracket-0.5), arrowprops=dict(arrowstyle='<->', color=bracket_color))
        
        # Root
        from math import pi
        #plt.plot([0, 6], [0, 0], '--', color='black')
        plt.plot([pi], [0], 'o', color='red')
        plt.annotate("Root", xy=(3, -0.5), color='red', fontsize=label_size)
        return plt

def ch4_intro_plot():
    enlarge_plot_size()
    plt = intro_plot()
    label_size = 'x-large'
    # Estimates
    step = 0.3
    maxx = 4
    x = arange(pi, maxx, step)
    plt.plot(x, tan(x), '.-', color='red')
    plt.annotate('Estimates', xy=(maxx-step, tan(pi+step)), color='red', fontsize=label_size)
    plt.show()

def incremental_search_plot():
    plt = intro_plot()
    label_size = 'x-large'
    plt.plot([])
    x1 = 2.5
    x2 = 3.5
    label_color = 'blue'
    plt.plot([x1, x2], [tan(x1), tan(x2)], 'o', color=label_color)
    shift = 0.4
    plt.annotate(s="$f(x_1)$", xy=(x1-shift/2, tan(x1)-shift), fontsize=label_size, color=label_color)
    plt.annotate(s="$f(x_2)$", xy=(x2-shift/2, tan(x2)+shift/2), fontsize=label_size, color=label_color)
    plt.show()

def plot_estimates(estimates):
    # Plot
    plt = intro_plot()
    label_size = 'x-large'
    plt.xlim(2, 4)
    plt.plot(estimates, zeros(len(estimates)), 'x', color='blue')
    plt.annotate("Estimates", xy=(2.5, 1), color='blue', fontsize=label_size)
    plt.show()

def bisection_init():
    graph = intro_plot()
    graph.title("Initialization")
    graph.show()

def bisection_iter1():
    graph = intro_plot()
    graph.title("Iteration 1")
    graph.plot([3], [0], 'x', color='blue')
    graph.plot([3, 3], [-5, 5], '--', color='green')
    rect = graph.Rectangle([2, -5], 1, 10, color='gray')
    graph.gca().add_patch(rect)
    graph.show()

def bisection_iter2():
    graph = intro_plot()
    graph.title("Iteration 1")
    graph.plot([3], [0], 'x', color='blue')
    graph.plot([3, 3], [-5, 5], '--', color='green')
    graph.plot([3.5], [0], 'x', color='blue')
    graph.plot([3.5, 3.5], [-5, 5], '--', color='green')
    rect = graph.Rectangle([2, -5], 1, 10, color='gray')
    graph.gca().add_patch(rect)
    rect = graph.Rectangle([3.5, -5], 0.5, 10, color='gray')
    graph.gca().add_patch(rect)
    graph.show()

def plot_false_position():
    plt = intro_plot()
    label_size = 'x-large'
    a = 2 ; b = 4
    plt.plot([a, b], [tan(a), tan(b)], '-')
    plt.plot([a, b], [tan(a), tan(b)], 'o', color='darkorange')
    plt.annotate("Linear interpolation", xy=(2.5, -2), fontsize=label_size, color='darkorange')
    x = (a*tan(b)-b*tan(a))/(tan(b)-tan(a))
    plt.plot([x], [0], 'x', color='blue')
    plt.annotate("Linear interpolation", xy=(2.5, -2), fontsize=label_size, color='darkorange')
    plt.show()

def difftan(x): # derivative
     from math import sin, cos
     return sin(x)**2/cos(x)**2 + 1

def newton_plot(f, diff, xmin, xmax, ymin, ymax, xi, root=True):


    x = arange(xmin, xmax, 0.01)
    vals = zeros(x.size)
    for i in range(x.size):
        vals[i] = f(x[i])
    plt.plot(x, vals, '-')
    plt.plot(x, zeros(x.size), '-', color='black')
    miny = ymin
    maxy = ymax
    plt.ylim(miny, maxy)
    plt.xlim(xmin, xmax)

    # Annotation
    label_size = 'x-large'
    plt.annotate("f(x)", xy=(0.5, 2), fontsize=label_size, color='royalblue')

    # Root
    if root:
        from math import pi
        #plt.plot([0, 6], [0, 0], '--', color='black')
        plt.plot([pi], [0], 'o', color='red')
        plt.annotate("Root", xy=(2.8, -1), color='red', fontsize=label_size)

    #xi
    plt.plot([xi], [0], 'o', color='blue')
    plt.plot([xi], [f(xi)], 'o', color='blue')
    plt.annotate("$x_i$", xy=(xi-.05, -.7), color='blue', fontsize=label_size)
    plt.plot([xi, xi], [0, f(xi)], '--', color='blue')

    # tangent line
    plt.plot([xi-1, xi, xi+1], [f(xi)-diff(xi), f(xi), f(xi)+diff(xi)], color='darkorange')
    plt.annotate("$y=f'(x_i)x+cte$", xy=(xi+0.3, tan(xi)+1), color='darkorange', fontsize=label_size)
    # xi+1
    xi=4.3
    plt.annotate("$f(x_i)$", xy=(xi+0.05, tan(xi)/2), color='blue', fontsize=label_size)
    x2 = xi-tan(xi)/diff(xi)
    plt.plot([x2], [0], 'o', color='blue')
    plt.annotate("$x_{i+1}$", xy=(x2-.2, -.7), color='blue', fontsize=label_size)
    plt.title("Newton-Raphson Method: i -> i+1")
    return plt

def plot_newton_raphson_1():   
    xmin=2
    xmax=5
    ymin=-5
    ymax=5
    xi=4.3
    return newton_plot(tan, difftan, xmin, xmax, ymin, ymax, xi, True)


def plot_newton_raphson_2():
    # xi+2
    label_size = 'x-large'
    graph = plot_newton_raphson_1()
    xi = 4.3
    x2 = xi-tan(xi)/difftan(xi)
    x3 = x2-tan(x2)/difftan(x2)

    # tangent line at xi+1
    graph.plot([x2], [0], 'o', color='blue')
    graph.annotate("$x_{i+1}$", xy=(x2-.2, -.7), color='blue', fontsize=label_size)
    plt.plot([x2, x2], [0, tan(x2)], '--', color='blue')
    graph.plot([x2], [tan(x2)], 'o', color='blue')
    graph.plot([x3], [0], 'o', color='blue')
    graph.annotate("$x_{i+2}$", xy=(x3-.2, -.7), color='blue', fontsize=label_size)
    graph.plot([x2-1, x2, x2+1], [tan(x2)-difftan(x2), tan(x2), tan(x2)+difftan(x2)], color='green')
    graph.annotate("$y=f'(x_{i+1})x+cte$", xy=(x2-1, tan(x2)+0.5), color='green', fontsize=label_size)
    graph.title("Newton-Raphson Method: i+1 -> i+2")
    graph.show()

def plot_function(f):
    from numpy import arange
    x = arange(0, 15, 0.1)
    
    from matplotlib import pyplot as plt
    y = zeros(x.size)
    for i in range(x.size):
        y[i] = f(x[i])
    plt.plot(x, y)
    plt.plot(x, zeros(x.size), color='black')
    plt.title("f(x)")
    plt.show()

# Chapter r6

def newton_cotes_plot():
    from numpy import arange, array
    from matplotlib import pyplot as plt
    from scipy.interpolate import lagrange
    
    x = arange(5)
    y = array([1, 2.5, 3.2, 2.1, 6])
    plt.plot(x, y, 'o')
    
    poly = lagrange(x, y)
    xfine = arange(0, 5, 0.1)
    yfine = poly(xfine)
    plt.plot(xfine, yfine)
    plt.annotate("$P_n(x)$", xy=(2.5, 3.5))
    plt.annotate(s='', xy=(2.5, 3.4), xytext=(2.5,2.8), arrowprops=dict(arrowstyle='<-'))
    for i in range(5):
        plt.plot([x[i], x[i]], [0, y[i]], '--', color='gray')
    plt.annotate(s='', xy=(x[1], 1), xytext=(x[2],1), arrowprops=dict(arrowstyle='<->'))
    plt.annotate("h", xy=(1.5, 1.1))
    
    xtics = array(["$x_0=a$", "$x_1$", "$x_{...}$", "$x_{n-1}$", "$x_n=b$"])
    ytics = array(["$f(x_0)$", "$f(x_1)$", "$f(x_{,,,})$", "$f(x_{n-1})$", "$f(x_n)$"])
    plt.xticks(x, xtics)
    plt.yticks(y, ytics)
    plt.xlim(-0.2,4.2)
    plt.ylim(0,6.2)
    plt.show()
    #plt.xtics(values, xticks)
    
    plt.show()

def trapezoidal_plot():
    from numpy import arange, array
    from matplotlib import pyplot as plt
    from scipy.interpolate import lagrange

    x = arange(5)
    y = array([1, 2.5, 3.2, 2.1, 6])
    #plt.plot(x, y, 'o')

    line = lagrange([0, 4], [1, 6])
    xfine = arange(0, 5, 0.1)
    yfine = line(xfine)
    plt.plot(xfine, yfine)

    poly = lagrange(x, y)
    f = poly(xfine)
    plt.plot(xfine, f)
    plt.annotate("$f(x)$", xy=(2.5, 3.5))
    plt.annotate(s='', xy=(2.5, 3.4), xytext=(2.5,2.8), arrowprops=dict(arrowstyle='<-'))
    plt.annotate(s='', xy=(x[0], 1), xytext=(x[4],1), arrowprops=dict(arrowstyle='<->'))
    plt.annotate("h", xy=(2.5, 1.1))
    plt.plot([0, 0], [0, y[0]], '--', color='gray')
    plt.plot([4, 4], [0, y[4]], '--', color='gray')
    area = plt.Polygon([(0,0), (0, y[0]), (4, y[4]), (4,0), (0,0)], closed=True, color='#EEEEEE')
    plt.gca().add_patch(area)
    xtics = array(["$x_0=a$", "$x_1=b$"])
    plt.xticks([0, 4], xtics)
    plt.xlim(-0.2,4.2)
    plt.ylim(0,6.2)
    plt.show()
    #plt.xtics(values, xticks)

    plt.show()

def composite_trapezoidal_plot():
    from numpy import arange, array
    from matplotlib import pyplot as plt
    from scipy.interpolate import lagrange

    x = arange(5)
    y = array([1, 2.5, 3.2, 2.1, 6])
    #plt.plot(x, y, 'o')

    xfine = arange(0, 5, 0.1)


    poly = lagrange(x, y)
    f = poly(xfine)
    plt.plot(xfine, f)
    plt.annotate("$f(x)$", xy=(2.5, 3.5))

    for i in range(4):
        xfine = arange(x[i], x[i+1], 0.01)
        line = lagrange([x[i], x[i+1]], [y[i], y[i+1]])
        yfine = line(xfine)
        plt.plot(xfine, yfine)
        area = plt.Polygon([(x[i],0), (x[i], y[i]), (x[i+1], y[i+1]), (x[i+1],0), (x[i],0)], closed=True, color='#EEEEEE')
        plt.gca().add_patch(area)
        plt.plot([x[i], x[i]],[0, y[i]], '--', color='gray')

    plt.annotate(s='', xy=(2.5, 3.4), xytext=(2.5,2.8), arrowprops=dict(arrowstyle='<-'))
    plt.annotate(s='', xy=(x[2], 1), xytext=(x[3],1), arrowprops=dict(arrowstyle='<->'))
    plt.annotate("h", xy=(2.5, 1.1))

    plt.plot([0, 0], [0, y[0]], '--', color='gray')
    plt.plot([4, 4], [0, y[4]], '--', color='gray')
    xtics = array(["$x_0=a$", "$x_1=b$"])
    plt.xticks([0, 4], xtics)
    plt.xlim(-0.2,4.2)
    plt.ylim(0,6.2)
    plt.show()
    #plt.xtics(values, xticks)

    plt.show()

# Chapter 7

def plot_euler():
    from matplotlib import pyplot as plt
    from numpy import arange, exp
    x = arange(0, 3.1, 0.1)
    plt.plot(x, exp(x))
    plt.annotate(s='$k_0$', xy=(0, 1), xytext=(1, 2), arrowprops=dict(arrowstyle='<-', color='red'))
    plt.plot([0, 3], [1, 4], '--', color='orange')
    plt.plot([3, 3], [0, 4], '--', color='orange')
    plt.annotate("y(x+h)", xy=(3, 4.5), color='orange')
    plt.plot([3], [4], 'o')
    plt.xticks([0, 3], ["x", "x+h"])
    plt.yticks([])
    plt.ylim(0, exp(3.1))
    plt.show()

def plot_rk2():
    from matplotlib import pyplot as plt
    from numpy import arange, exp
    x = arange(0, 3.1, 0.1)
    plt.plot(x, exp(x))

    x0 = 0
    h = 3
    y = exp(0)
    k = y
    plt.annotate(s='$k_0$', xy=(x0, y), xytext=(x0+1, y + k), arrowprops=dict(arrowstyle='<-', color='red'))
    plt.plot([x0, x0+h/2], [y, y + h/2*k], '--', color='orange')
    plt.plot([h/2, h/2], [0, y + h/2*k], '--', color='orange')
    plt.plot([h/2], [y + h/2*k], 'o', color='orange')
    plt.annotate("y(x+h/2)", xy=(h/2, y + h/2*k+0.5), color='orange')

    x0 = x0+h/2
    y = y + h/2*k
    k = y
    plt.annotate(s='$k_1$', xy=(x0, y), xytext=(x0+1, y + k), arrowprops=dict(arrowstyle='<-', color='green'))
    plt.plot([0, h], [exp(0), exp(0) + h*k], '--', color='orange')
    plt.plot([h, h], [0, exp(0) + h*k], '--', color='orange')
    plt.plot([h], [exp(0) + h*k], 'o', color='orange')
    plt.annotate("y(x+h)", xy=(h, exp(0) + h*k + 0.5), color='orange')
    plt.annotate(s='', xy=(0, exp(0)), xytext=(1, exp(0) + k), arrowprops=dict(arrowstyle='<-', color='green'))

    plt.xticks([0, 1.5, 3], ["x", "x+h/2", "x+h"])
    plt.yticks([])
    plt.ylim(0, exp(3.1))
    plt.show()

def plot_rk4():
    from matplotlib import pyplot as plt
    from math import sqrt
    from numpy import arange, exp
    x = arange(0, 3.1, 0.1)
    plt.plot(x, exp(x))
    
    sumk = 0
    # k0
    x0 = 0
    h = 3
    y = exp(0)
    k = y
    sumk += k
    step = 1 # so that the vector has a unit norm
    plt.annotate(s='$k_0$', xy=(x0, y), xytext=(x0+step, y + step*k), arrowprops=dict(arrowstyle='<-', color='red'))
    plt.plot([x0, x0+h/2], [y, y + h/2*k], '--', color='orange')
    plt.plot([h/2, h/2], [0, y + h/2*k], '--', color='orange')
    plt.plot([h/2], [y + h/2*k], 'o', color='orange')
    plt.annotate("y(x+h/2)", xy=(h/2, y + h/2*k+0.5), color='orange')
    
    # k1
    x0 = x0+h/2
    y = y + h/2*k
    k = y
    sumk += 2*k
    step =  1
    plt.annotate(s='$k_1$', xy=(x0, y), xytext=(x0+step, y + step*k), arrowprops=dict(arrowstyle='<-', color='green'))
    plt.plot([0, h/2], [exp(0), exp(0) + h/2*k], '--', color='orange')
    plt.plot([h/2, h/2], [0, exp(0) + h/2*k], '--', color='orange')
    plt.plot([h/2], [exp(0) + h/2*k], 'o', color='orange')
    plt.annotate("y(x+h/2)", xy=(h/2, exp(0) + h/2*k + 0.5), color='orange')
    plt.annotate(s='', xy=(0, exp(0)), xytext=(step, exp(0) + step*k), arrowprops=dict(arrowstyle='<-', color='green'))
    
    # k2
    x0 = h/2
    y = exp(0) + h/2*k
    k = y
    sumk += 2*k
    step = 1
    plt.annotate(s='$k_2$', xy=(x0, y), xytext=(x0+step, y + step*k), arrowprops=dict(arrowstyle='<-', color='blue'))
    plt.plot([0, h], [exp(0), exp(0) + h*k], '--', color='orange')
    plt.plot([h, h], [0, exp(0) + h*k], '--', color='orange')
    plt.plot([h], [exp(0) + h*k], 'o', color='orange')
    plt.annotate("y(x+h)", xy=(h, exp(0) + h*k + 0.5), color='orange')
    plt.annotate(s='', xy=(0, exp(0)), xytext=(step, exp(0) + step*k), arrowprops=dict(arrowstyle='<-', color='blue'))
    
    # k3
    x0 = h
    y = exp(0) + h*k
    k = y
    sumk += k
    step = 1
    plt.annotate(s='$k_3$', xy=(x0, y), xytext=(x0+step, y + step*k), arrowprops=dict(arrowstyle='<-', color='black'))
    
    k=sumk/6
    plt.annotate(s='$k$', xy=(0, exp(0)), xytext=(1, exp(0) + k), arrowprops=dict(arrowstyle='<-', color='chocolate'))
    plt.plot([0, h], [exp(0), exp(0) + h*k], '--', color='orange')
    plt.plot([h, h], [0, exp(0) + h*k], '--', color='orange')
    plt.plot([h], [exp(0) + h*k], 'o', color='orange')
    plt.annotate("y(x+h)", xy=(h, exp(0) + h*k + 0.5), color='orange')
    
    plt.xticks([0, 1.5, 3], ["x", "x+h/2", "x+h"])
    plt.yticks([])
    plt.ylim(0, exp(3.1))
    plt.show()