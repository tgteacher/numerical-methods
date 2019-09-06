from numpy import array, mean, sum

def linear_regression(x_data, y_data):
    '''
    Returns a tuple (a, b) representing the straight line of equation
    y = a + bx that fits, in the least-squares sense, the points 
    represented by arrays x_data and y_data. 
    '''

    x_bar = mean(x_data)
    y_bar = mean(y_data)
    b = sum(y_data * (x_data-x_bar)) / sum(x_data*(x_data - x_bar))
    a = y_bar - b * x_bar
    return (a, b)


x = array([1, 3, 8, 9])
y = array([3, 5, 8, 4])

from matplotlib import pyplot as plt
(a, b) = linear_regression(x, y)
plt.plot(x, y, '+', x, b*x+a, '-')
plt.show()
