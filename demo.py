from numpy import * # Calling all Numpy's functions!


    

def compute_error_for_given_points(b, m, points):
    """ Computing errors for given points """
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[0, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learning_rate):
    """ step gradient """
    # gradient descent
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b,new_m]



def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    """ Gradient descent runner """
    b = starting_b
    m = starting_m

    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b,m]

def run():
    """ Run function! """
    points = genfromtxt('data.csv', delimiter=';')
    # HyperParameters
    learning_rate = 0.0001
    # y = mx + b (slope formula)
    # y = theta0 + theta1 * x
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_given_points(initial_b, initial_m, points)))

    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m,learning_rate, num_iterations)
    
    print("After ",num_iterations," iterations")
    print("m = ",m ," b = ",b ," error = ",compute_error_for_given_points(b, m, points))

if __name__ == 'main': run() # Calling the run function !
