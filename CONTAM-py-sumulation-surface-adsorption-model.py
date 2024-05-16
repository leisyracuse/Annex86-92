import numpy as np
from scipy.integrate import cumulative_trapezoid

def surface_adsorption_model(time, ca, ci):
    '''
    surface adsorption model developed based on literatures
    Reference:
    1. Pétigny, N., J. Zhang, E. Horner, S. Steady, M. Chenal, G. Mialon, and V. Goletto. “Indoor Air Depolluting Material: Combining Sorption Testing and Modeling to Predict Product’s Service Life in Real Conditions.” Building and Environment 202 (September 2021): 107838. https://doi.org/10.1016/j.buildenv.2021.107838.

    Input parameters: 
    time -> np.array, relative time from injection, seconds
    ca -> np.array, concentration of room air, ug/m^3
    ci -> np.array, concentration of outdoor/injection, ug/m^3

    Model parameters:
    a -> coefficient of Cs^2
    b -> coefficient of Cs

    Output:
    sink -> removal rate, ug/s 
    '''
    '''
    Model parameters
    '''
    a = -0.29 # from 93 ppb test
    b = 743.05 # from 93 ppb test
    # a = 5.82 # from 260 ppb test
    # b = 4382.04 # from 260 ppb test

    '''
    Zone and material variables, revise as needed
    '''
    Am = 0.225*0.2*2 # MOF material surface area, unit m^2 
    V_chamber = 0.05 # chamber volume, m^3
    ACH = 1 # chamber's air change rate, /hour
    Q = V_chamber * ACH /3600 # chamber outdoor flow rate, m^3/s
    km = 1.63/3600 # from 93 ppb test # convective mass transfer coefficient, m/s

    '''
    Calcualtion of Ms and sink
    '''

    # Get Ms(t) by subract VOC mass in exhaust from VOC mass in injection
    ms_as_func_t = (cumulative_trapezoid(Q*ci, time, initial  = 0) 
                    - cumulative_trapezoid(Q*ca, time, initial=0))/Am
    # Solve Cs by equation (9)
    print("ms is:" +repr(ms_as_func_t[-1]))
    cs1, _ = solve_quadratic(a, b, ms_as_func_t[-1])

    # check if Cs1 is positive    
    if cs1 > -1e-5:
        sink = km * Am * (ca[-1] - cs1)
    else:
        raise ValueError("The first root is not greater than 0.")
    
    # sink = km * Am * (ca[-1] - cs1)
    print("solved sink term is: " +repr(sink))

    return sink
    
def solve_quadratic(a, b, y):
    """
    Solves the equation ax^2 + bx = y for x.
    
    Parameters:
    a (float): Coefficient of x^2
    b (float): Coefficient of x
    y (float): Value of the equation

    Returns:
    tuple: Two roots (can be real or complex)
    """
    c = -y
    # Calculating the discriminant
    discriminant = b**2 - 4*a*c
    
    # Calculating two roots
    root1 = (-b + np.sqrt(discriminant)) / (2 * a)
    root2 = (-b - np.sqrt(discriminant)) / (2 * a)
    print("1st root: " +repr(root1))
    print("2nd root: " +repr(root2))
    
    return root1, root2


