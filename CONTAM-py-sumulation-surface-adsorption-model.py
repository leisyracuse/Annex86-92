import numpy as np
from scipy.integrate import cumulative_trapezoid
from dataclasses import dataclass
import matplotlib.pyplot as plt 
from material_adsorption import SorptionMaterial

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

@dataclass
class ChamberPara():
    '''
    ACH -> air change rate, 1/h
    V_chamber -> chamber volume, m^3
    Q -> chamber air flow rate, m^3/s
    ci -> injection/outdoor concentration, ug/m^3
    '''
    ACH: float
    V_chamber: float
    Q: float
    ci: float
    ca0: float
    cs0: float

if __name__ == "__main__":
    MOF = SorptionMaterial(Am = 0.225*0.2*2, Km=2.09/3600, a=-1.1, b=1097.26)
    chamber = ChamberPara(ACH = 1, V_chamber=0.05, Q=1*0.05, ci = 93*0.0409*30,
                          ca0=0, cs0=0)
    
    dt = 60
    time = np.arange(0, 28*24*3600, dt)

    ca_array = np.zeros(len(time))
    # cs_array = np.zeros(len(time))
    ci_array = np.zeros(len(time))
    ca_array[0] = chamber.ca0
    # cs_array[0] = chamber.cs0
    ci_array[0] = chamber.ci
    
    for i in range (1,len(time)):
       print("i = " +repr(i))
       ca_i = ca_array[i-1]
       print("ca of step i: " +repr(ca_i))
    #    cs_i = cs_array[-1]

       sink_i = surface_adsorption_model(time[:i],ca_array[:i],ci_array[:i])
       ca_i1 = ca_i + dt*(chamber.ACH/3600*(chamber.ci - ca_i) - (sink_i)/chamber.V_chamber)
       
       ca_array[i] = ca_i1
       ci_array[i] = chamber.ci
    
    print(ca_array)

    test_time = np.array([0, 1,3,5,7,10,14,17,21,24,28])*24*3600
    test_conc = np.array([0, 25.736, 26.252, 29.794,30.738,34.429,35.628,34.533,40.389,42.340,47.159])
    test_conc = test_conc*0.0409*30
    plt.figure()
    plt.plot(test_time, test_conc, 'ro', label = 'test')
    plt.plot(time, ca_array, 'b-', label = 'model')
    plt.legend()
    plt.show()



