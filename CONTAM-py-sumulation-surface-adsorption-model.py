'''
Function for CONTAM simulation
'''

def surface_adsorption_model(time, ca, ci):
    '''
    surface adsorption model developed based on literatures
    Reference:
    1. Pétigny, N., J. Zhang, E. Horner, S. Steady, M. Chenal, G. Mialon, and V. Goletto. “Indoor Air Depolluting Material: Combining Sorption Testing and Modeling to Predict Product’s Service Life in Real Conditions.” Building and Environment 202 (September 2021): 107838. https://doi.org/10.1016/j.buildenv.2021.107838.

    Input parameters: 
    time -> np.array, relative time from injection, seconds
    ca -> np.array, concentration of room air, ug/m^3
    ci -> np.array, concentration of outdoor/injection, ug/m^3

    Output:
    sink -> removal rate, ug/s 
    '''
    from scipy.integrate import cumulative_trapezoid
    import numpy as np

    '''
    Zone and material variables, revise as needed
    '''
    Am = 0.225*0.2 # MOF material surface area, unit m^2 
    V_chamber = 0.05 # chamber volume, m^3
    ACH = 1 # chamber's air change rate, /hour
    Q = V_chamber * ACH /3600 # chamber outdoor flow rate, m^3/s

    '''
    Calcualtion of Ms and sink
    '''

    # Get Ms(t) by subract VOC mass in exhaust from VOC mass in injection
    ms_as_func_t = (cumulative_trapezoid(Q*ci, time, initial  = 0) 
                    - cumulative_trapezoid(Q*ca, time, initial=0))/Am
    # Solve dMs/dt, equation (8) in reference 1. 
    dms_dt = np.gradient(ms_as_func_t, time)
    # dms_dt = calculate_slope(time, ms_as_func_t)
    sink = dms_dt*Am
    return sink