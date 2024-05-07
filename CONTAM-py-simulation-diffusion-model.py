import numpy as np
import math

class Diffusion_model():
    '''
    diffusion model developed by Zhenlei and Jialei. 
    Written in Class for defining and storing extra parameter.
    '''
    def __init__(self):
        '''
        define model parameters
        '''
        self.dm = 9.99e-7 # m^2/s
        self.kma = 4.05e5
        self.cm0 = 0 # intitial concentration in material, ug/m^3
        self.delta_t = 60 # time step, seconds
        self.delta_y = 2e-4 # mesh size, meter
        '''
        define chamber parameters, may read from CONTAM
        '''
        self.c_out = 93 # VOC injection/outdoor concentration, ug/m^3
        self.V_chamber = 0.05 # chamber volume, m^3
        self.ACH = 1 # chamber's air change rate, /hour
        self.ACS = self.ACH/3600 # chamber's air change rate, /hour
        self.Q = self.V_chamber * self.ACH /3600 # chamber outdoor flow rate, m^3/s
        self.km = 3.42/3600 # from 93 ppb test # convective mass transfer coefficient, m/s

        '''
        define material parameters
        '''
        self.Am = 0.225*0.2 # MOF material surface area, unit m^2
        self.depth_m = 4e-3 # MOF material thickness, unit m
        self.L = self.Am / self.V_chamber # Loading ratio, material exposure area / chamber volume, 1/m 

    def gen_mesh(self, depth, delta_y):
        '''
        funtion to discrete material for 1D diffusion problem in y-axis (bottom->top). 
        
        Input parameters:
        depth -> float, depth of material, meter
        delta_y -> float, mesh size of material, meter

        Output:
        y -> array, the location of mesh center, meter
        num_node -> integer, total number of meshes
        cm_array -> array, concentration distribution in material, ug/m^3
        '''
        num_node = depth / delta_y
        print('number of node is: ' + repr(num_node))

        y = np.zeros(int(num_node))
        if num_node.is_integer():
            for i in range(int(num_node)):
                y[i] = (i + 0.5) * delta_y
        else:
            print("Number of node is not a integer")

        node = y
        num_node = num_node
        cm_array = np.ones(len(node)) * self.cm0 # initialize with concentration at time 0
        return y, num_node, cm_array
 
    def create_coef_matrix(self, M, r, alpha):
        '''
        Building coefficient matrix for iteration
        '''
        M[0][0] = 1 + 2 *r
        M[0][1] = -2 * r
        M[M.shape[1]-1][M.shape[1]-1] = 1 + 2*r + alpha
        M[M.shape[1]-1][M.shape[1]-2] = -2*r

        for i in range(1, M.shape[1]-1):
            M[i][i] = 1 + 2*r
            M[i][i-1] = -r
            M[i][i+1] = -r
        return M
    
    def solve_matrix(self, delta_t, delta_y, ca_i, cm_array_i):
        '''
        function to solve Ca_i+1 and Cm_i+1 from Ca_i and Cm_i.
        Cm_i and Cm_i+1 are array along y-axis
        
        Input parameters:
        delta_t -> float, time step, seconds
        delta_y -> float, mesh size of material, meter
        ca_i -> float, air concentration of current time step from CONTAM, ug/m^3
        cm_array_i -> array, concentration distribution in material at time i, ug/m^3
        '''

        '''
        Variables to simply calculation
        '''
        X = self.ACS * delta_t + self.L * self.km * delta_t + 1
        r = self.dm * delta_t / delta_y**2
        alpha = 1 + (2 * self.km * delta_t * (1 + self.ACS * delta_t))/(self.kma * delta_y * X)
        beta = (2 * self.km * delta_t) / (delta_y * X)

        Y = cm_array_i
        # update the top layer concentration in material by ca_i and corresponding mass flux from air to material
        Y[len(cm_array_i)-1] = 2 * cm_array_i[len(cm_array_i)-1] + beta * ca_i + beta * self.ACS *delta_t * self.c_out
        M = np.zeros([len(cm_array_i),len(cm_array_i)]) # create empty coefficent matrix for calculation
        M = Diffusion_model().create_coef_matrix(M,r,alpha) # update matrix with data from time i

        cm_array_i_plus_1 = np.linalg.solve(M,Y) # solve cm_i+1
        # update c_i+1 with cm_i+1
        ca_i_plus_1 = (self.L * self.km * delta_t) / (self.kma * X) * cm_array_i_plus_1[len(cm_array_i_plus_1) - 1] + 1.0 / (X) * ca_i + (self.ACS * delta_t) / (X) * self.c_out

        return ca_i_plus_1, cm_array_i_plus_1


if __name__ == "__main__":
    model  = Diffusion_model()
    y_axis, num_node, cm_array = model.gen_mesh(model.depth_m, model.delta_y)
    ca_i = 0
    model.cm0 = 1
    cm_i = np.ones(len(cm_array))*model.cm0

    ca_i_plus_1, cm_i_plus_1 = model.solve_matrix(model.delta_t, model.delta_y, ca_i, cm_i)

    ca_i = ca_i_plus_1
    cm_i = cm_i_plus_1



    print(ca_i)
    print(cm_i)



