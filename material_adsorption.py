"""
material_adsorption.py - Material adsorption models.

Surface adsorption models developed based on literature
Reference for each model:
1. Polynomial Model: 
    Pétigny, N., J. Zhang, E. Horner, S. Steady, M. Chenal, G. Mialon, and V. Goletto. “Indoor Air Depolluting Material: Combining Sorption Testing and Modeling to Predict Product’s Service Life in Real Conditions.” Building and Environment 202 (September 2021): 107838. https://doi.org/10.1016/j.buildenv.2021.107838.

"""
from dataclasses import dataclass
import math
import numpy as np

# =============================================== class SorptionMaterial ====

@dataclass
class SorptionMaterial:
    """
    Am: Material surface [m2]
    Km: Average (convective) room mass transfer coefficient [m/s]

    Polynomial model parameters:
    a: Sorption equilibrium constant a [-]
    b: Sorption equilibrium constant b [-]

    Henry's law model parameters:
    Kma: Henry's law constant []

    Dm: Diffusion coefficient [m2/s]
    """
    Am: float
    Km: float
    a: float = 0.0
    b: float = 0.0
    Kma: float = 0.0
    Dm: float = 0.0
    
# ====================================================== class Diffusion ====

class Diffusion():
    def __init__(self, sorption_material):
        self.mat = sorption_material
        self.Dm = self.mat.Dm
        self.y_array = np.array([]) # location of mesh center, meter
        self.Cm_array = np.array([]) # concentration distribution in material at time step i, ug/m^3


    def gen_mesh(self, depth, delta_y):
        '''
        funtion to discrete material for 1D diffusion problem in y-axis (bottom->top). 
        
        Input parameters:
        depth -> float, depth of material, meter
        delta_y -> float, mesh size of material, meter

        Output:
        y -> array, the location of mesh center, meter
        num_node -> integer, total number of meshes
        Cm_array -> array, concentration distribution in material, ug/m^3
        '''
        self.depth = depth
        self.delta_y = delta_y

        num_node = int(self.depth / self.delta_y) + 1
        # print('number of node is: ' + repr(num_node))

        y = np.linspace(0, self.depth, num_node)
        self.y_array = y
        self.init_Cm_array()

    def init_Cm_array(self, Cm0 = 0):
        '''
        Initialize concentration distribution in material.

        Args:
            Cm0: `float`
                Initial concentration in the material [ug/m3]
        '''
        self.Cm_array = np.ones(len(self.y_array)) * Cm0
    
    def solve_explicit_central(self, delta_t, S):
        '''
        Solve "∂Cm(y,t)/∂t = Dm * (∂²Cm(y,t)/∂y²)" numerically with the explicit finite difference scheme (central difference), 
        with boundary conditions:
            - "Dm * ∂Cm(y = 0, t) / ∂y = 0" (Neumann boundary condition at the bottom)
            - "Dm * ∂Cm(y = depth, t) / ∂y = S" (adsorption rate at the surface)
        
        Args:
            delta_t: `float`
                Time step for solving the diffusion equation, seconds
            S: `float`
                Adsorption rate at the surface, [ug/m2/s]
        
        Returns:
            Cm_array: Updated concentration distribution in the material at the next time step [ug/m^3]
        '''
        # Number of nodes
        num_node = len(self.Cm_array)
        if num_node == 0:
            raise ValueError("Mesh not initialized. Please call `gen_mesh` before solving.")
        
        # Create a new array for the next time step
        Cm_next = np.copy(self.Cm_array)

        # Compute diffusion constant terms
        alpha = self.Dm * delta_t / (self.delta_y**2)

        # Check stability condition
        if alpha > 0.5:
            raise ValueError("Stability condition violated: alpha > 0.5\nPlease adjust parameters to ensure stability,\nOR try to use Crank-Nicolson scheme for discretization.")

        # Interior nodes
        for i in range(1, num_node - 1):
            Cm_next[i] = self.Cm_array[i] + alpha * (self.Cm_array[i+1] - 2 * self.Cm_array[i] + self.Cm_array[i-1])

        # Boundary condition at y = 0 (Neumann: zero flux)
        Cm_next[0] = Cm_next[1]

        # Boundary condition at y = depth (adsorption rate S)
        # Apply boundary condition at y = depth
        flux_contribution = (S / self.mat.Am) * self.delta_y / self.Dm  # Contribution to concentration [ug/m^3]; S has unit [ug/s]
        Cm_next[-1] = Cm_next[-2] + flux_contribution

        # Update the concentration array
        self.Cm_array = Cm_next
        return self.Cm_array
        
    def solve_crank_nicolson(self, delta_t, S):
        '''
        Solve "∂Cm(y,t)/∂t = Dm * (∂²Cm(y,t)/∂y²)" numerically with the Crank-Nicolson scheme, 
        with boundary conditions:
            - "Dm * ∂Cm(y = 0, t) / ∂y = 0" (Neumann boundary condition at the bottom)
            - "Dm * ∂Cm(y = depth, t) / ∂y = S" (adsorption rate at the surface)
        
        Matrix form:
            A * Cm_next = B * Cm_current + b

        Args:
            delta_t: `float`
                Time step for solving the diffusion equation, seconds
            S: `float`
                Adsorption rate at the surface, [ug/m2/s]
        
        Returns:
            Cm_array: Updated concentration distribution in the material at the next time step [ug/m^3]
        '''
        # Number of nodes
        num_node = len(self.Cm_array)
        if num_node == 0:
            raise ValueError("Mesh not initialized. Please call `gen_mesh` before solving.")
        
        # Create a new array for the next time step
        Cm_next = np.copy(self.Cm_array)

        # Compute diffusion constant terms
        alpha = self.Dm * delta_t / (2 * self.delta_y**2) # alpha for Crank-Nicolson is half of the explicit scheme

        # Matrix A for the implicit part
        # Matrix A Format:
        # 0     1     2     3     4 ... N-2     N-1     N       --> index
        # _______________________________________________________
        # 1+2a  -a    0     0     0 ... 0       0       0       --> row 0
        # -a    1+2a  -a    0     0 ... 0       0       0       --> row 1
        # 0     -a    1+2a  -a    0 ... 0       0       0       --> row 2
        # 0     0     -a    1+2a  -a... 0       0       0       --> row 3
        # ...                                                   ...
        # 0     0     0     0     0 ... -a      1+2a    -a      --> row N-1
        # 0     0     0     0     0 ... 0       -a      1+2a    --> row N
        diag = np.ones(num_node) * (1 + 2 * alpha)
        off_diag = np.ones(num_node - 1) * (-alpha)
        A = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

        # Matrix B for the explicit part
        # Matrix B Format:
        # 0     1     2     3     4 ... N-2     N-1     N       --> index
        # _______________________________________________________
        # 1-2a  a     0     0     0 ... 0       0       0       --> row 0
        # a     1-2a  a     0     0 ... 0       0       0       --> row 1
        # 0     a     1-2a  a     0 ... 0       0       0       --> row 2
        # 0     0     a     1-2a  a ... 0       0       0       --> row 3
        # ...                                                   ...
        # 0     0     0     0     0 ... a       1-2a    a       --> row N-1
        # 0     0     0     0     0 ... 0       a       1-2a    --> row N
        diag_b = np.ones(num_node) * (1 - 2 * alpha)
        off_diag_b = np.ones(num_node - 1) * alpha
        B = np.diag(diag_b) + np.diag(off_diag_b, k=1) + np.diag(off_diag_b, k=-1)
    
        # Adjust A and B for Neumann boundary condition at y = 0
        A[0, 1] = -2 * alpha
        B[0, 1] = 2 * alpha

        # Adjust A and B for Neumann boundary condition at y = depth
        A[-1, -2] = -2 * alpha
        A[-1, -1] = 1 + 2 * alpha
        B[-1, -2] = 2 * alpha

        # Right-hand side vector
        b = B @ self.Cm_array
        b[-1] += (S / self.mat.Am) * self.delta_y / self.Dm  # Contribution to concentration [ug/m^3]; S has unit [ug/s]

        # Solve for the next time step
        Cm_next = np.linalg.solve(A, b)

        # Update the concentration array
        self.Cm_array = Cm_next
        return self.Cm_array

    def solve_implicit_central(self, delta_t, S):
        '''
        Solve "∂Cm(y,t)/∂t = Dm * (∂²Cm(y,t)/∂y²)" numerically with the implicit scheme, 
        with boundary conditions:
            - "Dm * ∂Cm(y = 0, t) / ∂y = 0" (Neumann boundary condition at the bottom)
            - "Dm * ∂Cm(y = depth, t) / ∂y = S" (adsorption rate at the surface)
        
        Matrix form:
            A * Cm_next = B * Cm_current + b

        Args:
            delta_t: `float`
                Time step for solving the diffusion equation, seconds
            S: `float`
                Adsorption rate at the surface, [ug/m2/s]
        
        Returns:
            Cm_array: Updated concentration distribution in the material at the next time step [ug/m^3]
        '''
        # Number of nodes
        num_node = len(self.Cm_array)
        if num_node == 0:
            raise ValueError("Mesh not initialized. Please call `gen_mesh` before solving.")
        
        # Create a new array for the next time step
        Cm_next = np.copy(self.Cm_array)

        # Compute diffusion constant terms
        alpha = self.Dm * delta_t / (self.delta_y**2) 
        # Matrix A for the implicit part
        # Matrix A Format:
        # 0     1     2     3     4 ... N-2     N-1     N       --> index
        # _______________________________________________________
        # 1+2a  -a    0     0     0 ... 0       0       0       --> row 0
        # -a    1+2a  -a    0     0 ... 0       0       0       --> row 1
        # 0     -a    1+2a  -a    0 ... 0       0       0       --> row 2
        # 0     0     -a    1+2a  -a... 0       0       0       --> row 3
        # ...                                                   ...
        # 0     0     0     0     0 ... -a      1+2a    -a      --> row N-1
        # 0     0     0     0     0 ... 0       -a      1+2a    --> row N
        diag = np.ones(num_node) * (1 + 2 * alpha)
        off_diag = np.ones(num_node - 1) * (-alpha)
        A = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

        # Matrix B 
        # Matrix B Format:
        # 0     1     2     3     4 ... N-2     N-1     N       --> index
        # _______________________________________________________
        # 1     0     0     0     0 ... 0       0       0       --> row 0
        # 0     1     0     0     0 ... 0       0       0       --> row 1
        # 0     0     1     0     0 ... 0       0       0       --> row 2
        # 0     0     0     1     0 ... 0       0       0       --> row 3
        # ...                                                   ...
        # 0     0     0     0     0 ... 0       1       0       --> row N-1
        # 0     0     0     0     0 ... 0       0       1       --> row N
        # diag_b = np.ones(num_node) * (1 - 2 * alpha)
        # off_diag_b = np.ones(num_node - 1) * alpha
        # B = np.diag(diag_b) + np.diag(off_diag_b, k=1) + np.diag(off_diag_b, k=-1)
        B = np.diag(np.ones(num_node))

        # Adjust A and B for Neumann boundary condition at y = 0
        A[0, 1] = -2 * alpha
        # B[0, 1] = 2 * alpha

        # Adjust A and B for Neumann boundary condition at y = depth
        # A[-1, -2] = -2 * alpha
        A[-1, -1] = 1 + 1 * alpha
        # B[-1, -2] = 2 * alpha

        # Right-hand side vector
        b = B @ self.Cm_array
        # b[-1] += (S / self.mat.Am) * self.delta_y / self.Dm  # Contribution to concentration [ug/m^3]; S has unit [ug/s]
        b[-1] += (S / self.mat.Am) * delta_t / self.delta_y
        # Solve for the next time step
        Cm_next = np.linalg.solve(A, b)

        # Update the concentration array
        self.Cm_array = Cm_next
        return self.Cm_array

# ====================================================== class Sorption =====

class Sorption():
    def __init__(self, sorption_material):
        self.mat = sorption_material
        self.Am = self.mat.Am
        self.Km = self.mat.Km
        self.Cm = 0.0   # Gas phase concentration on material surface [ug/m3]
        self.Ms = 0.0   # Adsorbed mass concentration per unit surface area [ug/m2]
        self.S = 0.0    # Adsorption rate

class Polynomial(Sorption):
    """
    Polynomial model for adsorption of pollutants on material surface.

    Sorption equilibrium characteristics:
        Ms(Cm) = Cm * (a * Cm + b)
    Mass balance in the sorption material:
        Am * (dMs / dt) = S = Am * km * (Ca - Cm)
    """
    def __init__(self, sorption_material):
        super().__init__(sorption_material)
        self.a = self.mat.a
        self.b = self.mat.b

    def get_S(self, Ca) -> float:
        '''
        Get the adsorption rate of the material, S [ug/s].

        Args:
            Ca: `float`
                Current value of pollutant concentration in room air [ug/m3]

        :returns: Adsorption rate [ug/s]
        :rtype: float
        '''
        S = self.Am * self.Km * (Ca - self.Cm)
        self.S = S
        return S
    
    
    def get_Ms(self, dt) -> float:
        '''
        Calculate adsorbed mass per unit area (Ms) of the material.

        Args:
            dt: `float`
                Time step [s]

        :returns: Adsorbed mass per unit area (Ms) [ug/m2]
        :rtype: float

        Governing equations:
            S = Am * (dMs / dt) => dMs = S / Am * dt
            Ms = Ms + dMs
        '''
        dMs = (self.S / self.Am) * dt   # mass adsorbed at time t (dt)
        Ms = self.Ms + dMs
        self.Ms = Ms         # set Ms of the material, ready for the next step
        return Ms
    

    def get_Cm(self) -> float:
        '''
        Gas phase concentration on material surface (Cm).

        Args:
            
        :returns: Gas phase concentration on surface, Cm [ug/m3]
        :rtype: float

        Sorption equilibrium characteristics:
            Ms(Cm) = a * Cm * Cm + b * Cm = Cm * (a * Cm + b)
        '''
        discriminant = self.b**2 - 4 * self.a * (-1 * self.Ms)
        if discriminant < 0:
            raise ValueError("Discriminant is negative, no real roots exist.")
        
        discriminant_sqrt = math.sqrt(discriminant)

        x1 = (-1 * self.b + discriminant_sqrt) / (2 * self.a)
        x2 = (-1 * self.b - discriminant_sqrt) / (2 * self.a)

        Cm = x1
        if x1 > x2:
            Cm = x2 # Cm is the smaller result
        self.Cm = Cm
        return Cm
    
class InterfaceModel(Sorption):
    """
    Material-Air Interface Model for adsorption of pollutants on material surface.

    """
    def __init__(self, sorption_material):
        super().__init__(sorption_material)
        self.Kma = self.mat.Kma

    def get_S(self, Ca) -> float:
        '''
        Get the adsorption rate of the material, S [ug/s].

        Args:
            Ca: `float`
                Current value of pollutant concentration in room air [ug/m3]

        :returns: Adsorption rate [ug/s]
        :rtype: float
        '''

        # Cm = Kma * Cas
        # Cm(b,t) is the VOC concentration at the material surface (at the thickness of b) [ug/m3]; Cas(t) is the VOC concentration in the near material surface air (ug/m3); Kma is the material/air partition coefficient
        S = self.Am * self.Km * (Ca - self.Cm / self.Kma) 
        self.S = S
        return S

    def get_Ms(self, dt) -> float:
        '''
        Calculate adsorbed mass per unit area (Ms) of the material.

        Args:
            dt: `float`
                Time step [s]

        :returns: Adsorbed mass per unit area (Ms) [ug/m2]
        :rtype: float

        Governing equations:
            dMs = S / Am * dt
            Ms = Ms + dMs
        '''
        dMs = (self.S / self.Am) * dt   # mass adsorbed at time t (dt)
        Ms = self.Ms + dMs
        self.Ms = Ms         # set Ms of the material, ready for the next step
        return Ms
    

    def get_Cm(self, Cm) -> float:
        '''
        Gas phase concentration on material surface (Cm).
        Calculated from the Diffusion model. Cm is the surface cell concentration of the material concentration array (Cm_array[-1]) [ug/m3].

        '''
        self.Cm = Cm
        return Cm