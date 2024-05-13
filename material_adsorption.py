"""
material_adsorption.py - Surface adsorption model.

Surface adsorption model developed based on literature
Reference:
1. Pétigny, N., J. Zhang, E. Horner, S. Steady, M. Chenal, G. Mialon, and V. Goletto. “Indoor Air Depolluting Material: Combining Sorption Testing and Modeling to Predict Product’s Service Life in Real Conditions.” Building and Environment 202 (September 2021): 107838. https://doi.org/10.1016/j.buildenv.2021.107838.

"""
from dataclasses import dataclass
import math


@dataclass
class SorptionMaterial:
    """
    Am: Material surface [m2]
    Km: Average room mass transfer coefficient [m/s]
    a: Sorption equilibrium constant a [-]
    b: Sorption equilibrium constant b [-]
    """
    Am: float
    Km: float
    a: float
    b: float

# ====================================================== class PI_Control =====


class Sorption:
    """
    Sorption equilibrium characteristics:
        Ms(Cs) = CS * (a * Cs + b)
    Mass balance in the sorption material:
        Am * (dMs / dt) = S = Am * km * (Cr - Cs)
    """
    def __init__(self, sorption_material):
        self.mat = sorption_material
        self.Am = sorption_material.Am
        self.Km = sorption_material.Km
        self.a = sorption_material.a
        self.b = sorption_material.b
        self.Cs = 0.0   # Gas phase concentration on material surface [ug/m3]
        self.Ms = 0.0   # Adsorbed mass concentration per unit surface area [ug/m2]
        self.S = 0.0    # Adsorption rate

    def get_S(self, Cr) -> float:
        '''
        Get the adsorption rate of the material, S [ug/s].

        Args:
            Cr: `float`
                Current value of pollutant concentration in room air [ug/m3]

        :returns: Adsorption rate [ug/s]
        :rtype: float
        '''
        S = self.Am * self.Km * (Cr - self.Cs)
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
        # dMs = (self.S / self.Am) * dt   # mass adsorbed at time t (dt)
        dMs = (self.S / self.Am) * dt   # mass adsorbed at time t (dt)
        Ms = self.Ms + dMs
        self.Ms = Ms         # set Ms of the material, ready for the next step
        return Ms
    

    def get_Cs(self) -> float:
        '''
        Gas phase concentration on material surface (Cs).

        Args:
            

        :returns: Gas phase concentration on surface, Cs [ug/m3]
        :rtype: float

        Sorption equilibrium characteristics:
            Ms(Cs) = a * Cs * Cs + b * Cs = Cs * (a * Cs + b)
        '''
        discriminant = math.sqrt(self.b**2 - 4 * self.a * (-1 * self.Ms))
        x1 = (-1 * self.b + discriminant) / (2 * self.a)
        x2 = (-1 * self.b - discriminant) / (2 * self.a)

        Cs = x1
        if x1 > x2:
            Cs = x2 # Cs is the smaller result
        self.Cs = Cs
        return Cs