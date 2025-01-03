"""
Copyright (c) 2025. All rights reserved.
Author: Zhenlei Liu - Ecowise Building Solution, INC.
Description: analytical solution of dry material in material diffusion and convection mass transfer.
Created: 2025-01-03
Last modified:
Reference: 
    1. Deng, Baoqing, and Chang Nyung Kim. “An Analytical Model for VOCs Emission from Dry Building Materials.” Atmospheric Environment 38, no. 8 (March 2004): 1173–80. https://doi.org/10.1016/j.atmosenv.2003.11.009.
    2. Liu, Zhenlei, Andreas Nicolai, Marc Abadie, Menghao Qin, John Grunewald, and Jianshun Zhang. “Development of a Procedure for Estimating the Parameters of Mechanistic VOC Emission Source Models from Chamber Testing Data.” Building Simulation 14, no. 2 (April 2020): 269–82. https://doi.org/10.1007/s12273-020-0616-3.
"""
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class MaterialProperties:
    dm: float    # diffusion coefficient
    kma: float   # partition coefficient
    cm0: float   # initial concentration in material
    lm: float    # thickness of material
    Am: float    # surface area of material
    
@dataclass
class SystemProperties:
    V: float     # volume of chamber
    N: float     # air change rate per second
    hm: float    # convective mass transfer coefficient

class MassTransferSimulation:
    def __init__(self, material: MaterialProperties, system: SystemProperties):
        self.material = material
        self.system = system
        self.Bim = (system.hm * material.lm) / material.dm
        self.Alpha = (system.N * material.lm**2) / material.dm
        self.Beta = material.Am * material.lm / system.V
        
    def opt_fun(self, qn):
        """Characteristic equation for eigenvalues."""
        return (qn * np.tan(qn) - 
                (self.Alpha - qn**2) / (self.material.kma * self.Beta + 
                                      (self.Alpha - qn**2) * self.material.kma / self.Bim))
    
    def solve_qn(self, n_roots: int) -> np.ndarray:
        """Solve for eigenvalues"""
        qn_list = []
        
        # First root (special case)
        res = optimize.brentq(self.opt_fun, 0, np.pi/2 - 1e-10)
        qn_list.append(res)
        
        # Subsequent roots
        for i in range(1, n_roots):
            x1 = np.pi/2 + (i-1)*np.pi + 1e-10
            x2 = np.pi/2 + i*np.pi - 1e-10
            try:
                res = optimize.brentq(self.opt_fun, x1, x2)
                qn_list.append(res)
            except ValueError:
                print(f"Warning: Could not find root {i+1}")
                break
                
        return np.array(qn_list)
    
    def solve_An(self, qn):
        """Calculate normalization constants."""
        kma, Beta, Alpha, Bim = (self.material.kma, self.Beta, 
                                self.Alpha, self.Bim)
        return ((kma*Beta + (Alpha - qn**2) * kma/Bim + 2) * qn**2 * np.cos(qn) +
                qn*np.sin(qn) * (kma*Beta+(Alpha-3*qn**2)*kma/Bim + Alpha - qn**2))
    
    def Ca_at_time(self, t, qn_list):
        """Calculate gas phase concentration at time t."""
        terms = []
        for qn in qn_list:
            An = self.solve_An(qn)
            terms.append(qn*np.sin(qn)/An * 
                        np.exp(-self.material.dm*self.material.lm**-2*qn**2*t))
        return 2 * self.material.cm0 * self.Beta * np.sum(terms)
    
    def Cm_at_location_and_time(self, t, y, qn_list):
        """Calculate material phase concentration at location y and time t."""
        terms = []
        for qn in qn_list:
            An = self.solve_An(qn)
            terms.append((self.Alpha - qn**2)/An * 
                        np.cos(y/self.material.lm*qn) * 
                        np.exp(-self.material.dm*self.material.lm**-2*qn**2*t))
        return 2 * self.material.cm0 * np.sum(terms)
    
    def simulate(self, hours: int, time_step: int = 3600) -> dict:
        """Run simulation for specified duration."""
        time_seconds = hours * 3600
        qn_list = self.solve_qn(201)
        print(f"Found {len(qn_list)} eigenvalues")
        
        times = np.arange(0, time_seconds, time_step)
        ca_values = [self.Ca_at_time(t, qn_list) for t in times]
        cm_values = [self.Cm_at_location_and_time(t, self.material.lm, qn_list) 
                    for t in times]
        
        return {
            'times': times/3600,  # Convert to hours
            'ca_values': ca_values,
            'cm_values': cm_values
        }
    

# Example usage
material = MaterialProperties(
    dm=7.65e-11,  # diffusion coefficient
    kma=3289,     # partition coefficient
    cm0=5.28e7,   # initial concentration, ug/m^3
    lm=0.0159,    # thickness
    Am=0.212 * 0.212  # surface area
)

system = SystemProperties(
    V=0.05,      # chamber volume
    N=1/3600,    # air change rate
    hm=1/3600    # mass transfer coefficient
)

# Run simulation
sim = MassTransferSimulation(material, system)
results = sim.simulate(hours=24)

# Print results
print("\nGas phase concentrations:")
for t, ca in zip(results['times'], results['ca_values']):
    print(f"t={t:.1f}h: {ca:.2e}")

print("\nMaterial surface concentrations:")
for t, cm in zip(results['times'], results['cm_values']):
    print(f"t={t:.1f}h: {cm:.2e}")