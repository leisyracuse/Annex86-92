"""
MOF-SorptionControl.py - simulate air contaminant adsorbed by MOF material.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from argparse import ArgumentParser
from dataclasses import dataclass
from contamxpy import cxLib
from material_adsorption import SorptionMaterial, Diffusion, Sorption, Polynomial, InterfaceModel


@dataclass
class ControlIdxName:
    """
    Contains Index and Name of a control within either `cxLib.outputControls`
    list or `cxLib.inputControls` list.
    """
    index: int
    name: str

# =========================================================================
# USER INPUT - Set control-related GLOBAL variables.
# =========================================================================
#   INDEXes and Names can be obtained by running this script and viewing
# OUTPUT CONTROLS and INPUT CONTROLS lists. These indices are not
# necessarily the same as the control IDs in ContamW. They are indices into
# the corresponding named SET and SPLIT controls in the project file.

T_ROOM = ControlIdxName(1, "T_room")  # PRJ Control to get room temperature, degC (not used)
HCHO_ROOM = ControlIdxName(2, "HCHO_room")  # PRJ Control to get room HCHO conc, ug/m3
SINK_SET = ControlIdxName(1, "sink_set")  # PRJ Control to set HCHO sink rate, ug/s

PV_PASS_CTRL = HCHO_ROOM    # process value (pv) passed to controller
CO_SET_CTRL = SINK_SET      # control output (co) set by the controller

'''
0: single zone
1: chamber L (50L, 1ACH, 93ppb HCHO)
2: chamber H (50L, 1ACH, 260ppb HCHO)
'''
sim_case = 1

match sim_case:
    case 0:
        material = SorptionMaterial(Am = 0.045, Km = 9.5e-4, a = -44.03, b = 3751.57, Kma = 4.05e5, Dm = 9.99e-7) # define sorption material
    case 1:
        material = SorptionMaterial(Am = 0.09, Km = 4.528e-4, a = -0.29, b = 743.05, Kma = 1e5, Dm = 1e-7) # define sorption material
    case 2: 
        material = SorptionMaterial(Am = 0.09, Km = 1.136e-3, a = -0.01, b = 1272.99, Kma = 4.05e5, Dm = 9.99e-7) # define sorption material
        # original parameters: material = SorptionMaterial(Am = 0.09, Km = 1e-5, a = -10.908, b = 9333.1, Kma = 4.05e5, Dm = 9.99e-7) # define sorption material

sim_mode = 1  # 0: Polynomial, 1: InterfaceModel (diffusion model)


# ========================================= Curve Fitting Function =========================================
def run_simulation_and_get_results(args):
    """
    Runs the full CONTAM simulation and extracts results.
    """
    return run_simulation(args.filename, args.verbose, args.test, args.fitting)


def curve_fitting(args):
    """
    Fits the sorption model parameters to the experimental data.
    """
    print("Running curve fitting...")

    # Load experimental data
    exp_data_path = args.fitting
    data = pd.read_csv(exp_data_path)
    time_exp = data["Time"].values
    conc_exp = data["HCHO_Concentration"].values

    def objective(params):
        """Objective function to minimize."""
        if sim_mode == 0:
            material.a, material.b = params
        else:
            material.Kma, material.Dm = params

        T_array, Ca_array = run_simulation_and_get_results(args)
        conc_sim = np.interp(time_exp, T_array, Ca_array)
        error = np.sum((conc_exp - conc_sim) ** 2)
        # Print error for each iteration
        if sim_mode == 0:
            print(f"Optimizing Polynomial Model: a = {material.a:.4f}, b = {material.b:.4f}")
        else:
            print(f"Optimizing Interface Model: Kma = {material.Kma:.4e}, Dm = {material.Dm:.4e}")
        print(f"Error: {error:.4f}")
        return error

    # # Optimization method: minimize (L-BFGS-B)
    # # Initial guesses & bounds
    # if sim_mode == 0:
    #     initial_guess = [material.a, material.b]
    #     param_bounds = [(-1000, 10000), (-1000, 10000)]
    # else:
    #     initial_guess = [material.Kma, material.Dm]
    #     param_bounds = [(1e5, 1e7), (1e-8, 1e-4)]
    # # Optimize parameters
    # result = minimize(objective, initial_guess, bounds=param_bounds, method='L-BFGS-B', 
    #                   options={'eps': 1e-2, 'ftol': 1e-6, 'maxiter': 500})

    # Alternative optimization method: Differential Evolution
    # Parameter bounds (wider for exploration)
    if sim_mode == 0:
        param_bounds = [(-5000, 5000), (-5000, 5000)]
    else:
        param_bounds = [(1e4, 1e8), (1e-9, 1e-3)]  # More relaxed bounds
    # Use global optimization (differential evolution)
    result = differential_evolution(objective, bounds=param_bounds, strategy='best1bin',
                                    mutation=(0.5, 1.0), recombination=0.7, tol=1e-6, seed=42)

    # Update material properties with optimized values
    if sim_mode == 0:
        material.a, material.b = result.x
        optimized_param = f"Optimized Polynomial Model: a = {result.x[0]:.4f}, b = {result.x[1]:.4f}"
    else:
        material.Kma, material.Dm = result.x
        optimized_param = f"Optimized Interface Model: Kma = {result.x[0]:.4e}, Dm = {result.x[1]:.4e}"
    
    print(optimized_param)
    print("Curve fitting completed. Running simulation with optimized parameters...")

    # ===================================== Plot Results ===================================== #
    plt.figure(figsize=(8, 5))
    plt.scatter(time_exp, conc_exp, color="red", label="Experimental Data", marker="o")
    T_array, Ca_array = run_simulation_and_get_results(args)
    plt.plot(T_array, Ca_array, color="blue", label=optimized_param, linestyle="--")

    plt.xlabel("Time (s)")
    plt.ylabel("HCHO Concentration (ug/m3)")
    plt.legend()
    plt.title("Optimized Parameter Fitting for Sorption Models")
    plt.show()


# ========================================= Full Simulation Process =========================================
def run_simulation(prj_file_path, verbose, test, fitting):
    """
    Runs the full simulation process, using either default or optimized parameters.
    """
    my_prj = cxLib(prj_file_path, 0, True, callback_set_initial_values)
    my_prj.setVerbosity(verbose)

    if my_prj.setupSimulation(1) > 0:
        print("ERROR: Invalid simulation parameters.")
        print(f"ABORT - sim_not_ok Returned by setupSimulation() = {my_prj.setupSimulation(1)}")
        print(" => invalid simulation parameters for co-simulation.")
        my_prj.endSimulation()
        sys.exit(1)

    err_count = check_controls(my_prj, test, PV_PASS_CTRL, CO_SET_CTRL)
    if err_count > 0 or test is True:
        return
    
    # Initialize sorption model
    if sim_mode == 0:
        sorption_sink = Polynomial(material)
        if not fitting: print("Sim model: Polynomial")
    elif sim_mode == 1:
        sorption_sink = InterfaceModel(material)
        if not fitting: print("Sim model: Interface")
    else:
        print("ERROR: Invalid simulation mode.")
        sys.exit(1)

    # Initialize diffusion model if needed
    if sim_mode == 1:
        mat_diff = Diffusion(material)
        mat_diff.gen_mesh(depth=4e-3, delta_y=4e-4)

    # Get simulation details
    # ----- Get simulation run info
    start_day = my_prj.getSimStartDate()
    end_day = my_prj.getSimEndDate()
    start_time = my_prj.getSimStartTime()
    end_time = my_prj.getSimEndTime()
    dt = my_prj.getSimTimeStep()

    sim_begin_sec = (start_day - 1) * 86400 + start_time
    sim_end_sec = (end_day - 1) * 86400 + end_time

    # ----- Calculate the simulation duration in seconds and total time steps
    sim_begin_sec = (start_day - 1) * 86400 + start_time
    sim_end_sec = (end_day - 1) * 86400 + end_time
    if sim_begin_sec <= sim_end_sec:
        sim_duration_sec = sim_end_sec - sim_begin_sec
    else:
        sim_duration_sec = 365 * 86400 - sim_end_sec + sim_begin_sec
    num_time_steps = 0
    if sim_duration_sec != 0:
        num_time_steps = int(sim_duration_sec / dt)
        if not fitting: print(f"PRJ settings => Transient simulation w/ {num_time_steps} time steps. (Sorption material: Am={material.Am}, Km={material.Km}, a={material.a}, b={material.b}, Kma={material.Kma}, Dm={material.Dm})")
    else:
        if not fitting: print("PRJ settings => Steady state simulation. (Sorption material: Am={material.Am}, Km={material.Km}, a={material.a}, b={material.b}, Kma={material.Kma}, Dm={material.Dm})")

    # ----- Get the current date/time after initial steady state simulation
    current_date = my_prj.getCurrentDayOfYear()
    current_time = my_prj.getCurrentTimeInSec()
    if verbose > 0:
        print(f"Sim days = {start_day}:{end_day}")
        print(f"Sim times = {start_time}:{end_time}")
        print(f"Sim time step = {dt}")
        print(f"Number of steps = {num_time_steps}")

    # ----- Initialize result files

    # ----- Output initial results.
    current_date = my_prj.getCurrentDayOfYear()
    current_time = my_prj.getCurrentTimeInSec()
    Ca = my_prj.getOutputControlValue(PV_PASS_CTRL.index) # HCHO concn in room air [ug/m3]
    # print initial values, except when fitting
    if not fitting: 
        print("day\ttime\tCa[ug/m3]\tS[ug/s]\tMs[ug/m2]\tCm[ug/m3]") 
        print(f"{current_date}\t{current_time}\t{Ca}\t0\t0\t0") # print initial values
    Ms = 0 # initial adsorbed mass [ug/m2]

    if fitting:
        T_array = []
        Ca_array = []

    # initialize material diffusion model
    mat_diff = Diffusion(material)
    mat_diff.gen_mesh(depth = 4e-3, delta_y = 4e-4)

    # if sim_mode == 1, export concentration distribution in material, save to file (save Cm_array)
    if sim_mode == 1:
        mat_Cm_dist_print = "Material Concentration Distribution [ug/m3] at y depths [m]\n"
        mat_Cm_dist_print += "day\ttime\t" + "\t".join([f"y={y:.4f}" for y in mat_diff.y_array]) + "\n"
        mat_Cm_dist_print += f"{current_date}\t{current_time}\t" + "\t".join([f"{Cm:.4f}" for Cm in mat_diff.Cm_array]) + "\n"

    for i in range(num_time_steps):
        if sim_mode == 0:
            S = sorption_sink.get_S(Ca)
            Ms = sorption_sink.get_Ms(dt)
            Cm = sorption_sink.get_Cm()
        else:
            S = sorption_sink.get_S(Ca)
            Ms = sorption_sink.get_Ms(dt)
            Cm_array = mat_diff.solve_implicit_central(dt, S)
            Cm = sorption_sink.get_Cm(Cm_array[-1])

            # export concentration distribution in material, save to file (save Cm_array)
            mat_Cm_dist_print += f"{current_date}\t{current_time}\t" + "\t".join([f"{Cm:.4f}" for Cm in mat_diff.Cm_array]) + "\n"

        my_prj.setInputControlValue(CO_SET_CTRL.index, -1 * S)

        # =====================================================================
        # Run next time step.
        # =====================================================================
        my_prj.doSimStep(1)

        # =====================================================================
        # Tasks to perform AFTER current time step.
        # =====================================================================
        current_date = my_prj.getCurrentDayOfYear()
        current_time = my_prj.getCurrentTimeInSec()
        Ca = my_prj.getOutputControlValue(PV_PASS_CTRL.index)
        if fitting:
            T_array.append(i * dt)
            Ca_array.append(Ca)
        if not fitting: 
            print(f"{current_date}\t{current_time}\t{Ca}\t{S}\t{Ms}\t{Cm}") # print results, except when fitting

    my_prj.endSimulation()

    if sim_mode == 1 and not fitting:
            # save mat_Cm_dist_print to file
            with open("mat_Cm_dist.txt", "w") as f:
                f.write(mat_Cm_dist_print)
    
    return T_array, Ca_array


# ====================================================== check_controls() ========================
def check_controls(cxl, test, pv_ctrl, co_ctrl) -> int:
    '''
    Check to see if controls are properly defined within the PRJ.
    There should be a named SPLIT/PASS control to provide the HCHO conc
    to this module, i.e., process value (pv) and a named SET control
    for the sorption sink, i.e., controller output (co).

        Args:
            cxl: `cxLib` instance for specific PRJ
            test: `bool`
                Print list of input/output controls in the PRJ
            pv_ctrl: `CtrlIdxName`
                Index and Name of cxLib.outputControls[] list item
            co_ctrl: `CtrlIdxName`
                Index and Name of cxLib.inputControls[] list item

        :returns: Number of errors found. 0 if `test` is True
        :rtype: int
    '''
    input_error = 0
    if test is True:
        # ----- CONTROLS PROPERTIES
        print("----- OUTPUT CONTROLS -----")
        for i, ctrl in zip(range(cxl.nOutputControls), cxl.outputControls):
            ctrl_name = ctrl.name
            print(f"{i+1} {ctrl_name}")
        print("\n")
        print("----- INPUT CONTROLS -----")
        for i, ctrl in zip(range(cxl.nInputControls), cxl.inputControls):
            ctrl_name = ctrl.name
            print(f"{i+1} {ctrl_name}")
        print("\n")
        return input_error

    # Check to see if control indices are set correctly.
    if pv_ctrl.index > cxl.nOutputControls or \
       cxl.outputControls[pv_ctrl.index-1].name != pv_ctrl.name:
        print(f"Incorrect index ({pv_ctrl.index}) for output control: {pv_ctrl.name}")
        input_error += 1
    if co_ctrl.index > cxl.nInputControls or \
       cxl.inputControls[co_ctrl.index-1].name != co_ctrl.name:
        print(f"Incorrect index ({co_ctrl.index}) for input control: {co_ctrl.name}")
        input_error += 1

    if input_error > 0:
        print("Run this script with -t option to determine correct control indices.")

    return input_error


# ========================================= callback_set_initial_values() =========================
def callback_set_initial_values(cxl: cxLib):
    '''
    Callback function to initialize sorption sink control value (cv)
    '''
    cxl.setInputControlValue(SINK_SET.index, 0) # assuming as 0


# ========================================= Main Function =========================================
def main():
    parser = ArgumentParser(
        prog='Sorption',
        description='Implements a sorption sink with optional curve fitting.'
    )

    parser.add_argument('filename', help="Path to the CONTAM project file.")
    parser.add_argument('-v', '--verbose', type=int, choices=range(0, 3), default=0,
                        help="Define verbose output level: 0=Min, 1=Medium, 2=Maximum.")
    parser.add_argument('-t', '--test', action='store_true', default=False,
                        help="Do not run simulation, but output lists of available controls.")
    parser.add_argument('-f', '--fitting', type=str, metavar="EXPERIMENTAL_DATA", default=None,
                        help="Run curve fitting using the specified experimental data file (must be .csv).")
    parser.set_defaults(verbose=0, test=False, fitting=None)
    args = parser.parse_args()

    # Validate fitting argument
    if args.fitting:
        if not os.path.exists(args.fitting):
            print(f"ERROR: Experimental data file not found: {args.fitting}")
            sys.exit(1)
        if not args.fitting.lower().endswith(".csv"):
            print("ERROR: Experimental data file must be a CSV file (*.csv).")
            sys.exit(1)

        # Run fitting function before simulation
        curve_fitting(args)
        return

    # Run simulation
    run_simulation(args.filename, args.verbose, args.test)


# ========================================= Run Script =========================================
if __name__ == "__main__":
    main()