"""
MOF-SorptionControl.py - simulate air contaminant adsorbed by MOF material.
"""

import os
from argparse import ArgumentParser
from dataclasses import dataclass
from contamxpy import cxLib
from material_adsorption import SorptionMaterial
from material_adsorption import Sorption


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
1: chamber L
2: chamber H
'''
sim_case = 1

match sim_case:
    case 0:
        material = SorptionMaterial(0.045, 9.5e-4, -44.03, 3751.57) # define sorption material
    case 1:
        material = SorptionMaterial(0.09, 5.8083e-4, -1.1, 1097.26) # define sorption material
    case 2: 
        material = SorptionMaterial(0.09, 1e-5, -10.908, 9333.1) # define sorption material

'''
Am: float
Km: float
a: float
b: float
'''


# ====================================================== check_controls() =====


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


# ========================================= callback_set_initial_values() =====


def callback_set_initial_values(cxl: cxLib):
    '''
    Callback function to initialize sorption sink control value (cv)
    '''
    cxl.setInputControlValue(SINK_SET.index, 0) # assuming as 0


# ================================================================ main() =====


def main():
    '''
    This program takes the full name of a PRJ file and runs the simulation from
    beginning to end. It implements a sorption sink.
    '''
    # ----- Manage option parser
    parser = ArgumentParser(
        prog='Sorption',
        description='Implements a sorption sink.'
    )
    parser.add_argument('filename')
    parser.add_argument('-v', '--verbose', type=int, choices=range(0, 3),
                        help="define verbose output level: 0=Min, 1=Medium, 2=Maximum.")
    parser.add_argument('-t', '--test', action='store_true',
                        help="do not run simulation, but output lists of available controls.")
    parser.set_defaults(verbose=0, test=False)
    args = parser.parse_args()

    # ----- Process command line options -v/-t
    verbose = args.verbose
    test = args.test

    prj_file_path = args.filename

    if not os.path.exists(prj_file_path):
        print(f"ERROR: PRJ file not found: {prj_file_path}")
        return

    msg_cmd = args
    print(msg_cmd, "\n")

    if verbose > 1:
        print(f"cxLib attributes =>\n{chr(10).join(map(str, dir(cxLib)))}\n")

    # ----- Initialize contamx-lib object w/ wp_mode and cb_option.
    #   wp_mode = 0 => use wind pressure profiles, WTH and CTM files or
    # associated API calls.
    #   cb_option = True => set callback function to get PRJ INFO
    # from the ContamXState.
    my_prj = cxLib(prj_file_path, 0, True, callback_set_initial_values)
    my_prj.setVerbosity(verbose)

    # ----- Setup Simulation for PRJ
    sim_not_ok = my_prj.setupSimulation(1)
    if sim_not_ok > 0:
        print(f"ABORT - sim_not_ok Returned by setupSimulation() = {sim_not_ok}")
        print(" => invalid simulation parameters for co-simulation.")
        my_prj.endSimulation()
        return

    err_count = check_controls(my_prj, test, PV_PASS_CTRL, CO_SET_CTRL)
    if err_count > 0 or test is True:
        return

    # =========================================================================
    # USER INPUT - Initialize controller class.
    # =========================================================================

    sorption_sink = Sorption(material)

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
        print(f"PRJ settings => Transient simulation w/ {num_time_steps} time steps. (Sorption material: Am={material.Am}, Km={material.Km}, a={material.a}, b={material.b})")
    else:
        print("PRJ settings => Steady state simulation. (Sorption material: Am={material.Am}, Km={material.Km}, a={material.a}, b={material.b})")

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
    Cr = my_prj.getOutputControlValue(PV_PASS_CTRL.index) # HCHO concn in room air [ug/m3]
    print("day\ttime\tCr[ug/m3]\tS[ug/s]\tMs[ug/m2]\tCs[ug/m3]")
    print(f"{current_date}\t{current_time}\t{Cr}\t0\t0\t0") # set initial values

    # =========================================================================
    # Run Simulation
    # =========================================================================
    for i in range(num_time_steps):

        # =====================================================================
        # Tasks to perform BEFORE current time step.
        # =====================================================================
        # Calculate sorption rate S, adsorbed mass Ms, and gas phase conc on material surface Cs, based on room air conc Cr

        S = sorption_sink.get_S(Cr)
        Ms = sorption_sink.get_Ms(dt)
        Cs = sorption_sink.get_Cs()
        
        my_prj.setInputControlValue(CO_SET_CTRL.index, -1 * S) # Set sorption sink to negative because it is defined with a generation rate in PRJ

        # =====================================================================
        # Run next time step.
        # =====================================================================
        my_prj.doSimStep(1)

        # =====================================================================
        # Tasks to perform AFTER current time step.
        # =====================================================================
        current_date = my_prj.getCurrentDayOfYear()
        current_time = my_prj.getCurrentTimeInSec()
        Cr = my_prj.getOutputControlValue(PV_PASS_CTRL.index)
        print(f"{current_date}\t{current_time}\t{Cr}\t{S}\t{Ms}\t{Cs}")

    my_prj.endSimulation()

# --- End main() ---#


if __name__ == "__main__":
    main()
