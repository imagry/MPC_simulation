import os
import sys
import math
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from Classes import StanleyController, VehicleDynamicModel, VehicleKinemaicModel, MPC
import Functions
import json
import CurvesGenerator.cubic_spline as cs
from tqdm import tqdm

with open('vehicle_config.json', "r") as f:
    vehicle_params = json.loads(f.read())
simulation_params = {'dt': 0.01, 't_end': 50, 'ego_frame_placement': 'front_axle', 'velocity_KPH': 10,
                     'path_spacing': 0.1,
                     'model': 'Kinematic', #'Kinematic', 'Dynamic'
                     'animate': True, 'plot_results': True, 'save_results': False}
# generate path
traj_samples_x = np.arange(0, 50, 0.5)
scenario = 'random_curvature' # 'sin', 'straight_line', 'square', shiba, random_curvature
traj_spline_x, traj_spline_y, traj_spline_psi, _, s = Functions.calc_desired_path(scenario)
traj_length = s[-1]
# create vehicle agent
error_x = 0.0
error_y = 0.0
if simulation_params['model'] == 'Dynamic':
    vehicle_obj = VehicleDynamicModel(x=traj_spline_x[0] + error_x, y=traj_spline_y[0] + error_y, psi=traj_spline_psi[0],
                                      vehicle_params=vehicle_params, simulation_params=simulation_params,
                                      steering_uncertainty_factor=1.0, lr_uncertainty_factor=1.0, WB_uncertainty_factor=1.0,
                                      m_uncertainty_factor=1.0, I_uncertainty_factor=1.0, C_uncertainty_factor=1.0)
elif simulation_params['model'] == 'Kinematic':
    vehicle_obj = VehicleKinemaicModel(vehicle_params=vehicle_params, simulation_params=simulation_params,
                                       steering_uncertainty_factor=1.0, lr_uncertainty_factor=1.0, WB_uncertainty_factor=1.0)
t = np.arange(0, simulation_params['t_end'], simulation_params['dt'])
vehicle_obj.vx = simulation_params['velocity_KPH'] / 3.6
# variable to keep
x = []
y = []
psi = []
x.append(vehicle_obj.x)
y.append(vehicle_obj.y)
psi.append(vehicle_obj.psi)

delta = []
velocity = []
ef = []
psi_traj = []
t_acumulated = []
# stanly gain
Ks = 1.0
SC = StanleyController(Ks=Ks, desired_traj_x=traj_spline_x, desired_traj_y=traj_spline_y, desired_traj_psi=traj_spline_psi)
ref_path_point = np.vstack([traj_spline_x, traj_spline_y]).T
speed_profile = simulation_params['velocity_KPH'] * np.ones(len(traj_spline_psi)) / 3.6
MPC_obj = MPC(vehicle_params, simulation_params, ref_path_points=ref_path_point, ref_path_heading=traj_spline_psi,
              speed_profile=speed_profile)
stop_condition = False
i = 0
if simulation_params['animate']:
    # plt.ion()
    animation_figure = plt.figure('animation')
    vehicle_animation_axis = plt.subplot(1, 2, 1)
    plt.title("stanly lateral control")
    # vehicle_animation_axis.clear()
    ref_traj_line = vehicle_animation_axis.plot(traj_spline_x, traj_spline_y, color='gray', linewidth=2.0)
    vehicle_traj_line = vehicle_animation_axis.plot([vehicle_obj.x], [vehicle_obj.y], linewidth=2.0, color='darkviolet')
    vehicle_line = Functions.draw_car(vehicle_obj.x, vehicle_obj.y, vehicle_obj.psi, steer=0, car_params=vehicle_params, ax=vehicle_animation_axis)
    vehicle_animation_axis.axis("equal")
    vehicle_animation_axis.grid(True)
    vehicle_animation_axis.set_xlabel('x [m]')
    vehicle_animation_axis.set_ylabel('y [m]')
    lateral_error_axis = plt.subplot(2, 2, 2)
    lateral_error_axis.grid(True)
    lateral_error_axis.set_ylabel('lateral error [m]')
    velocity_axis = plt.subplot(2, 2, 4)
    velocity_axis.grid(True)
    velocity_axis.set_ylabel('velocity [m/sec]')

animation_dt = 0.1
ndt = int(animation_dt/simulation_params['dt'])
pbar = tqdm(total=len(t))
s = 0
while not stop_condition:
    ti = t[i]
    t_acumulated.append(ti)
    SC.calc_steering_command(vehicle_obj)
    MPC_obj.calc_steering_command(vehicle_obj)
    vehicle_obj.update(a=MPC_obj.a_exc, delta=MPC_obj.delta_exc)
    # store values
    x.append(vehicle_obj.x)
    y.append(vehicle_obj.y)
    dx = x[-1] - x[-2]
    dy = y[-1] - y[-2]
    s += np.linalg.norm(np.array([dx, dy]))
    psi.append(vehicle_obj.psi)
    delta.append(MPC_obj.delta_exc)
    ef.append(SC.ef)
    velocity.append(vehicle_obj.vx)
    # psi_traj.append(psi_traj_i)
    cond1 = i >= t.shape[0] - 1
    cond2 = np.linalg.norm(np.array([vehicle_obj.x - traj_spline_x[-1], vehicle_obj.y - traj_spline_y[-1]])) < 1.0
    cond3 = s >= traj_length * 5.0
    stop_condition = cond1 or cond2 or cond3
    if cond1: print('condition 1 is met')
    if cond2: print('condition 2 is met')
    if cond3: print('condition 3 is met')
    i += 1
    pbar.update(1)
    if np.mod(i, ndt) == 0 and simulation_params['animate']:
        plt.cla()
        vehicle_animation_axis.clear()
        vehicle_animation_axis.plot(traj_spline_x, traj_spline_y, color='gray', linewidth=2.0)
        # vehicle_traj_line.remove()
        vehicle_traj_line = vehicle_animation_axis.plot(x, y, linewidth=2.0, color='darkviolet')
        vehicle_line = Functions.draw_car(vehicle_obj.x, vehicle_obj.y, vehicle_obj.psi, steer=MPC_obj.delta_exc,
                                          car_params=vehicle_params, ax=vehicle_animation_axis)
        vehicle_animation_axis.scatter(MPC_obj.x_opt, MPC_obj.y_opt, s=10)
        vehicle_animation_axis.set_xlim(vehicle_obj.x - 10, vehicle_obj.x + 10)
        vehicle_animation_axis.set_ylim(vehicle_obj.y - 10, vehicle_obj.y + 10)
        vehicle_animation_axis.grid(True)
        lateral_error_axis.clear()
        lateral_error_axis.plot(t_acumulated, ef)
        lateral_error_axis.grid(True)
        velocity_axis.clear()
        velocity_axis.plot(t_acumulated, velocity)
        velocity_axis.grid(True)
        vehicle_animation_axis.set_xlabel('x [m]')
        vehicle_animation_axis.set_ylabel('y [m]')
        lateral_error_axis.set_xlabel('t [sec]')
        lateral_error_axis.set_ylabel('lateral error [m]')
        velocity_axis.set_ylabel('velocity [m/sec]')
        velocity_axis.set_xlabel('t [sec]')
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event:
                                     [exit(0) if event.key == 'escape' else None])
        plt.pause(0.01)
pbar.close()
if simulation_params['save_results']:
    vehicle_states_dic = {'x': x, 'y': y, 'psi': psi, 'time': t[:i + 1]}
    control_states_dic = {'delta': delta, 'ef': ef, 'time': t[:i]}
    trajectory_dic = {'traj_x': traj_spline_x, 'traj_y': traj_spline_y, 'traj_psi': traj_spline_psi}
    Functions.save_csv(vehicle_states_dic, path='./data/vehicle_states.csv', print_message=True)
    Functions.save_csv(control_states_dic, path='./data/control_states.csv', print_message=True)
    Functions.save_csv(trajectory_dic, path='./data/desired_traj.csv', print_message=True)
if simulation_params['plot_results']:
    animation_figure = plt.figure('stanley control on dynamic model')
    vehicle_animation_axis = plt.subplot(1, 2, 1)
    plt.title("BEV"), plt.xlabel('[m]'), plt.ylabel('[m]')
    ref_traj_line = vehicle_animation_axis.plot(traj_spline_x, traj_spline_y, color='gray', linewidth=2.0)
    vehicle_traj_line = vehicle_animation_axis.plot(x, y, linewidth=2.0, color='darkviolet')
    vehicle_animation_axis.axis("equal")
    vehicle_animation_axis.grid(True)
    lateral_error_axis = plt.subplot(1, 2, 2)
    plt.title('lateral error')
    lateral_error_axis.plot(t[:i], abs(np.array(ef)))
    lateral_error_axis.grid(True)
    plt.xlabel('time [sec]'), plt.ylabel('ef [m]')
    plt.show()
