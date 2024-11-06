import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from Classes import StanleyController, VehicleDynamicModel, VehicleKinemaicModel, MPC
import Functions
import json
from copy import copy, deepcopy

with open('vehicle_config.json', "r") as f:
    vehicle_params = json.loads(f.read())
simulation_params = {'dt': 0.01, 't_end': 50, 'ego_frame_placement': 'front_axle', 'velocity_KPH': 25,
                     'path_spacing': 1.0,
                     'model': 'Kinematic', #'Kinematic', 'Dynamic'
                     'animate': True, 'plot_results': True, 'save_results': False}
# generate path
scenario = 'random_curvature' # 'sin', 'straight_line', 'square', shiba, random_curvature,turn, original_from_repo
traj_spline_x, traj_spline_y, traj_spline_psi, _, s = Functions.calc_desired_path(scenario, ds=simulation_params['path_spacing'])
# create vehicle agent
error_x = 0.0
error_y = 0.0
vehicle_obj = VehicleKinemaicModel(x=traj_spline_x[0] + error_x, y=traj_spline_y[0] + error_y, psi=traj_spline_psi[0],
                                   vehicle_params=copy(vehicle_params), simulation_params=copy(simulation_params),
                                   steering_uncertainty_factor=1.0, lr_uncertainty_factor=1.0, WB_uncertainty_factor=1.0)
t = np.arange(0, simulation_params['t_end'], simulation_params['dt'])
vehicle_obj.vx = simulation_params['velocity_KPH'] / 3.6 * 0
# stanly gain
Ks = 1.0
SC = StanleyController(Ks=Ks, desired_traj_x=traj_spline_x, desired_traj_y=traj_spline_y, desired_traj_psi=traj_spline_psi)
ref_path_point = np.vstack([traj_spline_x, traj_spline_y]).T
speed_profile = simulation_params['velocity_KPH'] * np.ones(len(traj_spline_psi)) / 3.6
MPC_obj = MPC(copy(vehicle_params), copy(simulation_params), ref_path_points=ref_path_point, ref_path_heading=traj_spline_psi,
              speed_profile=speed_profile)

MPC_obj.calc_steering_command(vehicle_obj.clone())
vehicle_temp = vehicle_obj.clone()
vehicle_temp.simulation_params["dt"] = MPC_obj.MPC_params.dt
z_ref, _ = MPC_obj.calc_ref_trajectory_in_T_step(vehicle_temp)
x_ref = z_ref[0, :]
y_ref = z_ref[1, :]
v_ref = z_ref[2, :]
psi_ref = z_ref[3, :]
# vehicle_obj.update(a=MPC_obj.a_exc, delta=MPC_obj.delta_exc)



animation_figure = plt.figure()
vehicle_animation_axis = plt.subplot(1, 1, 1)
ref_traj_line = vehicle_animation_axis.plot(traj_spline_x, traj_spline_y, color='gray', linewidth=2.0)
vehicle_line = Functions.draw_car(vehicle_obj.x, vehicle_obj.y, vehicle_obj.psi, steer=0, car_params=vehicle_params, ax=vehicle_animation_axis)
vehicle_animation_axis.axis("equal")
vehicle_animation_axis.grid(True)
vehicle_animation_axis.set_xlabel('x [m]')
vehicle_animation_axis.set_ylabel('y [m]')
vehicle_animation_axis.scatter(MPC_obj.x_opt, MPC_obj.y_opt, s=10)
vehicle_animation_axis.set_xlim(vehicle_obj.x - 25, vehicle_obj.x + 25)
vehicle_animation_axis.set_ylim(vehicle_obj.y - 25, vehicle_obj.y + 25)
vehicle_animation_axis.grid(True)

plt.figure('state')
plt.subplot(4,2,1)
plt.plot(x_ref)
plt.plot(MPC_obj.x_opt)
plt.legend(["ref", "optimal solution"])
plt.ylabel("x [m]")
plt.grid(True)
plt.subplot(4,2,2)
plt.plot(x_ref - MPC_obj.x_opt)
plt.ylabel(r"$e_x [m]$")
plt.grid(True)
txt = "x error cost component = " + str("%.2f" % MPC_obj.cost_dict["x_cost"])
plt.title("state error cost component = " + str("%.2f" % MPC_obj.cost_dict["state_error_cost"]), fontsize=9)
plt.annotate(txt, xy=(0.02, 0.01), xycoords='axes fraction',
             fontsize=9, ha='left', va='bottom',
             bbox=dict(facecolor='white', alpha=0.5))
plt.subplot(4,2,3)
plt.plot(y_ref)
plt.plot(MPC_obj.y_opt)
plt.ylabel("y [m]")
plt.grid(True)
plt.subplot(4,2,4)
plt.plot(y_ref - MPC_obj.y_opt)
plt.ylabel(r"$e_y [m]$")
plt.grid(True)
txt = "y error cost component = " + str("%.2f" % MPC_obj.cost_dict["y_cost"])
plt.annotate(txt, xy=(0.02, 0.01), xycoords='axes fraction',
             fontsize=9, ha='left', va='bottom',
             bbox=dict(facecolor='white', alpha=0.5))
plt.subplot(4,2,5)
plt.plot(v_ref)
plt.plot(MPC_obj.v_opt)
plt.ylabel("v [m/sec]")
plt.grid(True)
plt.subplot(4,2,6)
plt.plot(v_ref - MPC_obj.v_opt)
plt.ylabel(r"$e_v [m/sec]$")
plt.grid(True)
txt = "v error cost component = " + str("%.2f" % MPC_obj.cost_dict["v_cost"])
plt.annotate(txt, xy=(0.02, 0.01), xycoords='axes fraction',
             fontsize=9, ha='left', va='bottom',
             bbox=dict(facecolor='white', alpha=0.5))
plt.subplot(4,2,7)
plt.plot(psi_ref)
plt.plot(MPC_obj.yaw_opt)
plt.ylabel("psi [rad]")
plt.grid(True)
plt.subplot(4,2,8)
plt.plot(psi_ref - MPC_obj.yaw_opt)
plt.ylabel(r"$e_{psi} [rad]$")
plt.grid(True)
txt = "psi error cost component = " + str("%.2f" % MPC_obj.cost_dict["psi_cost"])
plt.annotate(txt, xy=(0.02, 0.01), xycoords='axes fraction',
             fontsize=9, ha='left', va='bottom',
             bbox=dict(facecolor='white', alpha=0.5))


plt.figure('actuation')
plt.subplot(211)
plt.plot(MPC_obj.a_opt)
plt.ylabel(r'$a [m/sec^2]$')
txt = ("a cost component = " + str("%.2f" % MPC_obj.cost_dict["a_cost"]) + " \n" +
       "a change component = " + str("%.2f" % MPC_obj.cost_dict["a_change_cost"]))
plt.annotate(txt, xy=(0.02, 0.01), xycoords='axes fraction',
             fontsize=9, ha='left', va='bottom',
             bbox=dict(facecolor='white', alpha=0.5))
txt = ("actuation cost component = " + str("%.2f" % MPC_obj.cost_dict["actuation_cost"]) + " \n" +
       "actuation change component = " + str("%.2f" % MPC_obj.cost_dict["actuation_change_cost"]))
plt.title(txt)

plt.grid(True)
plt.subplot(212)
plt.plot(MPC_obj.delta_opt)
plt.ylabel(r'$delta [rad]$')
txt = ("delta cost component = " + str("%.2f" % MPC_obj.cost_dict["delta_cost"]) + "\n" +
       "delta change component = " + str("%.2f" % MPC_obj.cost_dict["delta_change_cost"]))
plt.annotate(txt, xy=(0.02, 0.01), xycoords='axes fraction',
             fontsize=9, ha='left', va='bottom',
             bbox=dict(facecolor='white', alpha=0.5))
plt.grid(True)
plt.show()