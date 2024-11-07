import math
from multipledispatch import dispatch
import matplotlib.pyplot as plt
import numpy as np
import Functions
from Functions import fold_angles, epsilon_limit
from os.path import join
import pandas as pd
import cvxpy
import copy
import CurvesGenerator.cubic_spline as cs
class VehicleKinemaicModel:
    def __init__(self,
                 vehicle_params:dict, simulation_params:dict,
                 x=0.0, y=0.0, psi=0.0, vx=0.0,
                 steering_uncertainty_factor=None,
                 max_steering_error=0.2,
                 lr_uncertainty_factor=None,
                 max_lr_error=0.2,
                 WB_uncertainty_factor=None,
                 max_WB_error=0.2):
        self.__dict__.update(locals())  # initializes the class attributes with inputs
        if steering_uncertainty_factor is None:
            self.steering_uncertainty_factor = 1 + max_steering_error * (np.random.random_sample(1) - 0.5)[-1]
        if lr_uncertainty_factor is None:
            self.lr_uncertainty_factor = 1 + max_lr_error * (np.random.random_sample(1) - 0.5)[-1]
        if WB_uncertainty_factor is None:
            self.WB_uncertainty_factor = 1 + max_WB_error * (np.random.random_sample(1) - 0.5)[-1]
        self.lr = self.vehicle_params['lr'] * self.lr_uncertainty_factor
        self.WB = self.vehicle_params['WB']* self.WB_uncertainty_factor
        # self.vx = self.simulation_params['velocity_KPH'] / 3.6

    def update(self, a, delta):
        delta = self.limit_input(delta) * self.steering_uncertainty_factor
        if self.simulation_params['ego_frame_placement'] == 'CG':
            delta = self.limit_input(delta)  * self.steering_uncertainty_factor
            beta = np.arctan(self.lr * np.tan(delta)/ self.WB)
            self.x += self.vx * np.cos(self.psi + beta) * self.simulation_params['dt']
            self.y += self.vx * np.sin(self.psi + beta) * self.simulation_params['dt']
            self.psi += self.vx * np.cos(beta) / self.WB * np.tan(delta) * self.simulation_params['dt']
        elif self.simulation_params['ego_frame_placement'] == 'front_axle':
            # delta = self.limit_input(delta)
            # self.x += self.v * math.cos(self.psi) * self.simulation_params['dt']
            # self.y += self.v * math.sin(self.psi) * self.simulation_params['dt']
            # self.psi += self.v / self.WB * math.tan(delta) * self.simulation_params['dt']
            delta = self.limit_input(delta)
            self.x += self.vx * np.cos(self.psi + delta) * self.simulation_params['dt']
            self.y += self.vx * np.sin(self.psi + delta) * self.simulation_params['dt']
            self.psi += self.vx * np.sin(delta) / self.WB * self.simulation_params['dt']
            # self.x += self.vx * np.cos(self.psi) * self.simulation_params['dt']
            # self.y += self.vx * np.sin(self.psi) * self.simulation_params['dt']
            # self.psi += self.vx / self.WB * np.tan(delta) * self.simulation_params['dt']
        elif self.simulation_params['ego_frame_placement'] == 'rear_axle':
            delta = self.limit_input(delta)
            self.x += self.vx * np.cos(self.psi) * self.simulation_params['dt']
            self.y += self.vx * np.sin(self.psi) * self.simulation_params['dt']
            # self.psi += self.vx / self.WB * np.tan(delta) * self.WB ##this looks wrong check eq
            self.psi += self.vx * np.sin(delta) / self.WB * self.simulation_params['dt']
        else:
            raise 'invalid ego frame placement'
        self.vx += a * self.simulation_params['dt']
    def limit_input(self, delta):
        delta = max(delta, -self.vehicle_params['MAX_STEER'])
        delta = min(delta, self.vehicle_params['MAX_STEER'])

        # if delta > C.MAX_STEER:
        #     return C.MAX_STEER

        # if delta < -C.MAX_STEER:
        #     return -C.MAX_STEER

        return delta
    def clone(self):
        return copy.deepcopy(self)


class VehicleDynamicModel:

    def __init__(self,
                 vehicle_params:dict, simulation_params:dict,
                 x=0.0, y=0.0, psi=0.0, v=0.0,
                 steering_uncertainty_factor=None,
                 max_steering_error=0.2,
                 lr_uncertainty_factor=None,
                 max_lr_error=0.2,
                 WB_uncertainty_factor=None,
                 max_WB_error=0.2,
                 m_uncertainty_factor=None,
                 max_m_error=0.2,
                 I_uncertainty_factor=None,
                 max_I_error=0.2,
                 C_uncertainty_factor=None,
                 max_C_error=0.2):
        self.__dict__.update(locals())  # initializes the class attributes with inputs
        if steering_uncertainty_factor is None:
            self.steering_uncertainty_factor = 1 + max_steering_error * (np.random.random_sample(1) - 0.5)[-1]
        if lr_uncertainty_factor is None:
            self.lr_uncertainty_factor = 1 + max_lr_error * (np.random.random_sample(1) - 0.5)[-1]
        if WB_uncertainty_factor is None:
            self.WB_uncertainty_factor = 1 + max_WB_error * (np.random.random_sample(1) - 0.5)[-1]
        if m_uncertainty_factor is None:
            self.m_uncertainty_factor = 1 + max_m_error * (np.random.random_sample(1) - 0.5)[-1]
        if I_uncertainty_factor is None:
            self.I_uncertainty_factor = 1 + max_I_error * (np.random.random_sample(1) - 0.5)[-1]
        if C_uncertainty_factor is None:
            self.C_uncertainty_factor = 1 + max_C_error * (np.random.random_sample(1) - 0.5)[-1]
        self.lr = self.vehicle_params['lr'] * self.lr_uncertainty_factor
        self.WB = self.vehicle_params['WB'] * self.WB_uncertainty_factor
        self.m = self.vehicle_params['m'] * self.m_uncertainty_factor
        self.I = self.vehicle_params['I'] * self.I_uncertainty_factor
        self.C = self.vehicle_params['C'] * self.C_uncertainty_factor
        self.vx = self.simulation_params['velocity_KPH'] / 3.6
        self.vy = 0.0
        self.alpha_r = 0.0
        self.alpha_f = 0.0
        self.psi_dot = 0.0
        if self.simulation_params['ego_frame_placement'] == 'CG':
            self.x_cg = self.x
            self.y_cg = self.y
        elif self.simulation_params['ego_frame_placement'] == 'front_axle':
            lf = self.WB - self.lr
            self.x_cg = self.x - lf * np.cos(self.psi)
            self.y_cg = self.y - lf * np.sin(self.psi)
        elif self.simulation_params['ego_frame_placement'] == 'rear_axle':
            lf = self.WB - self.lr
            self.x_cg = self.x + self.lr * np.cos(self.psi)
            self.y_cg = self.y + self.lr * np.sin(self.psi)
        else:
            raise 'invalid ego frame placement'
    def update(self, a, delta):
        delta = self.limit_input(delta) * self.steering_uncertainty_factor
        lf = self.WB - self.lr
        if self.vx ** 2 + self.vy ** 2 > 0:
            self.alpha_r = math.atan2(self.vy - self.psi_dot * self.lr, self.vx) # rear tire slip angle
            Fyr = -2 * self.C * np.sin(self.alpha_r)
            self.alpha_f = math.atan2(self.vy + self.psi_dot * lf, self.vx) - delta # front tire slip angle
            Fyf = -2 * self.C * np.sin(self.alpha_f)
        else:
            Fyr = 0.0
            Fyf = 0.0
        # linear acceleration equation
        vy_dot = 1 / self.m * Fyr + 1 / self.m * Fyf * np.cos(delta) - self.vx * self.psi_dot
        # vy_dot_lin = - 4 * self.C / self.m / self.vx * self.vy
        # rotational acceleration equation
        psi_dotdot = - self.lr / self.I * Fyr + np.cos(delta) * lf / self.I * Fyf

        self.vy += vy_dot * self.simulation_params['dt']
        self.psi_dot += psi_dotdot * self.simulation_params['dt']
        self.psi += self.psi_dot * self.simulation_params['dt']
        self.x_cg += (self.vx * np.cos(self.psi) - self.vy * np.sin(self.psi)) * self.simulation_params['dt']
        self.y_cg += (self.vx * np.sin(self.psi) + self.vy * np.cos(self.psi)) * self.simulation_params['dt']
        self.vx += a * self.simulation_params['dt']
        if self.simulation_params['ego_frame_placement'] == 'CG':
            self.x = self.x_cg
            self.y = self.y_cg
        elif self.simulation_params['ego_frame_placement'] == 'front_axle':
            self.x = self.x_cg + lf * np.cos(self.psi)
            self.y = self.y_cg + lf * np.sin(self.psi)
        elif self.simulation_params['ego_frame_placement'] == 'rear_axle':
            self.x = self.x_cg - self.lr * np.cos(self.psi)
            self.y = self.y_cg - self.lr * np.sin(self.psi)
        else:
            raise 'invalid ego frame placement'
    def limit_input(self, delta):
        delta = max(delta, -self.vehicle_params['MAX_STEER'])
        delta = min(delta, self.vehicle_params['MAX_STEER'])

        # if delta > C.MAX_STEER:
        #     return C.MAX_STEER

        # if delta < -C.MAX_STEER:
        #     return -C.MAX_STEER

        return delta


class StanleyController:

    def __init__(self, Ks, desired_traj_x, desired_traj_y, desired_traj_psi):
        self.__dict__.update(locals())  # initializes the class attributes with inputs
        self.target_index = None
        self.ef = None
        self.psi_traj = None
        self.delta = 0.0
        self.psi_e = None
        self.eps = 0.1
        self.num_points_to_average_ref_heading = 1

    @dispatch(VehicleKinemaicModel)
    def calc_theta_e_and_ef(self, vehicle: VehicleKinemaicModel):
        """
        calc theta_e and ef.
        theta_e = theta_car - theta_path
        ef = lateral distance in CG

        :param vehicle: current information of vehicle
        :return: theta_e and ef
        """
        fx = vehicle.x  # + car_params.WB * math.cos(vehicle.yaw)
        fy = vehicle.y  # + car_params.WB * math.sin(vehicle.yaw)

        dx = [fx - x for x in self.desired_traj_x]
        dy = [fy - y for y in self.desired_traj_y]

        # self.target_index = int(np.argmin(np.hypot(dx, dy)))
        self.target_index = Functions.project_point_on_path([fx, fy], np.vstack([self.desired_traj_x, self.desired_traj_y]).T,
                                                            exclude_points_behind_vehicle=False)
        assert self.target_index >= 0 & self.target_index < len(self.desired_traj_x)

        front_axle_vec_rot_90 = np.array([[math.cos(vehicle.psi - np.pi / 2.0)],
                                          [math.sin(vehicle.psi - np.pi / 2.0)]])

        vec_target_2_front = np.array([[dx[self.target_index]],
                                       [dy[self.target_index]]])

        self.ef = np.dot(vec_target_2_front.T, front_axle_vec_rot_90).squeeze()

        if self.num_points_to_average_ref_heading != 1:
            num_path_points = self.desired_traj_psi.shape()
            index_range = range(self.target_index,
                                min(self.target_index + self.num_points_to_average_ref_heading,
                                    num_path_points))
            self.psi_traj = np.mean(self.desired_traj_psi[index_range])
        else:
            self.psi_traj = self.desired_traj_psi[self.target_index]
        self.psi_e = fold_angles(self.psi_traj - vehicle.psi)

    @dispatch(VehicleDynamicModel)
    def calc_theta_e_and_ef(self, vehicle: VehicleDynamicModel):
        """
        calc theta_e and ef.
        theta_e = theta_car - theta_path
        ef = lateral distance in CG

        :param vehicle: current information of vehicle
        :return: theta_e and ef
        """

        fx = vehicle.x  # + car_params.WB * math.cos(vehicle.yaw)
        fy = vehicle.y  # + car_params.WB * math.sin(vehicle.yaw)

        dx = [fx - x for x in self.desired_traj_x]
        dy = [fy - y for y in self.desired_traj_y]

        # self.target_index = int(np.argmin(np.hypot(dx, dy)))
        self.target_index = Functions.project_point_on_path([fx, fy],
                                                            np.vstack([self.desired_traj_x, self.desired_traj_y]).T,
                                                            exclude_points_behind_vehicle=False)
        assert self.target_index >= 0 & self.target_index < len(self.desired_traj_x)

        front_axle_vec_rot_90 = np.array([[math.cos(vehicle.psi - np.pi / 2.0)],
                                          [math.sin(vehicle.psi - np.pi / 2.0)]])

        vec_target_2_front = np.array([[dx[self.target_index]],
                                       [dy[self.target_index]]])

        self.ef = np.dot(vec_target_2_front.T, front_axle_vec_rot_90).squeeze()

        if self.num_points_to_average_ref_heading != 1:
            num_path_points = self.desired_traj_psi.shape()
            index_range = range(self.target_index,
                                min(self.target_index + self.num_points_to_average_ref_heading,
                                    num_path_points))
            self.psi_traj = np.mean(self.desired_traj_psi[index_range])
        else:
            self.psi_traj = self.desired_traj_psi[self.target_index]
        self.psi_e = fold_angles(self.psi_traj - vehicle.psi)
    @dispatch(list)
    def calc_theta_e_and_ef(self, vehicle_state):
        """
        calc theta_e and ef.
        theta_e = theta_car - theta_path
        ef = lateral distance in CG

        :param vehicle_state: np.array(size = 4), [x, y, psi, v]
        :return: theta_e and ef
        """

        fx = vehicle_state[0]
        fy = vehicle_state[1]

        dx = [fx - x for x in self.desired_traj_x]
        dy = [fy - y for y in self.desired_traj_y]

        # self.target_index = int(np.argmin(np.hypot(dx, dy)))
        self.target_index = Functions.project_point_on_path([fx, fy],
                                                            np.vstack([self.desired_traj_x, self.desired_traj_y]).T,
                                                            exclude_points_behind_vehicle=True,
                                                            psi=vehicle_state[2])
        assert self.target_index >= 0 & self.target_index < len(self.desired_traj_x)

        front_axle_vec_rot_90 = np.array([[math.cos(vehicle_state[2] - np.pi / 2.0)],
                                          [math.sin(vehicle_state[2] - np.pi / 2.0)]])

        vec_target_2_front = np.array([[dx[self.target_index]],
                                       [dy[self.target_index]]])

        self.ef = np.dot(vec_target_2_front.T, front_axle_vec_rot_90).squeeze()

        if self.num_points_to_average_ref_heading != 1:
            num_path_points = len(self.desired_traj_psi)
            index_range = list(range(self.target_index,
                                min(self.target_index + self.num_points_to_average_ref_heading,
                                    num_path_points)))
            self.psi_traj = np.mean(Functions.continuous_angle(
                                np.array(self.desired_traj_psi)[index_range]))
        else:
            self.psi_traj = self.desired_traj_psi[self.target_index]
        self.psi_e = fold_angles(self.psi_traj - vehicle_state[2])
        if False:
            plt.figure()
            plt.scatter(self.desired_traj_x, self.desired_traj_y, s=5, color='black')
            Functions.Arrow(x=vehicle_state[0], y=vehicle_state[1], theta=vehicle_state[2], L = 2, c='red')
            Functions.Arrow(x=vehicle_state[0], y=vehicle_state[1], theta=self.psi_traj, L=2, c='blue')
            plt.grid(True)
            plt.show()

    @dispatch(VehicleKinemaicModel)
    def calc_steering_command(self, vehicle: VehicleKinemaicModel):
        self.calc_theta_e_and_ef(vehicle)
        self.delta = self.psi_e + math.atan2(self.Ks * self.ef, vehicle.vx + self.eps)

    @dispatch(VehicleDynamicModel)
    def calc_steering_command(self, vehicle: VehicleDynamicModel):
        self.calc_theta_e_and_ef(vehicle)
        self.delta = self.psi_e + math.atan2(self.Ks * self.ef, vehicle.vx + self.eps)

    @dispatch(list)
    def calc_steering_command(self, vehicle_state):
        self.calc_theta_e_and_ef(vehicle_state)
        self.delta = self.psi_e + math.atan2(self.Ks * self.ef, vehicle_state[3] + self.eps)

class MPC_params:
    # System config
    NX = 4  # state vector: z = [x, y, v, phi]
    NU = 2  # input vector: u = [acceleration, steer]
    T = 25  # finite time horizon length

    # MPC config
    Q = np.diag([1.0, 1.0, 1.0, 1.0])  # penalty for states
    Qf = np.diag([1.0, 1.0, 1.0, 1.0])  # penalty for end state
    R = np.diag([0.01, 0.1])  # penalty for inputs
    Rd = np.diag([0.1, 1.0])  # penalty for change of inputs

    dist_stop = 1.5  # stop permitted when dist to goal < dist_stop
    speed_stop = 0.5 / 3.6  # stop permitted when speed < speed_stop
    time_max = 500.0  # max simulation time
    iter_max = 5  # max iteration
    target_speed = 10.0 / 3.6  # target speed
    N_IND = 10  # search index number
    dt = 0.1  # time step
    d_dist = 1.0  # dist step
    d_a_res = 0.01  # threshold for stopping iteration
    d_delta_res = 0.01  # threshold for stopping iteration

    # vehicle config
    RF = 3.3  # [m] distance from rear to vehicle front end of vehicle
    RB = 0.8  # [m] distance from rear to vehicle back end of vehicle
    W = 1.8  # [m] width of vehicle
    WD = 1.5  # [m] distance between left-right wheels
    WB = 2.7  # [m] Wheel base
    TR = 0.44  # [m] Tyre radius
    TW = 0.7  # [m] Tyre width

    steer_max = np.deg2rad(45.0)  # max steering angle [rad]
    steer_change_max = np.deg2rad(30.0)  # maximum steering speed [rad/s]
    speed_max = 55.0 / 3.6  # maximum speed [m/s]
    speed_min = -20.0 / 3.6  # minimum speed [m/s]
    acceleration_max = 1.0  # maximum acceleration [m/s2]
# class Node:
#     def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, direct=1.0):
#         self.x = x
#         self.y = y
#         self.yaw = yaw
#         self.v = v
#         self.direct = direct
#
#     def update(self, a, delta, direct):
#         delta = self.limit_input_delta(delta)
#         self.x += self.v * math.cos(self.yaw) * self.MPC_params.dt
#         self.y += self.v * math.sin(self.yaw) * self.MPC_params.dt
#         self.yaw += self.v / self.MPC_params.WB * math.tan(delta) * self.MPC_params.dt
#         self.direct = direct
#         self.v += self.direct * a * self.MPC_params.dt
#         self.v = self.limit_speed(self.v)
#
#     @staticmethod
#     def limit_input_delta(delta):
#         if delta >= MPC_params.steer_max:
#             return MPC_params.steer_max
#
#         if delta <= -MPC_params.steer_max:
#             return -MPC_params.steer_max
#
#         return delta
#
#     @staticmethod
#     def limit_speed(v):
#         if v >= MPC_params.speed_max:
#             return MPC_params.speed_max
#
#         if v <= MPC_params.speed_min:
#             return MPC_params.speed_min
#
#         return v
# class P:
#     # System config
#     NX = 4  # state vector: z = [x, y, v, phi]
#     NU = 2  # input vector: u = [acceleration, steer]
#     T = 6  # finite time horizon length
#
#     # MPC config
#     Q = np.diag([1.0, 1.0, 1.0, 1.0])  # penalty for states
#     Qf = np.diag([1.0, 1.0, 1.0, 1.0])  # penalty for end state
#     R = np.diag([0.01, 0.1])  # penalty for inputs
#     Rd = np.diag([0.01, 0.1])  # penalty for change of inputs
#
#     dist_stop = 1.5  # stop permitted when dist to goal < dist_stop
#     speed_stop = 0.5 / 3.6  # stop permitted when speed < speed_stop
#     time_max = 7.0  # max simulation time
#     iter_max = 5  # max iteration
#     target_speed = 10.0 / 3.6  # target speed
#     N_IND = 10  # search index number
#     dt = 0.2  # time step
#     d_dist = 1.0  # dist step
#     du_res = 0.1  # threshold for stopping iteration
#
#     # vehicle config
#     RF = 3.3  # [m] distance from rear to vehicle front end of vehicle
#     RB = 0.8  # [m] distance from rear to vehicle back end of vehicle
#     W = 2.4  # [m] width of vehicle
#     WD = 0.7 * W  # [m] distance between left-right wheels
#     WB = 2.5  # [m] Wheel base
#     TR = 0.44  # [m] Tyre radius
#     TW = 0.7  # [m] Tyre width
#
#     steer_max = np.deg2rad(45.0)  # max steering angle [rad]
#     steer_change_max = np.deg2rad(30.0)  # maximum steering speed [rad/s]
#     speed_max = 55.0 / 3.6  # maximum speed [m/s]
#     speed_min = -20.0 / 3.6  # minimum speed [m/s]
#     acceleration_max = 1.0  # maximum acceleration [m/s2]

class MPC:
    def __init__(self, vehicle_params, simulation_params, ref_path_points, ref_path_heading, speed_profile):
        # self.car_model = VehicleKinemaicModel(vehicle_params=vehicle_params, simulation_params=simulation_params,
        #                                steering_uncertainty_factor=1.0, lr_uncertainty_factor=1.0, WB_uncertainty_factor=1.0)
        self.vehicle_params = vehicle_params
        self.simulation_params = simulation_params
        self.MPC_params = MPC_params
        self.MPC_params.WB = vehicle_params['WB']
        self.MPC_params.steer_max = vehicle_params['MAX_STEER']# in radians
        self.MPC_params.d_dist = simulation_params["path_spacing"]
        self.ref_path_points = ref_path_points
        self.ref_path_heading = ref_path_heading
        self.ref_path_heading = ref_path_heading
        self.speed_profile = speed_profile
        self.delta_old = [0.0] * self.MPC_params.T
        self.a_old = [0.0] * self.MPC_params.T
        self.delta_opt = None
        self.a_opt = None
        self.x_opt = None
        self.y_opt = None
        self.yaw_opt = None
        self.v_opt = None
        self.delta_exc = None
        self.a_exc = None
        self.cost_dict = 0.0
    def calc_steering_command(self, vehicle_obj):
        z_ref, target_ind = self.calc_ref_trajectory_in_T_step(vehicle_obj)
        # z0 = [vehicle_obj.x, vehicle_obj.y, vehicle_obj.v, vehicle_obj.yaw]
        z0 = [vehicle_obj.x, vehicle_obj.y, vehicle_obj.vx, vehicle_obj.psi]
        self.linear_mpc_control(z_ref, z0)
        if self.delta_opt is not None:
            self.delta_exc, self.a_exc = self.delta_opt[0], self.a_opt[0]
        # z_ref, target_ind = self.calc_ref_trajectory_in_T_step(vehicle_obj)
        # z0 = [vehicle_obj.x, vehicle_obj.y, vehicle_obj.vx, vehicle_obj.psi]
        # self.linear_mpc_control(z_ref, z0)
        # if self.delta_opt is not None:
        #     self.delta_exc, self.a_exc = self.delta_opt[0], self.a_opt[0]
    def linear_mpc_control(self, z_ref, z0):
        """
        linear mpc controller
        :param z_ref: reference trajectory in T steps
        :param z0: initial state vector
        :param a_old: acceleration of T steps of last time
        :param delta_old: delta of T steps of last time
        :return: acceleration and delta strategy based on current information
        """

        x, y, yaw, v = None, None, None, None

        for k in range(self.MPC_params.iter_max):
            z_bar = self.predict_states_in_T_step(z0, z_ref)
            a_rec, delta_rec = self.a_old[:], self.delta_old[:]

            self.a_old, self.delta_old, x, y, yaw, v , self.cost_dict = MPC.solve_linear_mpc(z_ref, z_bar, z0, delta_rec, self.MPC_params)

            du_a_max = max([abs(ia - iao) for ia, iao in zip(self.a_old, a_rec)])
            du_d_max = max([abs(ide - ido) for ide, ido in zip(self.delta_old, delta_rec)])
            if False:
                print("MPC iteration = " + str(k) + ", du_a_max = " + str(du_a_max) + ", du_d_max = " + str(du_d_max) +
                      ", cost = " + str(self.cost_dict["overall_cost"]))
            if du_a_max < self.MPC_params.d_a_res and du_d_max < self.MPC_params.d_delta_res:
                break
        self.a_opt = self.a_old
        self.delta_opt = self.delta_old
        self.x_opt = x
        self.y_opt = y
        self.yaw_opt = yaw
        self.v_opt = v
    def calc_ref_trajectory_in_T_step(self, car_model):
        """
        calc referent trajectory in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param car_model: current information
        :param ref_path: reference path: [x, y, yaw]
        :param sp: speed profile (designed speed strategy)
        :return: reference trajectory
        """

        z_ref = np.zeros((self.MPC_params.NX, self.MPC_params.T + 1))
        length = len(self.ref_path_heading)

        ind = Functions.project_point_on_path([car_model.x, car_model.y], self.ref_path_points,
                                              exclude_points_behind_vehicle=True, psi=car_model.psi)
        # ind = Functions.project_point_on_path([car_model.x, car_model.y], self.ref_path_points,
        #                                       exclude_points_behind_vehicle=False, psi=car_model.yaw)
        # ind, _ = ref_path.nearest_index(node)

        z_ref[0, 0] = self.ref_path_points[ind, 0]
        z_ref[1, 0] = self.ref_path_points[ind, 1]
        z_ref[2, 0] = self.speed_profile[ind]
        z_ref[3, 0] = self.ref_path_heading[ind]

        dist_move = 0.0

        for i in range(1, self.MPC_params.T + 1):
            dist_move += abs(car_model.vx) * self.MPC_params.dt
            # dist_move += abs(car_model.v) * self.MPC_params.dt
            # the dt is for the MPC prediction not the simulation therefore can differ.
            ind_move = int(round(dist_move / self.MPC_params.d_dist))
            index = min(ind + ind_move, length - 1)

            z_ref[0, i] = self.ref_path_points[index, 0]
            z_ref[1, i] = self.ref_path_points[index, 1]
            z_ref[2, i] = self.speed_profile[index]
            z_ref[3, i] = self.ref_path_heading[index]

        return z_ref, ind
    def predict_states_in_T_step(self, z0, z_ref):
        """
        given the current state, using the acceleration and delta strategy of last time,
        predict the states of vehicle in T steps.
        :param z0: initial state
        :param a: acceleration strategy of last time
        :param delta: delta strategy of last time
        :param z_ref: reference trajectory
        :return: predict states in T steps (z_bar, used for calc linear motion model)
        """

        z_bar = z_ref * 0.0

        for i in range(self.MPC_params.NX):
            z_bar[i, 0] = z0[i]
        car_model = VehicleKinemaicModel(vehicle_params=self.vehicle_params,
                                         simulation_params=self.simulation_params,
                                         x=z0[0], y=z0[1], vx=z0[2], psi=z0[3],
                                         steering_uncertainty_factor=1.0, lr_uncertainty_factor=1.0,
                                         WB_uncertainty_factor=1.0)
        car_model.simulation_params["dt"] = self.MPC_params.dt
        for ai, di, i in zip(self.a_old, self.delta_old, range(1, self.MPC_params.T + 1)):
            car_model.update(ai, di)
            z_bar[0, i] = car_model.x
            z_bar[1, i] = car_model.y
            z_bar[2, i] = car_model.vx
            z_bar[3, i] = car_model.psi

        return z_bar
    @staticmethod
    def calc_linear_discrete_model(v, phi, delta, P):
        """
        calc linear and discrete time dynamic model.
        :param v: speed: v_bar
        :param phi: angle of vehicle: phi_bar
        :param delta: steering angle: delta_bar
        :return: A, B, C
        """

        A = np.array([[1.0, 0.0, P.dt * math.cos(phi + delta) , - P.dt * v * math.sin(phi + delta)],
                      [0.0, 1.0, P.dt * math.sin(phi + delta) ,   P.dt * v * math.cos(phi + delta)],
                      [0.0, 0.0, 1.0                          , 0.0                               ],
                      [0.0, 0.0, P.dt * math.sin(delta) / P.WB,1.0                               ]])

        B = np.array([[0.0 , - P.dt * v * math.sin(phi + delta)],
                      [0.0 ,   P.dt * v * math.cos(phi + delta)],
                      [P.dt, 0.0                               ],
                      [0.0 , P.dt * v  * math.cos(delta) / P.WB]])

        C = np.array([P.dt * v * math.sin(phi + delta) * (phi + delta),
                      - P.dt * v * math.cos(phi + delta) * (phi + delta),
                      0.0,
                      -P.dt * v * delta * math.cos(delta) / P.WB])

        return A, B, C
    @staticmethod
    def calc_linear_discrete_model_org(v, phi, delta, P):
        """
        calc linear and discrete time dynamic model.
        :param v: speed: v_bar
        :param phi: angle of vehicle: phi_bar
        :param delta: steering angle: delta_bar
        :return: A, B, C
        """

        A = np.array([[1.0, 0.0, P.dt * math.cos(phi), - P.dt * v * math.sin(phi)],
                      [0.0, 1.0, P.dt * math.sin(phi), P.dt * v * math.cos(phi)],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, P.dt * math.tan(delta) / P.WB, 1.0]])

        B = np.array([[0.0, 0.0],
                      [0.0, 0.0],
                      [P.dt, 0.0],
                      [0.0, P.dt * v / (P.WB * math.cos(delta) ** 2)]])

        C = np.array([P.dt * v * math.sin(phi) * phi,
                      -P.dt * v * math.cos(phi) * phi,
                      0.0,
                      -P.dt * v * delta / (P.WB * math.cos(delta) ** 2)])

        return A, B, C
    @staticmethod
    def solve_linear_mpc(z_ref, z_bar, z0, d_bar, P):
        """
        solve the quadratic optimization problem using cvxpy, solver: OSQP
        :param z_ref: reference trajectory (desired trajectory: [x, y, v, yaw])
        :param z_bar: predicted states in T steps
        :param z0: initial state
        :param d_bar: delta_bar
        :return: optimal acceleration and steering strategy
        """

        z = cvxpy.Variable((P.NX, P.T + 1))
        u = cvxpy.Variable((P.NU, P.T))

        cost = 0.0
        actuation_cost = 0.0
        a_cost = 0.0
        a_change_cost = 0.0
        delta_cost = 0.0
        delta_change_cost = 0.0
        x_cost = 0.0
        y_cost = 0.0
        psi_cost = 0.0
        v_cost = 0.0
        state_error_cost = 0.0
        end_state_cost = 0.0
        actuation_change_cost = 0.0
        constrains = []

        for t in range(P.T):
            cost += cvxpy.quad_form(u[:, t], P.R)
            actuation_cost += cvxpy.quad_form(u[:, t], P.R)
            a_cost += (u[0, t] ** 2) * P.R[0,0]
            delta_cost += (u[1, t] ** 2) * P.R[1, 1]
            cost += cvxpy.quad_form(z_ref[:, t] - z[:, t], P.Q)
            state_error_cost += cvxpy.quad_form(z_ref[:, t] - z[:, t], P.Q)
            x_cost += ((z_ref[0, t] - z[0, t]) ** 2) * P.Q[0, 0]
            y_cost += ((z_ref[1, t] - z[1, t]) ** 2) * P.Q[1, 1]
            v_cost += ((z_ref[2, t] - z[2, t]) ** 2) * P.Q[2, 2]
            psi_cost += ((z_ref[3, t] - z[3, t]) ** 2) * P.Q[3, 3]

            A, B, C = MPC.calc_linear_discrete_model(z_bar[2, t], z_bar[3, t], d_bar[t], P)

            constrains += [z[:, t + 1] == A @ z[:, t] + B @ u[:, t] + C]

            if t < P.T - 1:
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], P.Rd)
                actuation_change_cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], P.Rd)
                a_change_cost += ((u[0, t + 1] - u[0, t]) ** 2) *  P.Rd[0, 0]
                delta_change_cost += ((u[1, t + 1] - u[1, t]) ** 2) * P.Rd[1, 1]
                constrains += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= P.steer_change_max * P.dt]

        cost += cvxpy.quad_form(z_ref[:, P.T] - z[:, P.T], P.Qf)
        end_state_cost += cvxpy.quad_form(z_ref[:, P.T] - z[:, P.T], P.Qf)

        constrains += [z[:, 0] == z0]
        constrains += [z[2, :] <= P.speed_max]
        constrains += [z[2, :] >= P.speed_min]
        constrains += [cvxpy.abs(u[0, :]) <= P.acceleration_max]
        constrains += [cvxpy.abs(u[1, :]) <= P.steer_max]

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constrains)
        prob.solve(solver=cvxpy.OSQP)

        a, delta, x, y, yaw, v = None, None, None, None, None, None

        if prob.status == cvxpy.OPTIMAL or \
                prob.status == cvxpy.OPTIMAL_INACCURATE:
            x = z.value[0, :]
            y = z.value[1, :]
            v = z.value[2, :]
            yaw = z.value[3, :]
            a = u.value[0, :]
            delta = u.value[1, :]
        else:
            print("Cannot solve linear mpc!")
        cost_dict = {"overall_cost": cost.value,
                     "actuation_cost":actuation_cost.value,
                     "a_cost": a_cost.value,
                     "delta_cost": delta_cost.value,
                     "state_error_cost":state_error_cost.value,
                     "end_state_cost":end_state_cost.value,
                     "actuation_change_cost":actuation_change_cost.value,
                     "a_change_cost": a_change_cost.value,
                     "delta_change_cost": delta_change_cost.value,
                     "x_cost": x_cost.value,
                     "y_cost": y_cost.value,
                     "v_cost": v_cost.value,
                     "psi_cost": psi_cost.value
        }
        return a, delta, x, y, yaw, v, cost_dict
    # def solve_linear_mpc(self, z_ref, z_bar, z0, d_bar):
    #     """
    #     solve the quadratic optimization problem using cvxpy, solver: OSQP
    #     :param z_ref: reference trajectory (desired trajectory: [x, y, v, yaw])
    #     :param z_bar: predicted states in T steps
    #     :param z0: initial state
    #     :param d_bar: delta_bar
    #     :return: optimal acceleration and steering strategy
    #     """
    #
    #     z = cvxpy.Variable((P.NX, P.T + 1))
    #     u = cvxpy.Variable((P.NU, P.T))
    #
    #     cost = 0.0
    #     constrains = []
    #
    #     for t in range(P.T):
    #         cost += cvxpy.quad_form(u[:, t], P.R)
    #         cost += cvxpy.quad_form(z_ref[:, t] - z[:, t], P.Q)
    #
    #         A, B, C = self.calc_linear_discrete_model(z_bar[2, t], z_bar[3, t], d_bar[t])
    #
    #         constrains += [z[:, t + 1] == A @ z[:, t] + B @ u[:, t] + C]
    #
    #         if t < P.T - 1:
    #             cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], P.Rd)
    #             constrains += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= P.steer_change_max * P.dt]
    #
    #     cost += cvxpy.quad_form(z_ref[:, P.T] - z[:, P.T], P.Qf)
    #
    #     constrains += [z[:, 0] == z0]
    #     constrains += [z[2, :] <= P.speed_max]
    #     constrains += [z[2, :] >= P.speed_min]
    #     constrains += [cvxpy.abs(u[0, :]) <= P.acceleration_max]
    #     constrains += [cvxpy.abs(u[1, :]) <= P.steer_max]
    #
    #     prob = cvxpy.Problem(cvxpy.Minimize(cost), constrains)
    #     prob.solve(solver=cvxpy.OSQP)
    #
    #     a, delta, x, y, yaw, v = None, None, None, None, None, None
    #
    #     if prob.status == cvxpy.OPTIMAL or \
    #             prob.status == cvxpy.OPTIMAL_INACCURATE:
    #         x = z.value[0, :]
    #         y = z.value[1, :]
    #         v = z.value[2, :]
    #         yaw = z.value[3, :]
    #         a = u.value[0, :]
    #         delta = u.value[1, :]
    #     else:
    #         print("Cannot solve linear mpc!")
    #
    #     return a, delta, x, y, yaw, v
    # def solve_linear_mpc(self, z_ref, z_bar, z0, d_bar):
    #     """
    #     solve the quadratic optimization problem using cvxpy, solver: OSQP
    #     :param z_ref: reference trajectory (desired trajectory: [x, y, v, yaw])
    #     :param z_bar: predicted states in T steps
    #     :param z0: initial state
    #     :param d_bar: delta_bar
    #     :return: optimal acceleration and steering strategy
    #     """
    #     z = cvxpy.Variable((self.MPC_params.NX, self.MPC_params.T + 1))
    #     u = cvxpy.Variable((self.MPC_params.NU, self.MPC_params.T))
    #
    #     cost = 0.0
    #     constrains = []
    #     for t in range(self.MPC_params.T):
    #         cost += cvxpy.quad_form(u[:, t], self.MPC_params.R)
    #         cost += cvxpy.quad_form(z_ref[:, t] - z[:, t], self.MPC_params.Q)
    #
    #         A, B, C = self.calc_linear_discrete_model(z_bar[2, t], z_bar[3, t], d_bar[t])
    #
    #         constrains += [z[:, t + 1] == A @ z[:, t] + B @ u[:, t] + C]
    #
    #         if t < self.MPC_params.T - 1:
    #             cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], self.MPC_params.Rd)
    #             constrains += [
    #                 cvxpy.abs(u[1, t + 1] - u[1, t]) <= self.MPC_params.steer_change_max * self.MPC_params.dt]
    #
    #     cost += cvxpy.quad_form(z_ref[:, self.MPC_params.T] - z[:, self.MPC_params.T], self.MPC_params.Qf)
    #     constrains += [z[:, 0] == z0]
    #     constrains += [z[2, :] <= self.MPC_params.speed_max]
    #     constrains += [z[2, :] >= self.MPC_params.speed_min]
    #     constrains += [cvxpy.abs(u[0, :]) <= self.MPC_params.acceleration_max]
    #     constrains += [cvxpy.abs(u[1, :]) <= self.MPC_params.steer_max]
    #
    #     prob = cvxpy.Problem(cvxpy.Minimize(cost), constrains)
    #     prob.solve(solver=cvxpy.OSQP)
    #     # z = cvxpy.Variable((self.MPC_params.NX, self.MPC_params.T + 1))
    #     # u = cvxpy.Variable((self.MPC_params.NU, self.MPC_params.T))
    #     #
    #     # overall_cost = 0.0
    #     # actuation_cost = 0.0
    #     # actuation_diff_cost = 0.0
    #     # output_cost = 0.0
    #     # cost = 0.0
    #     # constrains = []
    #     #
    #     # for t in range(self.MPC_params.T):
    #     #     cost += cvxpy.quad_form(u[:, t], self.MPC_params.R)
    #     #     cost += cvxpy.quad_form(z_ref[:, t] - z[:, t], self.MPC_params.Q)
    #     #
    #     #     A, B, C = self.calc_linear_discrete_model(z_bar[2, t], z_bar[3, t], d_bar[t])
    #     #
    #     #     constrains += [z[:, t + 1] == A @ z[:, t] + B @ u[:, t] + C]
    #     #
    #     #     if t < self.MPC_params.T - 1:
    #     #         cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], self.MPC_params.Rd)
    #     #         constrains += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= self.MPC_params.steer_change_max * self.MPC_params.dt]
    #     #
    #     # cost += cvxpy.quad_form(z_ref[:, self.MPC_params.T] - z[:, self.MPC_params.T], self.MPC_params.Qf)
    #     # constrains += [z[:, 0] == z0]
    #     # constrains += [z[2, :] <= self.MPC_params.speed_max]
    #     # constrains += [z[2, :] >= self.MPC_params.speed_min]
    #     # constrains += [cvxpy.abs(u[0, :]) <= self.MPC_params.acceleration_max]
    #     # constrains += [cvxpy.abs(u[1, :]) <= self.MPC_params.steer_max]
    #     #
    #     # prob = cvxpy.Problem(cvxpy.Minimize(cost), constrains)
    #     # prob.solve(solver=cvxpy.OSQP)
    #
    #     a, delta, x, y, yaw, v = None, None, None, None, None, None
    #
    #     if prob.status == cvxpy.OPTIMAL or \
    #             prob.status == cvxpy.OPTIMAL_INACCURATE:
    #         x = z.value[0, :]
    #         y = z.value[1, :]
    #         v = z.value[2, :]
    #         yaw = z.value[3, :]
    #         a = u.value[0, :]
    #         delta = u.value[1, :]
    #     else:
    #         print("Cannot solve linear mpc!")
    #
    #     return a, delta, x, y, yaw, v
    # def calc_linear_discrete_model(self, v, phi, delta):
    #     """
    #     calc linear and discrete time dynamic model.
    #     :param v: speed: v_bar
    #     :param phi: angle of vehicle: phi_bar
    #     :param delta: steering angle: delta_bar
    #     :return: A, B, C
    #     """
    #
    #     A = np.array([[1.0, 0.0, self.MPC_params.dt * math.cos(phi), - self.MPC_params.dt * v * math.sin(phi)],
    #                   [0.0, 1.0, self.MPC_params.dt * math.sin(phi), self.MPC_params.dt * v * math.cos(phi)],
    #                   [0.0, 0.0, 1.0, 0.0],
    #                   [0.0, 0.0, self.MPC_params.dt * math.tan(delta) / self.MPC_params.WB, 1.0]])
    #
    #     B = np.array([[0.0, 0.0],
    #                   [0.0, 0.0],
    #                   [self.MPC_params.dt, 0.0],
    #                   [0.0, self.MPC_params.dt * v / (self.MPC_params.WB * math.cos(delta) ** 2)]])
    #
    #     C = np.array([self.MPC_params.dt * v * math.sin(phi) * phi,
    #                   -self.MPC_params.dt * v * math.cos(phi) * phi,
    #                   0.0,
    #                   -self.MPC_params.dt * v * delta / (self.MPC_params.WB * math.cos(delta) ** 2)])
    #
    #     return A, B, C

class RearWheelFeedbackController:

    def __init__(self, K_theta, K_e, spline_x, spline_y, spline_psi, spline_curvature):
        self.__dict__.update(locals())  # initializes the class attributes with inputs
        self.target_index = None
        self.er = None
        self.psi_traj = None
        self.delta = None
        self.psi_e = None

    def calc_theta_e_and_er(self, vehicle: VehicleKinemaicModel):
        """
        calc theta_e and er.
        theta_e = theta_car - theta_path
        er = lateral distance in frenet frame

        :param node: current information of vehicle
        :return: theta_e and er
        """
        assert vehicle.simulation_params['ego_frame_placement'] == 'rear_axle'

        fx = vehicle.x
        fy = vehicle.y

        dx = [fx - x for x in self.spline_x]
        dy = [fy - y for y in self.spline_y]

        self.target_index = int(np.argmin(np.hypot(dx, dy)))
        assert self.target_index >= 0 & self.target_index < len(self.spline_x)

        k = self.spline_curvature[self.target_index]
        yaw = self.spline_psi[self.target_index]

        rear_axle_vec_rot_90 = np.array([[math.cos(vehicle.psi + math.pi / 2.0)],
                                         [math.sin(vehicle.psi + math.pi / 2.0)]])

        vec_target_2_rear = np.array([[vehicle.x - self.spline_x[self.target_index]],
                                      [vehicle.y - self.spline_y[self.target_index]]])

        self.er = np.dot(vec_target_2_rear.T, rear_axle_vec_rot_90).squeeze()
        self.psi_traj = self.spline_psi[self.target_index]
        self.psi_e = fold_angles(vehicle.psi - self.psi_traj)

    def calc_steering_command(self, vehicle: VehicleKinemaicModel):
        """ implementation of control law from
         'A Survey of Motion Planning and Control Techniques for Self-Driving Urban Vehicles'
         page 16 (48) equation (20)
        """
        self.calc_theta_e_and_er(vehicle)
        vr = vehicle.vx
        curvature = self.spline_curvature[self.target_index]
        omega = vr * curvature * math.cos(self.psi_e) / epsilon_limit(1.0 - curvature * self.er) - \
                    self.K_theta * abs(vr) * self.psi_e - self.K_e * vr * math.sin(self.psi_e) * self.er / epsilon_limit(self.psi_e)
        self.delta = math.atan2(vehicle.vehicle_params['WB'] * omega, vr)


class Trip:

    def __init__(self, path):
        self.gps = pd.read_csv(join(path, 'gps.csv'))
        self.speed = pd.read_csv(join(path, 'speed.csv'))
        self.steering = pd.read_csv(join(path, 'steering.csv'))
        self.imu = pd.read_csv(join(path, 'imu.csv'))
        time_vectors = [np.array(self.steering.time_stamp), np.array(self.gps.time_stamp),
                        np.array(self.speed.time_stamp),
                        np.array(self.imu.time_stamp)]
        self.common_time = Functions.allign_time_vectors(time_vectors)
        self.steering_interp = np.interp(self.common_time, self.steering.time_stamp, self.steering.data_value)
        self.speed_interp = np.interp(self.common_time, self.speed.time_stamp, self.speed.data_value) / 3.6
        self.lat_interp = np.interp(self.common_time, self.gps.time_stamp, self.gps.latitude)
        self.lon_interp = np.interp(self.common_time, self.gps.time_stamp, self.gps.longitude)
        self.alt_interp = np.interp(self.common_time, self.gps.time_stamp, self.gps.height)
        self.yaw_interp = np.interp(self.common_time, self.imu.time_stamp, self.imu.yaw) * np.pi / 180
        self.N, self.E, self.D = Functions.LLA2NED(lat=self.lat_interp, long=self.lon_interp, alt=self.alt_interp)
        self.W = -self.E
        self.common_time -= self.common_time[0]
        # spline_x_downsample, spline_y_downsample, _, _, _ = cs.calc_spline_course(self.N, self.W, ds=10.0)
        # spline_x, spline_y, spline_psi, _, _ = cs.calc_spline_course(spline_x_downsample, spline_y_downsample, ds=1)
        # pseudo_t = np.linspace(start=self.common_time[0], stop=self.common_time[-1], num=self.common_time.shape[0])
        # self.N_smooth = np.interp(self.common_time, pseudo_t, spline_x)
        # self.W_smooth = np.interp(self.common_time, pseudo_t, spline_y)
        # self.psi_smooth = np.interp(self.common_time, pseudo_t, spline_psi)

    def plot_lat_lon(self):
        plt.figure()
        plt.scatter(self.gps.latitude,self.gps.longitude)
        plt.grid(True)
    def plot_heading(self):
        self.yaw_interp = np.interp(self.common_time, self.imu.time_stamp, self.imu.yaw) * np.pi / 180
        plt.figure()
        plt.scatter(self.imu.time_stamp, self.imu.yaw)
        plt.xlabel("imu.time_stamp"),plt.ylabel("imu.yaw [deg]")
        plt.grid(True)
    def plot_steering_velocity(self):
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(self.steering.time_stamp - self.steering.time_stamp[0],self.steering.data_value)
        plt.grid(True), plt.ylabel('steering [deg]')
        plt.subplot(2,1,2)
        plt.plot(self.speed.time_stamp - self.speed.time_stamp[0],self.speed.data_value)
        plt.grid(True), plt.xlabel('time [sec]'), plt.ylabel('speed [KPH]')
    def segment_trip(self,t_start, t_end):
        start_index = np.argmin(abs(self.common_time - t_start))
        end_index = np.argmin(abs(self.common_time - t_end))
        self.common_time = self.common_time[start_index:end_index]
        self.steering_interp = self.steering_interp[start_index:end_index]
        self.speed_interp = self.speed_interp[start_index:end_index]
        self.lat_interp = self.lat_interp[start_index:end_index]
        self.lon_interp = self.lon_interp[start_index:end_index]
        self.alt_interp = self.alt_interp[start_index:end_index]
        self.yaw_interp = self.yaw_interp[start_index:end_index]
        self.N = self.N[start_index:end_index]
        self.E = self.E[start_index:end_index]
        self.D = self.D[start_index:end_index]
        self.W = self.W[start_index:end_index]


class ShortTermLocalization:
    def __init__(self):
        self.pos = None # rear axle
        self.update_time = None
        self.psi = None
        self.speed = None
        self.nominal_dt = None
        self.WB = None
        self.front_axle_pos = None
    def update_heading(self, psi):
        self.psi = psi
    def update_speed(self, speed):
        self.speed = speed
    def update_position(self, clock):
        # print('psi = ' + str("%.2f" % self.psi))
        # print('speed = ' + str("%.2f" % self.speed))
        dt = clock - self.update_time
        self.update_time = clock
        dx = np.cos(self.psi) * self.speed * dt
        dy = np.sin(self.psi) * self.speed * dt
        self.pos += np.array([dx,dy])
    def update_front_axle_position(self, clock, delta):
        # self.pos[0] += self.speed * np.cos(self.psi + delta) * self.nominal_dt
        # self.pos[1] += self.speed * np.sin(self.psi + delta) * self.nominal_dt
        dt = clock - self.update_time
        self.update_time = clock
        self.pos[0] += self.speed * np.cos(self.psi + delta) * dt
        self.pos[1] += self.speed * np.sin(self.psi + delta) * dt
    def update_rear_axle_position(self, clock):
        dt = clock - self.update_time
        self.update_time = clock
        self.pos[0] += self.speed * np.cos(self.psi) * dt
        self.pos[1] += self.speed * np.sin(self.psi) * dt
    def calc_front_axle_position(self):
        x = self.pos[0] + self.WB * np.cos(self.psi)
        y = self.pos[1] + self.WB * np.sin(self.psi)
        self.front_axle_pos = np.array([x,y])
    def reset_position(self, clock, state=[0.0, 0.0, 0.0, 0.0]):
        self.update_time = clock
        self.pos = np.array(state[:2])
        self.psi = state[2]
        self.speed = state[3]


class DelayConstant:
    def __init__(self, initial_output=0.0, initial_clock=0.0, time_delay=0.0):
        self.output = initial_output
        self.clock = initial_clock
        self.time_delay = time_delay
        self.value_buffer = []
        self.clock_buffer = []
        """
        test script:
        t = np.arange(0,1,0.01)
        signal = np.sin(2 * np.pi * t)
        delay_obj = DelayConstant(initial_output=- 1.0, initial_clock=0.0, time_delay=0.25)
        delayed_signal = []
        for t_i, u in zip(t, signal):
            delayed_signal.append(delay_obj.update(t_i, u))
        plt.plot(t, signal)
        plt.plot(t, delayed_signal)
        plt.grid(True), plt.xlabel('t')
        plt.show()
        """
    def update(self, clock, input):
        self.value_buffer.append(input)
        self.clock_buffer.append(clock)
        idices_from_before_the_delay = np.argwhere(np.array(self.clock_buffer) <= clock - self.time_delay) # np column vector shaped n X 1
        idices_from_before_the_delay = idices_from_before_the_delay[:, 0]
        if idices_from_before_the_delay.shape[0] > 0:
            for idx in idices_from_before_the_delay[:-1]:
                del self.value_buffer[idx]
                del self.clock_buffer[idx]
            # if len(self.clock_buffer) > 0:
            self.output = self.value_buffer[0]
        return self.output


class Buffer:
    def __init__(self, max_size):
        self.__values = []
        self.__max_size = max_size
    def update(self, new_value):
        self.__values.append(new_value)
        if len(self.__values) > self.__max_size:
            self.__values.pop(0)
    def get_value(self, idx):
        return self.__values[idx]

    def get_first_value(self):
        return self.__values[0]
    def get_last_value(self):
        return self.__values[-1]

    def closest_value(self, target):
        idx = int(np.argmin(abs(target - self.__values)))
        return idx, self.__values[idx]
    def size(self):
        return len(self.__values)


class Delay:
    def __init__(self, max_time_delay=1, nominal_dt=1, default_value=0.0):
        max_buffer_size = max_time_delay / nominal_dt
        self.__stored_values = Buffer(max_buffer_size)
        self.__clock = Buffer(max_buffer_size)
        self.__max_time_delay = max_time_delay
        self.__default_value = default_value
    def update(self, clock, value):
        self.__clock.update(clock)
        self.__stored_values.update(value)

    def get_delayed_value(self, delayed_clock):
        if self.__clock.size() == 0:
            return self.__default_value
        elif self.__clock.get_first_value() > delayed_clock:
            return self.__default_value
        else:
            idx, closest_clock_value = self.__clock.closest_value(delayed_clock)
            return self.__stored_values.get_value(idx)
    def test_function(self):
        t = np.arange(0, 10, 0.001)
        signal = np.sin(2 * np.pi * 0.2 * t) + np.sin(2 * np.pi * 0.3 * t + 0.3)
        nominal_delay = 0.1
        delay_fluctuation = 0.01
        actual_delay = nominal_delay + delay_fluctuation * (np.random.rand(t.shape[0])-0.5)
        delay_obj = Delay(max_time_delay=0.3, nominal_dt=0.001)
        delayed_signal = []
        for ti, si, di in zip(t, signal, actual_delay):
            delay_obj.update(clock=ti, value=si)
            delayed_signal.append(delay_obj.get_delayed_value(ti - di))
        plt.plot(t, signal, t, delayed_signal), plt.grid(True)
        plt.show()


class RateLimiter:
    def __init__(self, maximal_rate, clock, initial_value = 0.0, max_dt = 0.05, eps=1e-6):
        self.maximal_rate = maximal_rate
        self.output = initial_value
        self.max_dt = max_dt
        self.last_update_time = clock
        self.eps = eps
    def initialize(self, initial_value, clock):
        self.output = initial_value
        self.last_update_time = clock
    def update(self, input, clock, reference_value=None):
        if clock - self.last_update_time > self.max_dt:
            if reference_value is None:
                self.initialize(initial_value=self.output, clock=clock)
            else:
                self.initialize(initial_value=reference_value, clock=clock)
            return (self.output, 0)
        dt = clock - self.last_update_time
        self.last_update_time = clock
        dy = input - self.output
        if abs(dy) < self.eps:
            return (self.output, 0) #avoid division by zero
        if abs(dt) < self.eps:
            return (self.output, 0) #avoid division by zero
        sign = dy/abs(dy)
        if abs(dy) / dt < self.maximal_rate:
            delta_output = dy
        else:
            delta_output = sign * self.maximal_rate * dt
        self.output += delta_output
        return (self.output, delta_output / dt)
    def step_test(self):
        def step(clock, initial_value, step_value, step_time ):
            if clock < step_time:
                return initial_value
            else:
                return step_value

        t1 = np.arange(start=0, stop=10, step=0.01)
        t2 = np.arange(start=50, stop=60, step=0.01)
        t = np.concatenate([t1,t2])
        filer = RateLimiter(1.0, clock=t[0], initial_value=0.0, max_dt=0.05)
        u = []
        y = []
        for ti in t:
            if ti<25:
                ui = step(ti, initial_value=0.0, step_value=1.0, step_time=1.0)
            else:
                ui = step(ti, initial_value=0.3, step_value=-2.0, step_time=51.0)
            u.append(ui)
            y.append(filer.update(input=u[-1], clock=ti))
        plt.figure()
        plt.plot(t, u, t, y)
        plt.legend(["u", "y"])
        plt.grid(True)
    def sin_test(self):
        def sin_signal(clock):
            freq = 0.2 * 2 * np.pi
            amplitude = 1
            return amplitude * np.sin(freq * clock)

        t = np.arange(start=0, stop=10, step=0.01)
        filter = RateLimiter(1.0, clock=t[0], initial_value=0.0, max_dt=0.05)
        u = []
        y = []
        for ti in t:
            u.append(sin_signal(ti))
            y.append(filter.update(input=u[-1], clock=ti))
        plt.figure()
        plt.plot(t, u, t, y)
        plt.legend(["u", "y"])
        plt.grid(True)


class RateLimiter2ndOrder:
    def __init__(self, maximal_rate, maximal_acceleration, clock, initial_value = 0.0, initial_derivative = 0.0,  max_dt = 0.05, eps=1e-6):
        self.maximal_rate = maximal_rate
        self.maximal_acceleration = maximal_acceleration
        self.x = initial_value
        self.x_dot = initial_derivative
        self.max_dt = max_dt
        self.t_n = clock
        self.t_n_minus_1 = clock
        self.eps = eps
        self.rate_limiter_object = RateLimiter(maximal_rate = maximal_acceleration,
                                               clock=clock, initial_value = initial_derivative) # to be applied on the 1st derivative
    def initialize(self, initial_value, initial_derivative, clock):
        self.x = initial_value
        self.x_dot = initial_derivative
        self.t_n = clock
        self.rate_limiter_object.initialize(initial_value=initial_derivative, clock=clock)
    def update(self, input, clock, reference_value=None):
        if clock - self.t_n > self.max_dt:
            if reference_value is None:
                self.initialize(initial_value=self.x, initial_derivative=0.0, clock=clock)
            else:
                self.initialize(initial_value=reference_value, initial_derivative=0.0, clock=clock)
            return self.x
        dt = clock - self.t_n
        self.t_n = clock
        dx = input - self.x
        if abs(dx) < self.eps:
            self.rate_limiter_object.update(0, clock)
            return self.x #avoid division by zero
        if abs(dt) < self.eps:
            self.rate_limiter_object.update(0, clock)
            return self.x #avoid division by zero
        sign = dx/abs(dx)
        if abs(dx) / dt > self.maximal_rate:
            dx = sign * self.maximal_rate * dt
        self.x_dot = self.rate_limiter_object.update(dx / dt, clock)
        dx_filtered = self.x_dot * dt
        self.x += dx_filtered
        return self.x
    def step_test(self):
        def step(clock, initial_value, step_value, step_time ):
            if clock < step_time:
                return initial_value
            else:
                return step_value

        t1 = np.arange(start=0, stop=25, step=0.01)
        t2 = np.arange(start=50, stop=60, step=0.01)
        t = np.concatenate([t1,t2])
        filter = RateLimiter2ndOrder(maximal_rate=1.0, maximal_acceleration=2.0, clock=t[0],
                                    initial_value=0.0, initial_derivative=0, max_dt=0.05)
        u = []
        u_dot = [0.0]
        u_dotdot = [0.0, 0.0]
        y = []
        y_dot = [0.0]
        y_dotdot = [0.0, 0.0]
        for i, ti in enumerate(t):
            if ti<25:
                ui = step(ti, initial_value=0.0, step_value=8.0, step_time=1.0)
            else:
                ui = step(ti, initial_value=6, step_value=3.0, step_time=51.0)
            u.append(ui)
            y.append(filter.update(input=u[-1], clock=ti, reference_value=ui))
            if i > 0:
                dt = t[i] - t[i - 1]
                u_dot.append((u[-1] - u[-2]) / dt)
                y_dot.append((y[-1] - y[-2]) / dt)
            if i > 1:
                u_dotdot.append((u_dot[-1] - u_dot[-2]) / dt)
                y_dotdot.append((y_dot[-1] - y_dot[-2]) / dt)
        plt.figure()
        h = plt.subplot(3, 1, 1)
        plt.plot(t, u, t, y)
        plt.legend(["u", "y"]), plt.ylabel('$x$')
        plt.grid(True)
        plt.subplot(3, 1, 2, sharex=h)
        plt.plot(t, u_dot, t, y_dot)
        plt.grid(True), plt.ylabel('$dx/dt$'), plt.ylim(-25,25)
        plt.subplot(3, 1, 3, sharex=h)
        plt.plot(t, u_dotdot, t, y_dotdot)
        plt.grid(True), plt.ylabel('$dx/dt^2$'), plt.ylim(-10,10), plt.xlabel('t[sec]')
    def sin_test(self):
        def sin_signal(clock):
            freq = 0.2 * 2 * np.pi
            amplitude = 1
            return amplitude * np.sin(freq * clock)

        t = np.arange(start=0, stop=10, step=0.01)
        filter = RateLimiter(1.0, clock=t[0], initial_value=0.0, max_dt=0.05)
        u = []
        y = []
        for ti in t:
            u.append(sin_signal(ti))
            y.append(filter.update(input=u[-1], clock=ti))
        plt.figure()
        plt.plot(t, u, t, y)
        plt.legend(["u", "y"])
        plt.grid(True)

class FirstOrderLPF:
    def __init__(self, clock, freq_cutoff, freq_sampling, initial_value, max_dt = 0.05, eps=1e-6):
        """
        implement Tustin integration lowpass filter
        https://github.com/botprof/first-order-low-pass-filter/blob/main/first-order-lpf.ipynb
        freq is inserted in Hz and converted to rad/sec
        """
        self.dt = 1 / (2 * np.pi * freq_sampling)
        self.wc = freq_cutoff * 2 * np.pi # cut off frequency in rad/sec
        self.y_k_minus_1 = initial_value
        self.yk = initial_value
        self.u_k_minus_1 = None
        self.alpha = (2 - self.dt * self.wc) / (2 + self.dt * self.wc)
        self.beta = self.dt * self.wc / (2 + self.dt * self.wc)
        self.max_dt = max_dt
        self.last_update_time = clock
        self.eps = eps
    def initialize_clock(self, clock):
        self.last_update_time = clock
    def update(self, uk, clock):
        if clock - self.last_update_time > self.max_dt:
            self.initialize_clock(clock)
            return self.yk
        self.dt = clock - self.last_update_time
        self.last_update_time = clock
        self.alpha = (2 - self.dt * self.wc) / (2 + self.dt * self.wc)
        self.beta = self.dt * self.wc / (2 + self.dt * self.wc)
        if self.u_k_minus_1 == None:
            self.u_k_minus_1 = uk
        self.yk = self.alpha * self.y_k_minus_1 + self.beta * (uk + self.u_k_minus_1)
        self.y_k_minus_1 = self.yk
        self.u_k_minus_1 = uk
        return self.yk

    def step_test(self):
        def step(clock, initial_value, step_value, step_time):
            if clock < step_time:
                return initial_value
            else:
                return step_value

        t1 = np.arange(start=0, stop=25, step=0.01)
        t2 = np.arange(start=50, stop=60, step=0.01)
        t = np.concatenate([t1, t2])
        filter = FirstOrderLPF(t[0], 1.0, 0.01, 0.0, max_dt = 0.05, eps=1e-6)
        u = []
        u_dot = [0.0]
        u_dotdot = [0.0, 0.0]
        y = []
        y_dot = [0.0]
        y_dotdot = [0.0, 0.0]
        for i, ti in enumerate(t):
            if ti < 25:
                ui = step(ti, initial_value=0.0, step_value=8.0, step_time=1.0)
            else:
                ui = step(ti, initial_value=6, step_value=3.0, step_time=51.0)
            u.append(ui)
            y.append(filter.update(uk=u[-1], clock=ti))
            if i > 0:
                dt = t[i] - t[i - 1]
                u_dot.append((u[-1] - u[-2]) / dt)
                y_dot.append((y[-1] - y[-2]) / dt)
            if i > 1:
                u_dotdot.append((u_dot[-1] - u_dot[-2]) / dt)
                y_dotdot.append((y_dot[-1] - y_dot[-2]) / dt)
        plt.figure()
        h = plt.subplot(3, 1, 1)
        plt.plot(t, u, t, y)
        plt.legend(["u", "y"]), plt.ylabel('$x$')
        plt.grid(True)
        plt.subplot(3, 1, 2, sharex=h)
        plt.plot(t, u_dot, t, y_dot)
        plt.grid(True), plt.ylabel('$dx/dt$'), plt.ylim(-25, 25)
        plt.subplot(3, 1, 3, sharex=h)
        plt.plot(t, u_dotdot, t, y_dotdot)
        plt.grid(True), plt.ylabel('$dx/dt^2$'), plt.ylim(-10, 10), plt.xlabel('t[sec]')

class FirstOrderLPF_WithRateLimit:
    def __init__(self, clock, freq_cutoff, freq_sampling, max_rate,
                 initial_value, max_dt = 0.05, eps=1e-6):
        """
        implement LPF as wc * (uk-yk) -> integrator
        limit the input to the integrator
        """
        self.dt = 1 / (2 * np.pi * freq_sampling)
        self.wc = freq_cutoff * 2 * np.pi # cut off frequency in rad/sec
        self.y_k_minus_1 = initial_value
        self.yk = initial_value
        self.u_k_minus_1 = None
        self.alpha = (2 - self.dt * self.wc) / (2 + self.dt * self.wc)
        self.beta = self.dt * self.wc / (2 + self.dt * self.wc)
        self.max_dt = max_dt
        self.last_update_time = clock
        self.eps = eps
        self.max_rate = max_rate
        self.integrator = initial_value
    def initialize_clock(self, clock):
        self.last_update_time = clock
    def update(self, uk, clock):
        def limit_x(x, x_limit):
            return max(min(x, x_limit), -x_limit)
        if clock - self.last_update_time > self.max_dt:
            self.initialize_clock(clock)
            return self.yk
        self.dt = clock - self.last_update_time
        self.last_update_time = clock
        input_to_integrator = self.wc * (uk - self.y_k_minus_1)
        input_to_integrator_limited = limit_x(input_to_integrator, self.max_rate)
        self.integrator += input_to_integrator_limited * self.dt
        self.y_k_minus_1 = self.integrator.copy()
        return self.integrator

    def step_test(self):
        def step(clock, initial_value, step_value, step_time):
            if clock < step_time:
                return initial_value
            else:
                return step_value

        t1 = np.arange(start=0, stop=25, step=0.01)
        t2 = np.arange(start=50, stop=60, step=0.01)
        t = np.concatenate([t1, t2])
        filter = FirstOrderLPF_WithRateLimit(clock=t[0], freq_cutoff = 0.1, freq_sampling = 0.01, max_rate = 1.0,
                                             initial_value = 0.0, max_dt = 0.05, eps=1e-6)
        u = []
        u_dot = [0.0]
        u_dotdot = [0.0, 0.0]
        y = []
        y_dot = [0.0]
        y_dotdot = [0.0, 0.0]
        for i, ti in enumerate(t):
            if ti < 25:
                ui = step(ti, initial_value=0.0, step_value=8.0, step_time=1.0)
            else:
                ui = step(ti, initial_value=6, step_value=3.0, step_time=51.0)
            u.append(ui)
            y.append(filter.update(uk=u[-1], clock=ti))
            if i > 0:
                dt = t[i] - t[i - 1]
                u_dot.append((u[-1] - u[-2]) / dt)
                y_dot.append((y[-1] - y[-2]) / dt)
            if i > 1:
                u_dotdot.append((u_dot[-1] - u_dot[-2]) / dt)
                y_dotdot.append((y_dot[-1] - y_dot[-2]) / dt)
        plt.figure()
        h = plt.subplot(3, 1, 1)
        plt.plot(t, u, t, y)
        plt.legend(["u", "y"]), plt.ylabel('$x$')
        plt.grid(True)
        plt.subplot(3, 1, 2, sharex=h)
        plt.plot(t, u_dot, t, y_dot)
        plt.grid(True), plt.ylabel('$dx/dt$'), plt.ylim(-5, 5)
        plt.subplot(3, 1, 3, sharex=h)
        plt.plot(t, u_dotdot, t, y_dotdot)
        plt.grid(True), plt.ylabel('$dx/dt^2$'), plt.ylim(-20, 20), plt.xlabel('t[sec]')

class SecondOrderLPF_WithRateLimit:
    def __init__(self, clock, freq_cutoff, restraint_coefficient, freq_sampling, max_rate,
                 initial_value, max_dt = 0.05, eps=1e-6):
        """
        implement LPF as k * (uk-y[k-1])-> LPF -> integrator
        limit the input to the integrator
        """
        self.dt = 1 / (2 * np.pi * freq_sampling)
        self.wc = freq_cutoff * 2 * np.pi # cut off frequency in rad/sec
        self.zeta = restraint_coefficient
        self.wc_LPO1 = 2 * self.zeta * self.wc # cut off frequency in rad/sec of the 1st order filter
        self.K = (self.wc ** 2) / self.wc_LPO1
        self.low_pass_obj = FirstOrderLPF(clock=clock, freq_cutoff=self.wc_LPO1, freq_sampling=freq_sampling,
                                          initial_value=initial_value, max_dt = 0.05, eps=1e-6)
        self.y_k_minus_1 = initial_value
        self.yk = initial_value
        self.u_k_minus_1 = None
        self.max_dt = max_dt
        self.last_update_time = clock
        self.eps = eps
        self.max_rate = max_rate
        self.integrator = initial_value
    def initialize_clock(self, clock):
        self.last_update_time = clock
    def update(self, uk, clock):
        def limit_x(x, x_limit):
            return max(min(x, x_limit), -x_limit)
        if clock - self.last_update_time > self.max_dt:
            self.initialize_clock(clock)
            return self.integrator
        self.dt = clock - self.last_update_time
        self.last_update_time = clock
        input_to_LPO1 = self.K * (uk - self.y_k_minus_1)
        input_to_integrator = self.low_pass_obj.update(input_to_LPO1, clock)
        input_to_integrator_limited = limit_x(input_to_integrator, self.max_rate)
        self.integrator += input_to_integrator_limited * self.dt
        self.y_k_minus_1 = self.integrator.copy()
        return self.integrator

    def step_test(self):
        def step(clock, initial_value, step_value, step_time):
            if clock < step_time:
                return initial_value
            else:
                return step_value
        t1 = np.arange(start=0, stop=25, step=0.01)
        t2 = np.arange(start=50, stop=60, step=0.01)
        t = np.concatenate([t1, t2])
        filter = SecondOrderLPF_WithRateLimit(clock=t[0], freq_cutoff = 0.1, restraint_coefficient=0.5,
                                              freq_sampling = 0.01, max_rate = 1.0,
                                              initial_value = 0.0, max_dt = 0.05, eps=1e-6)
        u = []
        u_dot = [0.0]
        u_dotdot = [0.0, 0.0]
        y = []
        y_dot = [0.0]
        y_dotdot = [0.0, 0.0]
        for i, ti in enumerate(t):
            if ti < 25:
                ui = step(ti, initial_value=0.0, step_value=8.0, step_time=1.0)
            else:
                ui = step(ti, initial_value=0, step_value=3.0, step_time=55.0)
            u.append(ui)
            y.append(filter.update(uk=u[-1], clock=ti))
            if i > 0:
                dt = t[i] - t[i - 1]
                u_dot.append((u[-1] - u[-2]) / dt)
                y_dot.append((y[-1] - y[-2]) / dt)
            if i > 1:
                u_dotdot.append((u_dot[-1] - u_dot[-2]) / dt)
                y_dotdot.append((y_dot[-1] - y_dot[-2]) / dt)
        plt.figure()
        h = plt.subplot(3, 1, 1)
        plt.plot(t, u, t, y)
        plt.legend(["u", "y"]), plt.ylabel('$x$')
        plt.grid(True)
        plt.subplot(3, 1, 2, sharex=h)
        plt.plot(t, u_dot, t, y_dot)
        plt.grid(True), plt.ylabel('$dx/dt$'), plt.ylim(-5, 5)
        plt.subplot(3, 1, 3, sharex=h)
        plt.plot(t, u_dotdot, t, y_dotdot)
        plt.grid(True), plt.ylabel('$dx/dt^2$'), plt.ylim(-20, 20), plt.xlabel('t[sec]')

class SecondOrderLPF_WithRateAndAccLimit:
    def __init__(self, clock, freq_cutoff, restraint_coefficient, freq_sampling, max_rate, max_neg_rate,
                 max_acc, initial_value, initial_derivative, max_dt = 0.05, eps=1e-6):
        """
        implement LPF as k * (uk-y[k-1])-> LPF -> integrator
        limit the input to the integrator
        """
        self.dt = 1 / (2 * np.pi * freq_sampling)
        self.wc = freq_cutoff * 2 * np.pi # cut off frequency in rad/sec
        self.zeta = restraint_coefficient
        self.wc_LPO1 = 2 * self.zeta * self.wc # cut off frequency in rad/sec of the 1st order filter
        self.K = (self.wc ** 2) / self.wc_LPO1
        self.max_dt = max_dt
        self.last_update_time = clock
        self.eps = eps
        self.max_rate = max_rate
        self.max_neg_rate = max_neg_rate
        self.max_acc = max_acc
        self.integrator_1 = initial_derivative
        self.integrator_2 = initial_value
    def initialize_clock(self, clock):
        self.last_update_time = clock
    def initialize(self, clock, initial_value, initial_derivative):
        self.last_update_time = clock
        self.integrator_1 = initial_derivative
        self.integrator_2 = initial_value
    def update(self, uk, clock, ref=0, ref_dot=0):
        def limit_x(x, x_limit, x_neg_limit):
            return max(min(x, x_limit), -x_neg_limit)
        if clock - self.last_update_time > self.max_dt:
            self.initialize(clock, ref, ref_dot)
            return self.integrator_2, self.integrator_1
        self.dt = clock - self.last_update_time
        self.last_update_time = clock
        input_to_LPO1 = self.K * (uk - self.integrator_2)

        """if 1st integrator is out of range, bringing it to range with higher priority"""
        if self.integrator_1 < -self.max_neg_rate - self.max_acc * self.dt:
            self.integrator_1 += self.max_acc * self.dt
        elif self.integrator_1 > self.max_rate + self.max_acc * self.dt:
            self.integrator_1 -= self.max_acc * self.dt
        else:
            """1st order LPF with limited integrator"""
            input_to_integrator_LPO1 = self.wc_LPO1 * (input_to_LPO1 - self.integrator_1)
            input_to_integrator_LPO1_limited = limit_x(input_to_integrator_LPO1, self.max_acc, self.max_acc)
            self.integrator_1 += input_to_integrator_LPO1_limited * self.dt
            self.integrator_1 = limit_x(self.integrator_1, self.max_rate, self.max_neg_rate)

        """LPFO1 output is fed to 2nd integrator"""
        self.integrator_2 += self.integrator_1 * self.dt
        # self.y_k_minus_1 = self.integrator_2.copy()
        return self.integrator_2, self.integrator_1
    def step_test(self):
        def step(clock, initial_value, step_value, step_time):
            if clock < step_time:
                return initial_value
            else:
                return step_value
        t1 = np.arange(start=0, stop=25, step=0.01)
        t2 = np.arange(start=40, stop=70, step=0.01)
        t = np.concatenate([t1, t2])
        filter = SecondOrderLPF_WithRateAndAccLimit(clock=t[0], freq_cutoff = 0.8, restraint_coefficient=3.0,
                                              freq_sampling = 0.01, max_rate = 1.0, max_neg_rate = 0.5, max_acc=1.0,
                                              initial_value = 12, initial_derivative=3, max_dt = 0.05, eps=1e-6)
        u = []
        u_dot = [0.0]
        u_dotdot = [0.0, 0.0]
        y = []
        y_dot = [0.0]
        y_dotdot = [0.0, 0.0]
        for i, ti in enumerate(t):
            if ti < 25:
                ui = step(ti, initial_value=0.0, step_value=8.0, step_time=1.0)
            else:
                ui = step(ti, initial_value=10, step_value=3.0, step_time=58.0)
            u.append(ui)
            ref = 0 if not y else y[-1]
            y_i,_ = filter.update(uk=u[-1], clock=ti, ref=ref)
            y.append(y_i)
            if i > 0:
                dt = t[i] - t[i - 1]
                u_dot.append((u[-1] - u[-2]) / dt)
                y_dot.append((y[-1] - y[-2]) / dt)
            if i > 1:
                u_dotdot.append((u_dot[-1] - u_dot[-2]) / dt)
                y_dotdot.append((y_dot[-1] - y_dot[-2]) / dt)
        plt.figure()
        h = plt.subplot(3, 1, 1)
        plt.plot(t, u, t, y)
        plt.legend(["u", "y"]), plt.ylabel('$x$')
        plt.grid(True)
        plt.subplot(3, 1, 2, sharex=h)
        plt.plot(t, u_dot, t, y_dot)
        plt.grid(True), plt.ylabel('$dx/dt$'), plt.ylim(-5, 5)
        plt.subplot(3, 1, 3, sharex=h)
        plt.plot(t, u_dotdot, t, y_dotdot)
        plt.grid(True), plt.ylabel('$dx/dt^2$'), plt.ylim(-20, 20), plt.xlabel('t[sec]')

def filter_waypoint_velocities(input_p, input_v):

    def process_waypoint(p_t, v_t, p0, v0, t0, dt):
        velocity_filter = SecondOrderLPF_WithRateAndAccLimit(clock=t0, freq_cutoff=1, restraint_coefficient=3.0,
                                                             freq_sampling=0.01, max_rate=0.7, max_acc=0.8,
                                                             initial_value=v0, initial_derivative=0.0, max_dt=0.5,
                                                             eps=1e-6)
        p_filtered = [p0]
        v_filtered = [v0]
        t = [t0]
        while p_filtered[-1] < p_t:
            v_f, _ = velocity_filter.update(v_t, t[-1] + dt)
            v_filtered.append(v_f)
            t.append(t[-1] + dt)
            p_filtered.append(p_filtered[-1] + v_filtered[-1] * dt)
        return p_filtered, v_filtered, t

    # a0, v0, p0 = (0.0, 10.0, 0.0)
    # input_p = [60.0, 100.0]
    # input_v = [25.0, 5.0]
    a0, v0, p0 = (0.0, input_v[0], 0.0)
    dt = 0.1
    p_filtered, v_filtered, t_filtered = ([p0],[v0],[0.0])
    for pi,vi in zip(input_p, input_v):
        p_i_f, v_i_f, t_i_f = process_waypoint(pi, vi,p_filtered[-1], v_filtered[-1],t_filtered[-1],dt)
        p_filtered.extend(p_i_f)
        v_filtered.extend(v_i_f)
        t_filtered.extend(t_i_f)

    # plt.figure()
    desired_p = [p0, input_p[0], input_p[0], input_p[1]]
    desired_v = [input_v[0], input_v[0], input_v[1], input_v[1]]

    # plt.plot(desired_p, desired_v)
    # plt.plot(p_filtered, np.array(v_filtered) * 3.6)
    # plt.xlabel('$p [m]$'), plt.ylabel(r'$v [m/sec]$')
    # plt.grid(True)
    # plt.show()

    return p_filtered, v_filtered, t_filtered

def filter_waypoint_velocities_backwards():
    def process_waypoint1(p_t, v_t, p0, v0, a0, t0, dt):
        velocity_filter = SecondOrderLPF_WithRateAndAccLimit(clock=t0, freq_cutoff=1, restraint_coefficient=3.0,
                                                             freq_sampling=0.01, max_rate=1.0, max_acc=1.0,
                                                             initial_value=v0, initial_derivative=a0, max_dt=0.5,
                                                             eps=1e-6)
        p_filtered = [p0]
        v_filtered = [v0]
        a_filtered = [a0]
        t = [t0]
        print("p0, v0, a0", p0, v0, a0)
        print("pt, vt", p_t, v_t)
        while p_filtered[-1] < p_t:
            v_f, a_f = velocity_filter.update(v_t, t[-1] + dt)
            print("out", v_f, a_f)
            v_filtered.append(v_f)
            a_filtered.append(a_f)
            t.append(t[-1] + dt)
            p_filtered.append(p_filtered[-1] + v_filtered[-1] * dt)
        return p_filtered, v_filtered, a_filtered, t

    a0, v0, p0 = (0.0, 10.0, 0.0) # kph
    input_p = [0.0, 100.0]
    input_v = [20.0, 20.0] #kph

    input_p_reversed = [input_p[-1] - p for p in reversed(input_p)]
    input_p_reversed = input_p_reversed[1:]
    input_p_reversed.append(input_p[-1])
    input_v_reversed = np.array(list(reversed(input_v))) / 3.6 # m/s
    print("input_p_reversed", input_p_reversed)
    print("input_v_reversed", list(input_v_reversed))

    p_filtered_0, v_filtered_0, t_filtered_0 = filter_waypoint_velocities(input_p_reversed, input_v_reversed)
    input_p_new = [p_filtered_0[-1] - p for p in reversed(p_filtered_0)]
    input_v_new = np.array(list(reversed(v_filtered_0))) * 3.6 #kph

    dt = 0.1
    p_filtered, v_filtered, a_filtered, t_filtered = ([p0],[v0/3.6],[0.0], [0.0])
    # print(input_p_new)
    for pi,vi in zip(input_p_new, input_v_new):
        p_i_f, v_i_f, a_i_f, t_i_f = process_waypoint1(pi, vi/3.6,
                                                       p_filtered[-1], v_filtered[-1],
                                                       a_filtered[-1], t_filtered[-1],dt)
        p_filtered.extend(p_i_f)
        v_filtered.extend(v_i_f)
        a_filtered.extend(a_i_f)
        t_filtered.extend(t_i_f)

    plt.figure()
    desired_p = [p0, input_p[0], input_p[0], input_p[1]]
    desired_v = [input_v[0], input_v[0], input_v[1], input_v[1]]
    # desired_p = [p0, input_p_reversed[0], input_p_reversed[0], input_p_reversed[1]]
    # desired_v = [input_v_reversed[0], input_v_reversed[0], input_v_reversed[1], input_v_reversed[1]]

    plt.plot(desired_p, desired_v)
    # plt.plot(p_filtered_0, v_filtered_0)
    # plt.show()
    # plt.figure()
    plt.plot(input_p_new, input_v_new)
    plt.plot(p_filtered, np.array(v_filtered) * 3.6)
    plt.xlabel('$p [m]$'), plt.ylabel(r'$v [m/sec]$')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    f = SecondOrderLPF_WithRateAndAccLimit(clock=0, freq_cutoff=1, restraint_coefficient=3.0,
                                                             freq_sampling=0.01, max_rate=0.7, max_neg_rate=0.35, max_acc=0.8,
                                                             initial_value=0, initial_derivative=0.0, max_dt=0.5,
                                                             eps=1e-6)
    f.step_test()
    plt.show()
