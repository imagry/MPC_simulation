import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from Classes import StanleyController, VehicleDynamicModel, VehicleKinemaicModel, MPC
import Functions
import json
from copy import copy, deepcopy
import cvxpy
import math
class MPC_params:
    # System config
    NX = 4  # state vector: z = [x, y, v, phi]
    NU = 2  # input vector: u = [acceleration, steer]
    T = 25  # finite time horizon length

    # MPC config
    Q = np.diag([0.1, 1.0, 1.0, 100.0])  # penalty for states
    Qf = np.diag([0.1, 1.0, 0.1, 100.0])  # penalty for end state
    R = np.diag([0.01, 10.0])  # penalty for inputs acc and steer
    Rd = np.diag([0.1, 10.0])  # penalty for change of inputs

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
    speed_max = 100.0 / 3.6  # maximum speed [m/s]
    speed_min = -20.0 / 3.6  # minimum speed [m/s]
    acceleration_max = 1.0  # maximum acceleration [m/s2]
    min_velocity = 1
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
        v0 = max(P.min_velocity, z0[2])
        R_mod = np.diag([1, v0**2])  # penalty modification due to state
        Rd_mod = np.diag([1, v0**2])  # penalty modification due to state for change of inputs
        R = P.R @ R_mod
        Rd = P.Rd @ Rd_mod
        for t in range(P.T):
            cost += cvxpy.quad_form(u[:, t], R)
            actuation_cost += cvxpy.quad_form(u[:, t], R)
            a_cost += (u[0, t] ** 2) * R[0,0]
            delta_cost += (u[1, t] ** 2) * R[1, 1]
            cost += cvxpy.quad_form(z_ref[:, t] - z[:, t], P.Q)
            state_error_cost += cvxpy.quad_form(z_ref[:, t] - z[:, t], P.Q)
            x_cost += ((z_ref[0, t] - z[0, t]) ** 2) * P.Q[0, 0]
            y_cost += ((z_ref[1, t] - z[1, t]) ** 2) * P.Q[1, 1]
            v_cost += ((z_ref[2, t] - z[2, t]) ** 2) * P.Q[2, 2]
            psi_cost += ((z_ref[3, t] - z[3, t]) ** 2) * P.Q[3, 3]

            A, B, C = MPC.calc_linear_discrete_model(z_bar[2, t], z_bar[3, t], d_bar[t], P)

            constrains += [z[:, t + 1] == A @ z[:, t] + B @ u[:, t] + C]

            if t < P.T - 1: # horizon length
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
                actuation_change_cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
                a_change_cost += ((u[0, t + 1] - u[0, t]) ** 2) *  Rd[0, 0]
                delta_change_cost += ((u[1, t + 1] - u[1, t]) ** 2) * Rd[1, 1]
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

with open('vehicle_config.json', "r") as f:
    vehicle_params = json.loads(f.read())
simulation_params = {'dt': 0.01, 't_end': 50, 'ego_frame_placement': 'front_axle', 'velocity_KPH': 50,
                     'path_spacing': 1.0,
                     'model': 'Kinematic', #'Kinematic', 'Dynamic'
                     'animate': True, 'plot_results': True, 'save_results': False}
# generate path
scenario = 'straight_line' # 'sin', 'straight_line', 'square', shiba, random_curvature,turn, original_from_repo
traj_spline_x, traj_spline_y, traj_spline_psi, _, s = Functions.calc_desired_path(scenario, ds=simulation_params['path_spacing'])
# create vehicle agent
error_x = 0.0
error_y = 1.0
psi_error = 0.3
vehicle_obj = VehicleKinemaicModel(x=traj_spline_x[0] + error_x, y=traj_spline_y[0] + error_y, psi=traj_spline_psi[0] + psi_error,
                                   vehicle_params=copy(vehicle_params), simulation_params=copy(simulation_params),
                                   steering_uncertainty_factor=1.0, lr_uncertainty_factor=1.0, WB_uncertainty_factor=1.0)
t = np.arange(0, simulation_params['t_end'], simulation_params['dt'])
vehicle_obj.vx = simulation_params['velocity_KPH'] / 3.6
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
if False:
    # check linearization
    psi0 = 0.0
    x0 = 0
    y0 = 1
    v0 = 10
    # z_k = np.array([vehicle_obj.x, vehicle_obj.y, vehicle_obj.vx, vehicle_obj.psi])
    z_k = np.array([x0, y0, v0, psi0])
    dt = 0.1
    a = 0.25
    delta = 0.4
    MPC_obj.MPC_params.dt = dt
    A, B, C = MPC.calc_linear_discrete_model(z_k[2], z_k[3], MPC_obj.delta_exc, MPC_obj.MPC_params)
    Aorg, Borg, Corg = MPC.calc_linear_discrete_model_org(z_k[2], z_k[3], MPC_obj.delta_exc, MPC_obj.MPC_params)
    # u = np.array([MPC_obj.a_exc, MPC_obj.delta_exc])
    u = [a, delta]
    z_kp1_lin = A @ z_k + B @ u + C
    z_kp1_lin_org = Aorg @ z_k + Borg @ u + Corg
    print("z_kp1_lin = " + str(z_kp1_lin))
    print("z_kp1_lin_org = " + str(z_kp1_lin_org))
    vehicle_obj.x = z_k[0]
    vehicle_obj.y = z_k[1]
    vehicle_obj.vx = z_k[2]
    vehicle_obj.psi = z_k[3]
    vehicle_obj.simulation_params["dt"] = dt
    # vehicle_obj.update(a=MPC_obj.a_exc, delta=MPC_obj.delta_exc)
    vehicle_obj.update(a=a, delta=delta)
    z_kp1 = np.array([vehicle_obj.x, vehicle_obj.y, vehicle_obj.vx, vehicle_obj.psi])
    dz = z_kp1 - z_k
    print("z_kp1 = " + str(z_kp1))



# exit()
animation_figure = plt.figure()
vehicle_animation_axis = plt.subplot(1, 1, 1)
ref_traj_line = vehicle_animation_axis.plot(traj_spline_x, traj_spline_y, color='gray', linewidth=2.0)
vehicle_line = Functions.draw_car(vehicle_obj.x, vehicle_obj.y, vehicle_obj.psi, steer=0, car_params=vehicle_params, ax=vehicle_animation_axis)
vehicle_animation_axis.axis("equal")
vehicle_animation_axis.grid(True)
vehicle_animation_axis.set_xlabel('x [m]')
vehicle_animation_axis.set_ylabel('y [m]')
vehicle_animation_axis.scatter(MPC_obj.x_opt, MPC_obj.y_opt, s=10)
# vehicle_animation_axis.set_xlim(vehicle_obj.x - 25, vehicle_obj.x + 25)
# vehicle_animation_axis.set_ylim(vehicle_obj.y - 25, vehicle_obj.y + 25)


plt.figure('state')
# x
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
# y
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
# v
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
# psi
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
