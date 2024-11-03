import math
from multipledispatch import dispatch
import matplotlib.pyplot as plt
import numpy as np
import Functions
from Functions import fold_angles, epsilon_limit
from os.path import join
import pandas as pd
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
        self.vx = self.simulation_params['velocity_KPH'] / 3.6

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