import numpy as np
np.random.seed(0)
import math
import matplotlib.pyplot as plt
import pandas as pd
from sys import platform
import CurvesGenerator.cubic_spline as cs
from multipledispatch import dispatch
import json
EPSILON = 1e-6
class Arrow:
    def __init__(self, x, y, theta, L, c, ax=None):
        angle = np.deg2rad(30)
        d = 0.4 * L
        w = 2

        x_start = x
        y_start = y
        x_end = x + L * np.cos(theta)
        y_end = y + L * np.sin(theta)

        theta_hat_L = theta + math.pi - angle
        theta_hat_R = theta + math.pi + angle

        x_hat_start = x_end
        x_hat_end_L = x_hat_start + d * np.cos(theta_hat_L)
        x_hat_end_R = x_hat_start + d * np.cos(theta_hat_R)

        y_hat_start = y_end
        y_hat_end_L = y_hat_start + d * np.sin(theta_hat_L)
        y_hat_end_R = y_hat_start + d * np.sin(theta_hat_R)
        self.arrow_line_handle = []
        if ax == None:
            self.arrow_line_handle.append(plt.plot([x_start, x_end], [y_start, y_end], color=c, linewidth=w))
            self.arrow_line_handle.append(plt.plot([x_hat_start, x_hat_end_L],
                     [y_hat_start, y_hat_end_L], color=c, linewidth=w))
            self.arrow_line_handle.append(plt.plot([x_hat_start, x_hat_end_R],
                     [y_hat_start, y_hat_end_R], color=c, linewidth=w))
        else:
            self.arrow_line_handle.append(ax.plot([x_start, x_end], [y_start, y_end], color=c, linewidth=w))
            self.arrow_line_handle.append(ax.plot([x_hat_start, x_hat_end_L],
                     [y_hat_start, y_hat_end_L], color=c, linewidth=w))
            self.arrow_line_handle.append(ax.plot([x_hat_start, x_hat_end_R],
                     [y_hat_start, y_hat_end_R], color=c, linewidth=w))

def draw_car(x, y, yaw, steer, car_params, color='black', ax=None):
    CGB = car_params['RB'] + car_params['lr'] # distance from cg to back edge
    CGF = car_params['RF'] - car_params['lr'] # distance from cg to front edge
    lf = car_params['WB'] - car_params['lr']
    # rectangle with cg at [0,0]
    # vehicle config
    # RF = 3.3  # [m] distance from rear to vehicle front end of vehicle
    # RB = 0.8  # [m] distance from rear to vehicle back end of vehicle
    # W = 2.4  # [m] width of vehicle
    # WD = 0.7 * W  # [m] distance between left-right wheels
    # WB = 2.5  # [m] Wheel base this is L
    # TR = 0.44  # [m] Tyre radius
    # TW = 0.7  # [m] Tyre width
    car = np.array([[-CGB, -CGB, CGF, CGF, -CGB],
                    [car_params['W'] / 2, -car_params['W'] / 2, -car_params['W'] / 2, car_params['W'] / 2, car_params['W'] / 2]])

    wheel = np.array([[-car_params['TR'], -car_params['TR'], car_params['TR'], car_params['TR'], -car_params['TR']],
                      [car_params['TW'] / 4, -car_params['TW'] / 4, -car_params['TW'] / 4, car_params['TW'] / 4, car_params['TW'] / 4]])

    rlWheel = wheel.copy()
    rrWheel = wheel.copy()
    frWheel = wheel.copy()
    flWheel = wheel.copy()

    Rot1 = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])

    Rot2 = np.array([[math.cos(steer), -math.sin(steer)],
                     [math.sin(steer), math.cos(steer)]])

    frWheel = np.dot(Rot2, frWheel)
    flWheel = np.dot(Rot2, flWheel)

    frWheel += np.array([[lf], [-car_params['WD'] / 2]])
    flWheel += np.array([[lf], [car_params['WD'] / 2]])
    rrWheel += np.array([[-car_params['lr']], [-car_params['WD'] / 2]])
    rlWheel += np.array([[-car_params['lr']], [car_params['WD'] / 2]])

    frWheel = np.dot(Rot1, frWheel)
    flWheel = np.dot(Rot1, flWheel)

    rrWheel = np.dot(Rot1, rrWheel)
    rlWheel = np.dot(Rot1, rlWheel)
    car = np.dot(Rot1, car)

    frWheel += np.array([[x], [y]])
    flWheel += np.array([[x], [y]])
    rrWheel += np.array([[x], [y]])
    rlWheel += np.array([[x], [y]])
    car += np.array([[x], [y]])
    vehicle_line_handle = []
    if ax == None:
        vehicle_line_handle.append(plt.plot(car[0, :], car[1, :], color))
        vehicle_line_handle.append(plt.plot(frWheel[0, :], frWheel[1, :], color))
        vehicle_line_handle.append(plt.plot(rrWheel[0, :], rrWheel[1, :], color))
        vehicle_line_handle.append(plt.plot(flWheel[0, :], flWheel[1, :], color))
        vehicle_line_handle.append(plt.plot(rlWheel[0, :], rlWheel[1, :], color))
        arrow_instance = Arrow(x, y, yaw, car_params['WB'] * 0.6, color)
        vehicle_line_handle.extend(arrow_instance.arrow_line_handle)
    else:
        vehicle_line_handle.append(ax.plot(car[0, :], car[1, :], color))
        vehicle_line_handle.append(ax.plot(frWheel[0, :], frWheel[1, :], color))
        vehicle_line_handle.append(ax.plot(rrWheel[0, :], rrWheel[1, :], color))
        vehicle_line_handle.append(ax.plot(flWheel[0, :], flWheel[1, :], color))
        vehicle_line_handle.append(ax.plot(rlWheel[0, :], rlWheel[1, :], color))
        arrow_instance = Arrow(x, y, yaw, car_params['WB'] * 0.6, color, ax=ax)
        vehicle_line_handle.extend(arrow_instance.arrow_line_handle)
    return vehicle_line_handle


def fold_angles(u):
    """angles are folded to +-pi"""
    u = np.squeeze(u)
    if len(u.shape) == 0:
        u = u.reshape([1, 1])

    NumOfSamples = u.shape[0]
    y = np.zeros(u.shape)
    for i in range(NumOfSamples):
        if u[i] >= 0:
            y[i] = np.mod(u[i], 2 * np.pi)
            if y[i] > np.pi:
                y[i] = y[i] - 2 * np.pi
        else:
            y[i] = np.mod(u[i], -2 * np.pi)
            if y[i] < -np.pi:
                y[i] = y[i] + 2 * np.pi
    return y.squeeze()


def calc_desired_path(scenario, ds=0.1, traj_noise=None, plot_results=False):
    # generate path
    if scenario == 'sin':
        x_range = 250.0
        wave_length = 50.0
        f = 2 * np.pi / wave_length
        A0 = 3.0
        Ae = 250.0
        traj_samples_x = np.arange(0, x_range, 0.5)
        traj_samples_y = [(A0 + (Ae - A0) / x_range * ix) * math.sin(f * ix) for ix in traj_samples_x]
        traj_spline_x, traj_spline_y, traj_spline_psi, traj_spline_cur, _ = cs.calc_spline_course(traj_samples_x, traj_samples_y, ds=ds)
    elif scenario == 'original_from_repo':
        ax = [0.0, 15.0, 30.0, 50.0, 60.0]
        ay = [0.0, 40.0, 15.0, 30.0, 0.0]
        traj_spline_x, traj_spline_y, traj_spline_psi, traj_spline_cur, _ = cs.calc_spline_course(ax, ay, ds=ds)
    elif scenario == 'straight_line':
        x_range = 50.0
        initial_error = 1.0
        traj_samples_x = np.arange(0, x_range, 0.5)
        traj_samples_y = initial_error * np.ones(traj_samples_x.shape)
        traj_spline_x, traj_spline_y, traj_spline_psi, traj_spline_cur, _ = cs.calc_spline_course(traj_samples_x, traj_samples_y, ds=ds)
    elif scenario == 'shiba':
        if platform == 'linux':
            data = pd.read_csv("data/backed_data_files/shiba_traj_splines.csv")
        elif platform == 'win32':
            data = pd.read_csv("data\\backed_data_files\\shiba_traj_splines.csv")
        else:
            raise 'unhandled platform'
        traj_spline_x = np.array(data.x)
        traj_spline_y = np.array(data.y)
        traj_spline_psi = np.array(data.psi)
        traj_spline_cur = np.array(data.psi)
    elif scenario == 'turn':
        direction = 'right'
        if direction == 'right':
            SF = - 1.0
        else:
            SF = 1.0
        l1 = 10.0
        l2 = 30.0
        dl = 0.5
        traj_samples_x = np.arange(0, l1, dl)
        traj_samples_y = np.zeros([traj_samples_x.shape[0]])
        traj_samples_x = np.hstack([traj_samples_x, np.ones([int(l2/dl)]) * l1])
        traj_samples_y = np.hstack([traj_samples_y, SF * np.arange(0, l2, dl)])
        traj_spline_x, traj_spline_y, traj_spline_psi, traj_spline_cur, _ = cs.calc_spline_course(traj_samples_x, traj_samples_y, ds=ds)
    elif scenario == 'random_curvature':
        s = np.arange(0, 250, ds)
        k = 1.0 * (np.random.rand(s.shape[0])-0.5)#.cumsum()
        traj_spline_x, traj_spline_y = calculate_curve_from_curvature(s, k, plot_res=False)
        dx = np.diff(traj_spline_x)
        dy = np.diff(traj_spline_y)
        traj_spline_psi = np.arctan2(dy, dx)
        traj_spline_cur = k
    elif scenario == 'eight':
        num_points = 1000
        # Define the parameter s (curve length)
        s = np.linspace(0, 2*np.pi, num_points)

        # Define the parametric equations for the lemniscate (figure-eight curve)
        x = np.sin(s)
        y = np.sin(s) * np.cos(s)

        x_range = 100
        y_range = 50
        x = x_range / 2 * x
        y = y_range * y
        # s, k, psi = calc_path_features(x,y,plot_results=True)
        traj_spline_x, traj_spline_y, traj_spline_psi, traj_spline_cur, traj_spline_s = cs.calc_spline_course(x, y, ds=0.1)
    elif scenario == 'ellipse':
        def ellipse_points(a, b, num_points=100):
            # Generate theta values
            theta = np.linspace(0, 2*np.pi, num_points)
            # Compute x and y
            x = a * np.cos(theta)
            y = b * np.sin(theta)
            return x, y

        direction = 'right' # 'right', 'left'
        if direction == 'right':
            SF = 1.0
        else:
            SF = - 1.0
        rx = 25  # Major axis
        ry = 10  # Minor axis
        x, y = ellipse_points(rx, ry)
        y += ry
        traj_spline_x, traj_spline_y, traj_spline_psi, traj_spline_cur, traj_spline_s = cs.calc_spline_course(x, SF * y, ds=0.1)
    else:
        raise 'invalid scenario'

    ds = [0.0]
    for i in range(1, len(traj_spline_x) ):
        dx = traj_spline_x[i] - traj_spline_x[i-1]
        dy = traj_spline_y[i] - traj_spline_y[i-1]
        ds.append(np.linalg.norm([dx, dy]))
    # temp = [0.0]
    # temp.extend(ds)
    # ds = np.array(temp)
    s = np.cumsum(ds)
    traj_spline_psi = continuous_angle(traj_spline_psi, 'rad')
    if traj_noise is not None:
        metric_noise = np.random.uniform(low=-traj_noise[0], high=traj_noise[0],
                                        size=np.array(traj_spline_x).size -1)
        heading_noise = np.random.uniform(low=-traj_noise[1], high=traj_noise[1],
                                           size=np.array(traj_spline_x).size-1)
        diff_traj_x = np.diff(traj_spline_x)
        diff_traj_y = np.diff(traj_spline_y)
        diff_traj_xy = np.vstack([diff_traj_x, diff_traj_y]).T
        norm_traj = np.linalg.norm(diff_traj_xy, axis=1)
        scrambled_diff_x = ((norm_traj + metric_noise) *
                            np.cos(np.array(traj_spline_psi)[:-1] + heading_noise))
        scrambled_diff_y = ((norm_traj + metric_noise) *
                            np.sin(np.array(traj_spline_psi)[:-1] + heading_noise))
        scrambled_diff_xy = np.vstack([scrambled_diff_x, scrambled_diff_y]).T
        added_noise = scrambled_diff_xy - diff_traj_xy

        traj_spline_x[:-1] += added_noise[:,0]
        traj_spline_y[:-1] += added_noise[:,1]
        traj_spline_psi[:-1] += heading_noise
    if plot_results:
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(traj_spline_x, traj_spline_y)
        plt.grid(True), plt.xlabel('[m]'), plt.ylabel('[m]')#, plt.axis('equal')
        plt.subplot(2,2,2)
        plt.plot(s, traj_spline_cur)
        plt.grid(True), plt.ylabel('curvature [1/m]')
        plt.subplot(2, 2, 4)
        plt.plot(s, traj_spline_psi)
        plt.grid(True), plt.xlabel('curve length [m]'), plt.ylabel('heading [rad]')
        plt.show()
    return traj_spline_x, traj_spline_y, traj_spline_psi, traj_spline_cur, s


def save_csv(dic, path, print_message=False):
    df = pd.DataFrame(data=dic)
    df.to_csv(path,index=False)
    if print_message:
        print('results saved to ' + path)


def epsilon_limit(u):
    sign = np.sign(u)
    if abs(sign) < 1.0:
        sign = 1.0
    y = sign * max(abs(u), EPSILON)
    return y


def allign_time_vectors(time_vectors):
    """
    export a common time vector with the smallest dt, maximal t0, minimal tend
    time_vectors: list where each element  is a np.array 1D vector
    """
    n = len(time_vectors)
    dt_list = []
    t0_list = []
    tend_list = []
    for time_vector in time_vectors:
        dt_list.append(np.mean(np.diff(time_vector)))
        t0_list.append(time_vector[0])
        tend_list.append(time_vector[-1])
    dt = min(dt_list)
    t0 = max(t0_list)
    tend = min(tend_list)
    t = np.arange(start=t0,stop=tend,step=dt)
    return t


def Radius(Lat):
    R0 = 6.378388e6
    Rp = 6.356912e6
    e = np.sqrt(1 - (Rp / R0) ** 2)
    RN = R0 / np.sqrt(1 - e ** 2 * np.sin(Lat) ** 2)
    RM = R0 * (1 - e ** 2) / (1 - e ** 2 * np.sin(Lat) ** 2) ** (3 / 2)
    Re = R0 * (1 - e * np.sin(Lat) ** 2)

    return RN, RM, Re


def LLA2ECEF(lat, lon, alt):
    # WGS84 ellipsoid constants:

    a = 6378137
    e = 8.1819190842622e-2

    # intermediate calculation
    # (prime vertical radius of curvature)
    N = a / np.sqrt(1 - np.power(e, 2) * np.power(np.sin(lat * np.pi / 180), 2))

    # results:
    x = (N + alt) * np.cos(lat * np.pi / 180) * np.cos(lon * np.pi / 180)
    y = (N + alt) * np.cos(lat * np.pi / 180) * np.sin(lon * np.pi / 180)
    z = ((1 - np.power(e, 2)) * N + alt) * np.sin(lat * np.pi / 180)

    return x, y, z


def Mphi(Phi):
    Mphi = np.array(
        [[1, 0, 0],
         [0, np.cos(Phi), np.sin(Phi)],
         [0, -np.sin(Phi), np.cos(Phi)]])
    return Mphi


def Mtheta(Theta):
    Mtheta = np.array(
        [[np.cos(Theta), 0, -np.sin(Theta)],
         [0, 1, 0],
         [np.sin(Theta), 0, np.cos(Theta)]])
    return Mtheta


def Mpsi(Psi):
    Mpsi = np.array(
        [[np.cos(Psi), np.sin(Psi), 0],
         [-np.sin(Psi), np.cos(Psi), 0],
         [0, 0, 1]])
    return Mpsi


def DCM_Ned2ECEF(Long, Lat):
    M12 = Mtheta(Lat * np.pi / 180 + np.pi / 2)
    M01 = Mpsi(-Long * np.pi / 180)
    DCM = np.dot(M01, M12)
    return DCM


def DCM_ECEF2NED(Long, Lat):
    M10 = Mpsi(Long * np.pi / 180)
    M21 = Mtheta(-Lat * np.pi / 180 - np.pi / 2)
    DCM = np.dot(M21, M10)
    return DCM


def ECEF2LLA(x, y, z):
    # WGS84 ellipsoid constants:
    a = float(6378137)
    e = 8.1819190842622e-2
    # calculations:
    b = np.sqrt(np.power(a, 2) * (1 - np.power(e, 2)))
    ep = np.sqrt((np.power(a, 2) - np.power(b, 2)) / np.power(b, 2))
    p = np.sqrt(np.power(x, 2) + np.power(y, 2))
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z + np.power(ep, 2) * b * np.power(np.sin(th), 3),
                     p - np.power(e, 2) * a * np.power(np.cos(th), 3))
    N = a / np.sqrt(1 - np.power(e, 2) * np.power(np.sin(lat), 2))
    alt = p / np.cos(lat) - N
    # return lon in range [0,360)
    lon = np.mod(lon, 2 * np.pi)
    # correct for numerical instability in altitude near exact poles:
    # (after this correction, error is about 2 millimeters, which is about
    # the same as the numerical precision of the overall function)
    if abs(x) < 1 and abs(y) < 1:
        alt = abs(z) - b

    return lat, lon, alt


def LocalNav2Geo(p0, L, Psi):
    """Find LLA location from azimuth distance"""
    # 1. calc p0 in ECEF
    [Xecef, Yecef, Zecef] = LLA2ECEF(p0[0], p0[1], p0[2])
    P0_ECEF = np.array([Xecef, Yecef, Zecef])
    # 2. calculate DCM NED->ECEF at p0
    R_ECEF_NED = DCM_Ned2ECEF(p0[0], p0[1])
    # 3.calc p1 in LLLN
    p1_NED = L * np.array([np.cos(Psi), np.sin(Psi), 0])
    # 4. calc 2nd point in ECEF
    p1_ECEF = np.dot(R_ECEF_NED, p1_NED) + P0_ECEF
    # 5. convert ECEF 2 LLA
    [Lat, Long, Alt] = ECEF2LLA(p1_ECEF[0], p1_ECEF[1], p1_ECEF[2])
    p1_LLA = np.array([Lat * 180 / np.pi, Long * 180 / np.pi, Alt])
    return p1_LLA


def LLA2NED(lat, long, alt):
    x_ECEF, y_ECEF, z_ECEF = LLA2ECEF(lat, long, alt)
    ECEF_arr = np.vstack([x_ECEF, y_ECEF, z_ECEF]).T
    DCM = DCM_ECEF2NED(Long=long[0], Lat=lat[0])
    n_e_d = np.dot(DCM, ECEF_arr.T)
    Pn = n_e_d[0, :].squeeze() - n_e_d[0, 0]
    Pe = n_e_d[1, :].squeeze() - n_e_d[1, 0]
    Pd = n_e_d[2, :].squeeze() - n_e_d[2, 0]
    return Pn,Pe,Pd


def continuous_angle(U, units='rad'):
    n = len(U)
    Y = U.copy()
    Counter = 0
    if units == 'deg':
        Cycle = 360
    elif units == 'rad':
        Cycle = 2 * np.pi
    else:
        print("ContinuousAngle:wrong units")
    for i in range(1, n):
        # if abs(U[i] - U[i - 1]) > Cycle / 10:
        #     print(U[i] - U[i - 1])
        if (U[i] - U[i - 1]) > Cycle / 2:
            Counter = Counter - 1
        elif (U[i] - U[i - 1]) < - Cycle / 2:
            Counter = Counter + 1
        Y[i] = U[i] + Cycle * Counter
    return Y


def affine_transformation_matrix_2D(x,y,psi):
    """
    calculate an affine transformation matrix which projects 2D points in CS1
    in to CS2.
    x,y are the coordinates of the origin of CS2 in CS1
    psi is the angle of Cs1 relative to CS2
    Test function with this script:
    CS2_origin_x = 2
    CS2_origin_y = 1
    p = np.array([3, 4]).reshape([2, 1])
    psi = np.pi/2
    T = affine_transformation_matrix_2D(x=CS2_origin_x, y=CS2_origin_y, psi=psi)
    p_homo = np.vstack([p, 1])
    p_in_CS2_homo = T.dot(p_homo)
    p_in_CS2 = p_in_CS2_homo[:2]
    print(p_in_CS2)
    """
    R = Mpsi(psi)[:2, :2]
    O2 = np.array([x, y]).reshape(2,1)
    third_row = np.array([0, 0, 1]).reshape([1, 3])
    T = np.concatenate((R, -R.dot(O2)), axis=1)
    T = np.concatenate((T, third_row), axis=0)
    return T


def inv_affine_transformation_matrix_2D(T : np.array):
    """
    Test function with script:
    CS2_origin_x = 2
    CS2_origin_y = 1
    p = np.array([3, 4]).reshape([2, 1])
    psi = np.pi/2
    T = affine_transformation_matrix_2D(x=CS2_origin_x, y=CS2_origin_y, psi=psi)
    T_inv = inv_affine_transformation_matrix_2D(T)
    print(T_inv)
    print("check that you get unitary matrix")
    print(T.dot(T_inv))
    """
    R = T[:2, :2]
    O1 = T[:2, 2].reshape([2, 1])
    T_inv = np.concatenate((R.T, -R.T.dot(O1)), axis=1)
    T_inv = np.concatenate((T_inv, np.array([0, 0, 1]).reshape([1, 3])), axis=0)
    return T_inv


@dispatch(float, float, float, np.ndarray)
def project_points_2D(x, y, psi, points: np.array):
    """
    project multiple points in CS1 on a CS2
    x,y are the coordinates of the origin of CS2 in CS1
    psi is the angle of CS1 relative to CS2
    points is an nX2 array.
    test with script:
    from  copy import copy
    px = np.linspace(0, 10, num=11).reshape([11, 1])
    py = copy(px)
    plt.plot(px, py, label="points")
    points_1 = np.concatenate((px, py), axis=1)
    CS2_origin_x = 1
    CS2_origin_y = 1
    psi = 30 * np.pi / 180
    points_2, T = project_points_2D(CS2_origin_x, CS2_origin_y, psi, points_1)
    plt.plot(points_2[:, 0], points_2[:, 1], label="points projected")
    plt.grid(True), plt.legend(), plt.gca().axis('equal')
    plt.show()
    """
    n = points.shape[0]
    T = affine_transformation_matrix_2D(x=x, y=y, psi=psi)
    points_homo = np.concatenate((points.T, np.ones(n).reshape([1, n])))
    points_projected_homo = T.dot(points_homo)
    points_projected = points_projected_homo[:2]
    return points_projected.T, T


@dispatch(np.ndarray, np.ndarray)
def project_points_2D(T12, points: np.array):
    """
    project multiple points in CS1 on a CS2
    T12 is the affine transformation matrix from CS1 to CS2
    points is an nX2 array.
    Test function with this script:
    CS2_origin_x = 2
    CS2_origin_y = 1
    p = np.array([3, 4]).reshape([2, 1])
    psi = np.pi/2
    T = affine_transformation_matrix_2D(x=CS2_origin_x, y=CS2_origin_y, psi=psi)
    p_in_CS2 = project_points_2D(T, p)
    print(p_in_CS2)

    """
    assert len(points.shape) <= 2
    assert len(points.shape) > 0
    if len(points.shape) == 1:
        points = np.expand_dims(points, axis=0)
    n = points.shape[0]
    points_homo = np.concatenate((points.T, np.ones(n).reshape([1, n])))
    points_projected_homo = T12.dot(points_homo)
    points_projected = points_projected_homo[:2]
    return points_projected.T


def project_point_on_path(point, path, exclude_points_behind_vehicle=False, psi=None):
    if exclude_points_behind_vehicle:
        assert psi is not None
        # desired path observation in point frame
        path_ego_frame, trans_nav2ego = project_points_2D(point[0], point[1],psi,
                                                                             path)
        idx = np.argwhere(path_ego_frame[:,0] > 0).squeeze()
        idx_in_idx_vec = np.argmin(np.linalg.norm(path_ego_frame[idx], axis=1))
        # delete plot
        # plt.scatter(path_ego_frame[:,0], path_ego_frame[:,1])
        # plt.grid(True)
        # plt.show()
        # print("path_ego_frame = \n")
        # print(path_ego_frame)
        # print("path_ego_frame[idx] = \n")
        # print(path_ego_frame[idx])
        # print("path_ego_frame[idx[idx_in_idx_vec]] = \n")
        # print(path_ego_frame[idx[idx_in_idx_vec]])
        # print("distance to closets point = " +  str(np.linalg.norm(path_ego_frame[idx[idx_in_idx_vec]])))
        # print("idx[idx_in_idx_vec] = " + str(idx[idx_in_idx_vec]))
        return int(idx[idx_in_idx_vec])

    # dx = [point[0] - x for x in path[:, 0]]
    # dy = [point[1] - y for y in path[:, 1]]
    # return int(np.argmin(np.hypot(dx, dy)))
    return np.argmin(np.linalg.norm(path - np.array(point), axis=1))


def nav_est_2_nav_affine_transformation_2D(x_est, y_est, psi_est,\
                                           x, y, psi):
    """
    calculate estimated Nav to GT Nav affine transformation matrix
    """
    Ten = affine_transformation_matrix_2D(x, y, psi)
    Tne = inv_affine_transformation_matrix_2D(Ten)
    Ten_hat = affine_transformation_matrix_2D(x_est, y_est, psi_est)
    Tnn_hat = np.matmul(Tne, Ten_hat)
    return Tnn_hat

def calculate_curve_from_curvature(curve_length, curvature, plot_res=False):
    # Calculate the tangent angle by integrating the curvature
    theta = np.cumsum(curvature) * (curve_length[1]-curve_length[0])  # approximate integral

    # Calculate the x and y coordinates of the curve points
    x = np.cumsum(np.cos(theta)) * (curve_length[1]-curve_length[0])  # approximate integral
    y = np.cumsum(np.sin(theta)) * (curve_length[1]-curve_length[0])  # approximate integral

    if plot_res:
        plt.figure(figsize=(6,6))
        plt.plot(x, y)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('2D line from curve length and curvature')
        plt.axis('equal')
        plt.grid(True)
        plt.show()
    """
    test function with:
    s = np.linspace(0, 10, 100)  # replace with your actual data
    k = np.ones(s.shape) * 0.5  # replace with your actual data
    k[50:] = - 0.25
    calculate_curve_from_curvature(s, k, plot_res=True)
    """
    return x, y

def calc_path_features(x, y, plot_results=False):
    # Calculate the derivative of y with respect to x
    dx = np.gradient(x)
    dy = np.gradient(y)

    # Calculate the second derivative of y with respect to x
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    # Calculate the curvature
    k = (dx * d2y - dy * d2x) / (dx ** 2 + dy ** 2)**1.5
    # Calculate the cumulative arc length
    ds = np.sqrt(dx**2 + dy**2)
    s = np.cumsum(ds)

    psi = np.arctan2(dy, dx)
    psi = continuous_angle(psi, 'rad')

    # Now s is your curve length and k is your curvature
    if plot_results:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.scatter(x, y)
        plt.grid(True), plt.xlabel('y [m]'), plt.ylabel('x [m]')
        plt.axis('equal')
        plt.subplot(2,2,2)
        plt.plot(s, k)
        plt.grid(True)
        plt.ylabel('curvature [1/m]')
        plt.subplot(2,2,4)
        plt.plot(s, psi)
        plt.plot([s[0], s[-1]], np.array([1,1]) * np.mean(psi), linestyle='--', color='k')
        plt.xlabel('s'), plt.ylabel('heading [rad]')
        plt.grid(True)
        # plt.show()
    return s, k, psi

def regulate_random_signal(signal, set_point, gain, plot_res=False):
    u = []
    regulator = 0
    for i, si in enumerate(signal):
        e = set_point - (si + regulator)
        regulator += e * gain
        u.append(regulator)
    if plot_res:
        plt.figure()
        plt.plot(signal)
        plt.plot(signal + u)
        plt.grid(True)
        plt.show()
    return np.array(u) + signal

def RMS(u):
    return np.sqrt(np.mean(u ** 2))

def FoldAngles(u):
    """angles are folded to +-pi"""
    u = np.squeeze(u)
    if len(u.shape) == 0:
        u = u.reshape([1, 1])

    NumOfSamples = u.shape[0]
    y = np.zeros(u.shape)
    for i in range(NumOfSamples):
        if u[i] >= 0:
            y[i] = np.mod(u[i], 2 * np.pi)
            if y[i] > np.pi:
                y[i] = y[i] - 2 * np.pi
        else:
            y[i] = np.mod(u[i], -2 * np.pi)
            if y[i] < -np.pi:
                y[i] = y[i] + 2 * np.pi
    return y.squeeze()
@dispatch(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
def PlotEulerAngles(Phi, PhiHat, Theta, ThetaHat, Psi, PsiHat, t_ref, t_hat=None):
    "insert angles in radians"
    if t_hat is None:
        assert (len(list(Phi)) == len(list(PhiHat)) and
                len(list(Theta)) == len(list(ThetaHat)) and
                len(list(Psi)) == len(list(PsiHat)))
    else:
        print("interpolating for visualization \n")
        # segment reference to be contained in estimation time
        t_hat = np.array(t_hat)
        t_ref = np.array(t_ref)
        t_ref_start_idx = np.argmin(np.abs(t_ref - t_hat[0])) + 1
        t_ref_stop_idx = np.argmin(np.abs(t_ref - t_hat[-1])) - 1
        t_ref = t_ref[t_ref_start_idx:t_ref_stop_idx]
        Phi = np.array(Phi[t_ref_start_idx:t_ref_stop_idx])
        Theta = np.array(Theta[t_ref_start_idx:t_ref_stop_idx])
        Psi = np.array(Psi[t_ref_start_idx:t_ref_stop_idx])
        # interpolate estimation to reference time
        PhiHat = np.interp(x=t_ref, xp=t_hat, fp=PhiHat)
        ThetaHat = np.interp(x=t_ref, xp=t_hat, fp=ThetaHat)
        PsiHat = np.interp(x=t_ref, xp=t_hat, fp=PsiHat)
    plt.close('Euler Angles Plot')
    fig = plt.figure('Euler Angles Plot')
    Ax1 = fig.add_subplot(321)
    t_hat -= t_hat[0]
    Ax1.plot(t_ref, Phi * 180 / np.pi, color='blue', linewidth=1)
    Ax1.plot(t_ref, PhiHat * 180 / np.pi, color='red', linewidth=1)
    Ax1.set(title=r"$\phi$", xlabel="", ylabel="[deg]"), Ax1.grid(True)
    plt.legend(['ref', 'est'])

    ePhi = Phi - PhiHat
    Ax2 = fig.add_subplot(322, sharex=Ax1)
    Ax2.plot(t_ref, ePhi * 180 / np.pi, color='black', linewidth=1)
    Ax2.set(title=r"$e_{\Phi}$, RMS = " + str("%.2f" % (RMS(ePhi) * 180 / np.pi)), xlabel="", ylabel="[deg]"), Ax2.grid(True)

    Ax3 = fig.add_subplot(323, sharex=Ax1)
    Ax3.plot(t_ref, Theta * 180 / np.pi, color='blue', linewidth=1)
    Ax3.plot(t_ref, ThetaHat * 180 / np.pi, color='red', linewidth=1)
    Ax3.set(title=r"$\theta$", xlabel="", ylabel="[deg]"), Ax3.grid(True)

    eTheta = Theta - ThetaHat
    Ax4 = fig.add_subplot(324, sharex=Ax1)
    Ax4.plot(t_ref, eTheta * 180 / np.pi, color='black', linewidth=1)
    Ax4.set(title=r"$e_{\theta}$, RMS = " + str("%.2f" % (RMS(eTheta) * 180 / np.pi)), xlabel="", ylabel="[deg]"), Ax4.grid(True)

    Ax5 = fig.add_subplot(325, sharex=Ax1)
    Ax5.plot(t_ref, Psi * 180 / np.pi, color='blue', linewidth=1)
    Ax5.plot(t_ref, PsiHat * 180 / np.pi, color='red', linewidth=1)
    Ax5.set(title=r"$\psi$", xlabel="time [sec]", ylabel="[deg]"), Ax5.grid(True)

    ePsi = FoldAngles(Psi - PsiHat)
    Ax6 = fig.add_subplot(326, sharex=Ax1)
    Ax6.plot(t_ref, ePsi * 180 / np.pi, color='black', linewidth=1)
    Ax6.set(title=r"$e_{\psi}$, RMS = " + str("%.2f" % (RMS(ePsi) * 180 / np.pi)), xlabel="time [sec]", ylabel="[deg]"), Ax6.grid(True)

    plt.tight_layout()
    plt.show()

@dispatch(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
def PlotEulerAngles(Phi, Theta, Psi, t):
    "insert angles in radians"

    plt.close('Euler Angles Plot')
    fig = plt.figure('Euler Angles Plot')
    Ax1 = fig.add_subplot(311)
    Ax1.plot(t, Phi , color='blue', linewidth=1)
    Ax1.set(title=r"$\phi$", xlabel="", ylabel="[deg]"), Ax1.grid(True)

    Ax2 = fig.add_subplot(312, sharex=Ax1)
    Ax2.plot(t, Theta, color='blue', linewidth=1)
    Ax2.set(title=r"$\theta$", xlabel="", ylabel="[deg]"), Ax2.grid(True)

    Ax3 = fig.add_subplot(313, sharex=Ax1)
    Ax3.plot(t, Psi , color='blue', linewidth=1)
    Ax3.set(title=r"$\psi$", xlabel="time [sec]", ylabel="[deg]"), Ax3.grid(True)

    plt.tight_layout()
    plt.show()

def CalcO3PolyCoef(x0, x_final, v_max, plot_res=False, sim_params=None):
    """
    calculate coefficients for 3rd order polynomial
    x(t) = a0 + a1 * t + a2 * t^2 + a3 * t^3
    for given boundary conditions:
    x(t = 0) = x0
    x(t = T) = x_final
    d_dt[x(t = 0)] = 0
    d_dt[x(t = T)] = 0
    d2_dt2[x(t = t1)] = 0
    | d_dt[x(t1)] | = v_max
    return coefs = [a0, a1, a2, a3]
    """
    def calc_x(t):
        return a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3
    def calc_dx_dt(t):
        return a1 + 2 * a2 * t + 3 * a3 * t ** 2
    def calc_dx2_dt2(t):
        return 2 * a2 + 6 * a3 * t
    a0 = x0
    a1 = 0
    T = 1.5 * abs(x0 - x_final) / v_max
    a2 = - 3 * (x0 - x_final) / (T ** 2)
    a3 = 2 * (x0 - x_final) / (T ** 3)
    if plot_res:
        assert sim_params is not None
        x = []
        x_dot = []
        x_dot_2 = []
        t = np.arange(start=0, stop=T, step=sim_params['dt'])
        for ti in t:
            x.append(calc_x(ti))
            x_dot.append(calc_dx_dt(ti))
            x_dot_2.append(calc_dx2_dt2(ti))
        plt.figure()
        h = plt.subplot(3, 1, 1)
        plt.plot(t, x)
        plt.ylabel('$x$')
        plt.grid(True)
        plt.subplot(3, 1, 2, sharex=h)
        plt.plot(t, x_dot)
        plt.grid(True), plt.ylabel('$dx/dt$')
        plt.subplot(3, 1, 3, sharex=h)
        plt.plot(t, x_dot_2)
        plt.grid(True), plt.ylabel('$dx/dt^2$'), plt.xlabel('t[sec]')
        plt.show()
    return [a0, a1, a2, a3]


def CalcO3PolyCoefV2(x0, x_dot_0, x_final, x_dot_max, plot_res=False, sim_params=None):
    """
    calculate coefficients for 3rd order polynomial
    x(t) = a0 + a1 * t + a2 * t^2 + a3 * t^3
    for given boundary conditions:
    x(t = 0) = x0
    x(t = T) = x_final
    d_dt[x(t = 0)] = x_dot_0
    d_dt[x(t = T)] = 0
    d2_dt2[x(t = t1)] = 0
    | d_dt[x(t1)] | = v_max
    return coefs = [a0, a1, a2, a3]
    """
    def calc_x(t):
        return a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3
    def calc_dx_dt(t):
        return a1 + 2 * a2 * t + 3 * a3 * t ** 2
    def calc_dx2_dt2(t):
        return 2 * a2 + 6 * a3 * t
    a0 = x0
    a1 = x_dot_0
    T_1 = 1.5 * (x0 - x_final) / (x_dot_0 + x_dot_max)
    T_2 = 1.5 * (x0 - x_final) / (x_dot_0 - x_dot_max)
    assert T_1 > 0 or T_2 > 0
    if T_1 <= 0:
        T = T_2
    elif T_2 <= 0:
        T = T_1
    elif T_1 > 0 and T_2 > 0:
        T = min(T_1, T_2)
    else:
        raise 'somethings wrong'
    a3 =  (2 * x0 + x_dot_0 * T - 2 * x_final) / (T ** 3)
    a2 = - (x_dot_0 + 3 * a3 * T ** 2) / (2 * T)
    if plot_res:
        assert sim_params is not None
        x = []
        x_dot = []
        x_dot_2 = []
        t = np.arange(start=0, stop=T, step=sim_params['dt'])
        for ti in t:
            x.append(calc_x(ti))
            x_dot.append(calc_dx_dt(ti))
            x_dot_2.append(calc_dx2_dt2(ti))
        plt.figure()
        h = plt.subplot(3, 1, 1)
        plt.plot(t, x)
        plt.ylabel('$x$')
        plt.grid(True)
        plt.subplot(3, 1, 2, sharex=h)
        plt.plot(t, x_dot)
        plt.grid(True), plt.ylabel('$dx/dt$')
        plt.subplot(3, 1, 3, sharex=h)
        plt.plot(t, x_dot_2)
        plt.grid(True), plt.ylabel('$dx/dt^2$'), plt.xlabel('t[sec]')
        plt.show()
    return [a0, a1, a2, a3]

if __name__ == "__main__":
    calc_desired_path('random_curvature', ds=0.1, plot_results=True)
