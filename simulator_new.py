import numpy as np
from math import cos, sin, sqrt, pi, tan
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 物理常数和仿真参数
class Constants:
    """物理常数和仿真参数"""
    GRAVITY = 9.8  # 重力加速度 (m/s^2)
    R_EARTH = 6356.766  # 地球半径 (km)
    RCZ = 287.0529  # 气体常数 (J/(kg·K))
    GN = 9.80665  # 标准重力 (m/s^2)
    S_REF = 0.146  # 参考面积 (m^2)
    MF = 490  # 最终质量 (kg)
    M_INTER = 10  # 中间时间 (s)

    # 导弹参数
    JX = 3.03  # x轴转动惯量 (kg·m^2)
    JY = 306.3  # y轴转动惯量 (kg·m^2)
    JZ = 306.3  # z轴转动惯量 (kg·m^2)
    D = 1.26  # 距离参数 (m)
    COEFF_CX = 0.3
    COEFF_CDZ = 0.082
    COEFF_CDY = 0.082
    COEFF_MWZ_JZ = -0.32
    COEFF_MWY_JY = -0.32
    COEFF_MWX_JX = 1.278
    COEFF_MDX_JX = 9641
    COEFF_MDZ_JZ = -89.34
    COEFF_MDY_JY = -89.34

    # 控制参数
    DELTA_Y_MAX = pi / 6
    DELTA_Z_MAX = pi / 6
    D_DELTA_Y = pi / 30
    D_DELTA_Z = pi / 30
    K = 6
    ALPHA_MAX = 35 * pi / 180
    BETA_MAX = ALPHA_MAX

    # Beta 参数
    BETA_PARAMS = {
        '11': (10, 12, 0.95, 0.1),
        '21': (10, 0.95, 0.1),
        '31': (8, 15, 0.95, 0.1),
        '41': (6, 0.95, 0.1),
        '51': (20, 23, 0.92, 0.1),
        '61': (7, 0.92, 0.1),
        '71': (30, 26, 0.97, 0.1),
        '81': (9, 0.97, 0.1),
        '91': (10, 12, 0.95, 0.1),
        '101': (7, 0.92, 0.1),
        '111': (8, 15, 0.95, 0.1),
        '121': (6, 0.95, 0.1),
    }

# 空气动力学系数表
AERO_TABLE = np.array([
    [0.2, 0.241, 0.11, 9.15],
    [0.78, 0.213, 0.136, 7.51],
    [0.94, 0.258, 0.135, 7.61],
    [1.07, 0.407, 0.109, 9.63],
    [1.32, 0.445, 0.108, 9.89],
    [1.61, 0.372, 0.115, 9.18],
    [2.43, 0.255, 0.121, 8.91],
    [3.5, 0.19, 0.134, 7.94],
    [5.0, 0.15, 0.154, 7.03],
    [6.1, 0.145, 0.16, 6.93],
])

class MissileSimulation:
    """导弹仿真类"""
    def __init__(self):
        self.t_end = 24.79  # 仿真时间 (s)
        self.t_inter = 0.01  # 时间步长 (s)
        self.n_step = int(np.ceil(self.t_end / self.t_inter)) + 1
        self.t_start1 = self.t_inter
        self.start_step1 = int(self.t_start1 / self.t_inter)
        self.crash1 = False

        # 初始化状态数组
        self.T1_pde = np.zeros((6, self.n_step))
        self.T1_pde_tilde = np.zeros((6, self.n_step))
        self.T1_alg = np.zeros((3, self.n_step))
        self.T1_alg_tilde = np.zeros((3, self.n_step))
        self.M1_pde = np.zeros((26, self.n_step))
        self.M1_pde_tilde = np.zeros((26, self.n_step))
        self.M1_alg = np.zeros((34, self.n_step))
        self.M1_alg_tilde = np.zeros((34, self.n_step))
        self.r1 = np.zeros(self.n_step)

        # 初始条件
        self.initial_condition = np.array([
            3471.31099154196, -0.763399365336313, -3.06176266787756, 80000, 75000, -10000,
            269.258240356725, 0, 0, 2000, 1000, -1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 734
        ])
        self.initialize_states()

    def initialize_states(self):
        """初始化目标和导弹状态"""
        thrust01 = 110 * 1000
        self.M1_alg[33, self.start_step1] = thrust01

        # 目标初始条件
        self.T1_pde[0:6, 0:self.start_step1 + 1] = self.initial_condition[0:6].reshape(6, 1)

        # 导弹初始条件
        self.M1_pde[0:26, 0:self.start_step1 + 1] = self.initial_condition[6:32].reshape(26, 1)

        # 初始速度分量
        vmx01 = self.M1_pde[0, self.start_step1] * cos(self.M1_pde[1, self.start_step1]) * cos(self.M1_pde[2, self.start_step1])
        vmy01 = self.M1_pde[0, self.start_step1] * sin(self.M1_pde[1, self.start_step1])
        vmz01 = -self.M1_pde[0, self.start_step1] * cos(self.M1_pde[1, self.start_step1]) * sin(self.M1_pde[2, self.start_step1])
        self.M1_alg[29:32, self.start_step1] = [vmx01, vmy01, vmz01]

        # 目标轨迹初始值
        xt01, yt01, zt01 = 80 * 1000, 75 * 1000, -10 * 1000
        vtx01, vty01, vtz01 = -2500, -2400, 200
        vt01, theta_t01, psi_t01 = initialize_target(vtx01, vty01, vtz01)
        self.T1_pde[0, self.start_step1] = vt01
        self.T1_pde[1, 0:self.start_step1 + 1] = theta_t01
        self.T1_pde[2, 0:self.start_step1 + 1] = psi_t01
        self.T1_pde[3:6, self.start_step1] = [xt01, yt01, zt01]

        self.r1[self.start_step1] = sqrt((xt01 - self.M1_pde[3, self.start_step1])**2 +
                                        (yt01 - self.M1_pde[4, self.start_step1])**2 +
                                        (zt01 - self.M1_pde[5, self.start_step1])**2)

    def run_simulation(self):
        """运行仿真主循环"""
        for i in range(self.start_step1, self.n_step - 1):
            # 代数方程
            self.T1_alg_tilde[:, i] = eq_alg_target(self.T1_pde[:, i])
            self.M1_alg_tilde[:, i] = eq_alg_missile(self.M1_alg[:, i], self.M1_pde[:, i], self.T1_alg_tilde[:, i], self.T1_pde[:, i], i, self.t_inter)
            # 预测步
            self.T1_pde_tilde[:, i] = self.T1_pde[:, i] + self.t_inter * eq_pde_target(self.T1_alg_tilde[:, i], self.T1_pde[:, i])
            self.M1_pde_tilde[:, i] = self.M1_pde[:, i] + self.t_inter * eq_pde_missile(self.M1_alg_tilde[:, i], self.M1_pde[:, i])
            # 代数方程
            self.T1_alg[:, i + 1] = eq_alg_target(self.T1_pde_tilde[:, i])
            self.M1_alg[:, i + 1] = eq_alg_missile(self.M1_alg_tilde[:, i], self.M1_pde_tilde[:, i], self.T1_alg[:, i + 1], self.T1_pde_tilde[:, i], i, self.t_inter)
            # 修正步
            self.T1_pde[:, i + 1] = self.T1_pde[:, i] + 0.5 * self.t_inter * (
                eq_pde_target(self.T1_alg[:, i + 1], self.T1_pde[:, i]) +
                eq_pde_target(self.T1_alg[:, i + 1], self.T1_pde_tilde[:, i]))
            self.M1_pde[:, i + 1] = self.M1_pde[:, i] + 0.5 * self.t_inter * (
                eq_pde_missile(self.M1_alg[:, i + 1], self.M1_pde[:, i]) +
                eq_pde_missile(self.M1_alg[:, i + 1], self.M1_pde_tilde[:, i]))

            if i > self.start_step1:
                            self.crash1, self.r1[i] = check_crash(self.T1_pde[:, i + 1], self.M1_pde[:, i + 1], self.r1[i - 1], i)
                            if self.crash1:
                                print(f"Simulation terminated at step {i} due to crash.")
                                break
        self.plot_results(i)

    def plot_results(self, last_step):
        """绘制仿真结果"""
        time = np.linspace(0, self.t_end, self.n_step)[:last_step + 1]

        # 3D 轨迹图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.T1_pde[3, :last_step + 1], self.T1_pde[4, :last_step + 1], self.T1_pde[5, :last_step + 1],
                label='Target', color='blue')
        ax.plot(self.M1_pde[3, :last_step + 1], self.M1_pde[4, :last_step + 1], self.M1_pde[5, :last_step + 1],
                label='Missile', color='red')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectory of Target and Missile')
        ax.legend()
        plt.show()

        # 定义变量名称
        t_pde_labels = ['vt', 'theta_t', 'psi_t', 'xt', 'yt', 'zt']
        m_pde_labels = ['vm', 'theta_m', 'psi_m', 'xm', 'ym', 'zm', 'alpha', 'beta',
                        'theta_var', 'gamma', 'wx', 'wy', 'wz', 'z11', 'z12', 'z21',
                        'z22', 'z31', 'z32', 'z41', 'z42', 'z51', 'z52', 'z61', 'z62', 'm_DD']

        # 第一张图：前 16 个变量 (T_pde 的 6 个 + M_pde 的前 10 个)
        fig = plt.figure(figsize=(15, 10))
        for i in range(6):  # Target 的 6 个变量
            plt.subplot(4, 4, i + 1)
            plt.plot(time, self.T1_pde[i, :last_step + 1], label=t_pde_labels[i], color='blue')
            plt.xlabel('Time (s)')
            plt.ylabel(t_pde_labels[i])
            plt.legend()
            plt.grid(True)
        for i in range(10):  # Missile 的前 10 个变量
            plt.subplot(4, 4, i + 7)
            plt.plot(time, self.M1_pde[i, :last_step + 1], label=m_pde_labels[i], color='red')
            plt.xlabel('Time (s)')
            plt.ylabel(m_pde_labels[i])
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.suptitle('First 16 Variables (T_pde: 6, M_pde: 10)', y=1.02)
        plt.savefig('plot1.png')
        plt.show()

        # 第二张图：后 16 个变量 (M_pde 的后 16 个)
        fig = plt.figure(figsize=(15, 10))
        for i in range(16):  # Missile 的后 16 个变量
            plt.subplot(4, 4, i + 1)
            plt.plot(time, self.M1_pde[i + 10, :last_step + 1], label=m_pde_labels[i + 10], color='red')
            plt.xlabel('Time (s)')
            plt.ylabel(m_pde_labels[i + 10])
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.suptitle('Last 16 Variables (M_pde: 16)', y=1.02)
        plt.savefig('plot2.png')
        plt.show()

# 工具函数
def initialize_target(vtx0, vty0, vtz0):
    """初始化目标速度和角度"""
    vt0 = sqrt(vtx0**2 + vty0**2 + vtz0**2)
    theta_t0 = np.arctan(vty0 / sqrt(vtx0**2 + vtz0**2))
    psi_t0 = (np.arctan(-vtz0 / vtx0) + pi if vtz0 < -0.1 else
              np.arctan(-vtz0 / vtx0) - pi if vtz0 > 0.1 else pi)
    return vt0, theta_t0, psi_t0

def check_crash(t_pde, m_pde, r_prime, i):
    """检查目标与导弹是否碰撞"""
    vm, theta_m, psi_m, xm, ym, zm = m_pde[0:6]
    vt, theta_t, psi_t, xt, yt, zt = t_pde[0:6]
    
    r = sqrt((xt - xm)**2 + (yt - ym)**2 + (zt - zm)**2)
    vx_relative = vt * cos(theta_t) * cos(psi_t) - vm * cos(theta_m) * cos(psi_m)
    vy_relative = vt * sin(theta_t) - vm * sin(theta_m)
    vz_relative = -vt * cos(theta_t) * sin(psi_t) + vm * cos(theta_m) * sin(psi_m)
    v_relative = sqrt(vx_relative**2 + vy_relative**2 + vz_relative**2)
    rx_relative, ry_relative, rz_relative = xt - xm, yt - ym, zt - zm
    
    cos_relative = (vx_relative * rx_relative + vy_relative * ry_relative + vz_relative * rz_relative) / (r * v_relative)
    cos_relative = np.clip(cos_relative, -1, 1)
    tbl_relative = r * sqrt(1 - cos_relative**2)

    if r > r_prime and r_prime < 20:
        print(f"Step {i}: TBL = {tbl_relative}")
        return True, r
    return False, r

def eq_pde_target(t_alg, t_pde):
    """目标微分方程"""
    dT_pde = np.zeros(6)
    vt, theta_t = t_pde[0], t_pde[1]
    at_axis, at_ynormal, at_znormal = -3 * Constants.GRAVITY, -2 * Constants.GRAVITY, -2 * Constants.GRAVITY
    vtx, vty, vtz = t_alg
    
    dT_pde[0] = at_axis  # dvt
    dT_pde[1] = at_ynormal / vt  # dtheta_t
    dT_pde[2] = -at_znormal / (vt * cos(theta_t))  # dpsi_t
    dT_pde[3:6] = [vtx, vty, vtz]  # dx, dy, dz
    return dT_pde

def eq_alg_target(t_pde):
    """目标代数方程"""
    vt, theta_t, psi_t = t_pde[0:3]
    return np.array([
        vt * cos(theta_t) * cos(psi_t),
        vt * sin(theta_t),
        -vt * cos(theta_t) * sin(psi_t)
    ])

def eq_pde_missile(m_alg, m_pde):
    """
    计算导弹的微分方程。

    参数:
        m_alg (np.ndarray): 导弹代数变量数组，大小为 (34,)
        m_pde (np.ndarray): 导弹微分变量数组，大小为 (26,)

    返回:
        np.ndarray: 导弹微分方程的导数数组，大小为 (26,)
    """
    # 初始化导数数组
    dm_pde = np.zeros(26)

    # 固定参数（从 Constants 类中获取更佳，但这里为独立性直接定义）
    JX = 3.03
    JY = 306.3
    JZ = 306.3
    COEFF_CX = 0.3
    COEFF_CDZ = 0.082
    COEFF_CDY = 0.082
    COEFF_MWZ_JZ = -0.32
    COEFF_MWY_JY = -0.32
    COEFF_MWX_JX = 1.278
    COEFF_MDX_JX = 9641
    COEFF_MDZ_JZ = -89.34
    COEFF_MDY_JY = -89.34
    GRAVITY = 9.8
    S_REF = 0.146

    # Beta 参数 (beta_xx, beta_xy, a_x, d_x)
    BETA_PARAMS = {
        '11': (10, 12, 0.95, 0.1),
        '31': (8, 15, 0.95, 0.1),
        '51': (20, 23, 0.92, 0.1),
        '71': (30, 26, 0.97, 0.1),
        '91': (10, 12, 0.95, 0.1),
        '111': (8, 15, 0.95, 0.1),
    }

    # 从 m_alg 中提取变量
    thrust = m_alg[33]
    e1, e3, e5, e7, e9, e11 = m_alg[0:6]
    f_alpha, f_beta, f_gamma = m_alg[6:9]
    fwx, fwy, fwz = m_alg[9:12]
    u_alpha, u_beta, u_gamma = m_alg[12:15]
    u_ogl, pu_ogl = m_alg[22:24]
    c_lafa, c_k, c_d = m_alg[24:27]
    rou = m_alg[28]
    vmx, vmy, vmz = m_alg[29:32]
    m_dot = m_alg[32]

    # 从 m_pde 中提取变量
    vm, theta_m = m_pde[0:2]
    z12, z22, z32, z42, z52, z62 = m_pde[14], m_pde[16], m_pde[18], m_pde[20], m_pde[22], m_pde[24]
    m_dd = m_pde[25]
    s = m_pde[6:13]  # [alpha, beta, theta, gamma, wx, wy, wz]
    u = m_alg[15:22]  # [delta_x, delta_y, delta_z, ...]

    # 计算中间变量
    dz11 = z12 - BETA_PARAMS['11'][0] * e1 + fwz + u_alpha
    dz12 = -BETA_PARAMS['11'][1] * fal(e1, BETA_PARAMS['11'][2], BETA_PARAMS['11'][3])
    dz21 = z22 - BETA_PARAMS['31'][0] * e3 + f_alpha + s[6]
    dz22 = -BETA_PARAMS['31'][1] * fal(e3, BETA_PARAMS['31'][2], BETA_PARAMS['31'][3])
    dz31 = z32 - BETA_PARAMS['51'][0] * e5 + fwx + u_gamma
    dz32 = -BETA_PARAMS['51'][1] * fal(e5, BETA_PARAMS['51'][2], BETA_PARAMS['51'][3])
    dz41 = z42 - BETA_PARAMS['71'][0] * e7 + f_gamma + s[4]
    dz42 = -BETA_PARAMS['71'][1] * fal(e7, BETA_PARAMS['71'][2], BETA_PARAMS['71'][3])
    dz51 = z52 - BETA_PARAMS['91'][0] * e9 + fwy + u_beta
    dz52 = -BETA_PARAMS['91'][1] * fal(e9, BETA_PARAMS['91'][2], BETA_PARAMS['91'][3])
    dz61 = z62 - BETA_PARAMS['111'][0] * e11 + f_beta + s[5]
    dz62 = -BETA_PARAMS['111'][1] * fal(e11, BETA_PARAMS['111'][2], BETA_PARAMS['111'][3])

    # 默认空气动力学系数（后续可动态调整）
    coeff_alpha = 0.23
    coeff_malpha_jz = 11.58
    coeff_beta = 0.23
    coeff_mbetay_jy = 11.58

    # 计算角速度和加速度
    dalpha = (s[6] - s[4] * cos(s[0]) * tan(s[1]) + s[5] * sin(s[0]) * tan(s[1]) -
              COEFF_CX * sin(s[0]) / cos(s[1]) + GRAVITY * sin(s[2]) * sin(s[0]) / (vm * cos(s[1])) -
              (coeff_alpha * s[0] + COEFF_CDZ * u[2]) * cos(s[0]) / cos(s[1]) -
              u[3] * cos(s[0]) / (m_dd * vm * cos(s[1])) +
              GRAVITY * cos(s[3]) * cos(s[2]) * cos(s[0]) / (vm * cos(s[1])) -
              thrust * sin(s[0]) / (m_dd * vm * cos(s[1])))

    dbeta = (s[4] * sin(s[0]) + s[5] * cos(s[0]) - COEFF_CX * cos(s[0]) * sin(s[1]) +
             GRAVITY * sin(s[2]) * cos(s[0]) * sin(s[1]) / vm +
             (coeff_alpha * s[0] + COEFF_CDZ * u[2]) * sin(s[0]) * sin(s[1]) +
             u[3] * sin(s[0]) * sin(s[1]) / (m_dd * vm) +
             (coeff_beta * s[1] + COEFF_CDY * u[1]) * cos(s[1]) +
             u[4] * cos(s[1]) / (m_dd * vm) + GRAVITY * cos(s[2]) * sin(s[3]) * cos(s[1]) / vm -
             thrust / (m_dd * vm))

    dtheta_var = s[5] * sin(s[3]) + s[6] * cos(s[3])
    dgamma = s[4] - tan(s[2]) * (s[5] * cos(s[3]) - s[6] * sin(s[3]))

    dwx = COEFF_MDX_JX * u[0] + COEFF_MWX_JX * s[4]
    dwy = (JZ - JY) / JY * s[6] * s[4] + coeff_mbetay_jy * s[1] + COEFF_MDY_JY * u[1] + COEFF_MWY_JY * s[5] + u[5] / JY
    dwz = (JX - JY) / JZ * s[4] * s[5] + coeff_malpha_jz * s[0] + COEFF_MDZ_JZ * u[2] + COEFF_MWZ_JZ * s[6] + u[6] / JZ

    dtheta_m = u_ogl / vm
    dpsi_m = -pu_ogl / (vm * cos(theta_m))
    dvm = (thrust * cos(s[0]) * cos(s[1]) - (c_d + c_k * c_lafa**2 * s[0]**2) * 0.5 * rou * vm**2 * S_REF) / m_dd - GRAVITY * sin(theta_m)
    dxm, dym, dzm = vmx, vmy, vmz
    dm = -m_dot if thrust != 0 else 0

    # 填充导数数组
    dm_pde[0] = dvm
    dm_pde[1] = dtheta_m
    dm_pde[2] = dpsi_m
    dm_pde[3] = dxm
    dm_pde[4] = dym
    dm_pde[5] = dzm
    dm_pde[6] = dalpha
    dm_pde[7] = dbeta
    dm_pde[8] = dtheta_var
    dm_pde[9] = dgamma
    dm_pde[10] = dwx
    dm_pde[11] = dwy
    dm_pde[12] = dwz
    dm_pde[13] = dz11
    dm_pde[14] = dz12
    dm_pde[15] = dz21
    dm_pde[16] = dz22
    dm_pde[17] = dz31
    dm_pde[18] = dz32
    dm_pde[19] = dz41
    dm_pde[20] = dz42
    dm_pde[21] = dz51
    dm_pde[22] = dz52
    dm_pde[23] = dz61
    dm_pde[24] = dz62
    dm_pde[25] = dm

    return dm_pde

def eq_alg_missile(m_alg, m_pde, t_alg, t_pde, i, t_inter):
    """
    计算导弹的代数方程。

    参数:
        m_alg (np.ndarray): 导弹代数变量数组，大小为 (34,)
        m_pde (np.ndarray): 导弹微分变量数组，大小为 (26,)
        t_alg (np.ndarray): 目标代数变量数组，大小为 (3,)
        t_pde (np.ndarray): 目标微分变量数组，大小为 (6,)
        i (int): 当前时间步索引
        t_inter (float): 时间步长 (s)

    返回:
        np.ndarray: 更新后的导弹代数变量数组，大小为 (34,)
    """
    # 固定参数
    JX = 3.03
    JY = 306.3
    JZ = 306.3
    D = 1.26
    COEFF_CX = 0.3
    COEFF_CDZ = 0.082
    COEFF_CDY = 0.082
    COEFF_MWZ_JZ = -0.32
    COEFF_MWY_JY = -0.32
    COEFF_MWX_JX = 1.278
    COEFF_MDX_JX = 9641
    COEFF_MDZ_JZ = -89.34
    COEFF_MDY_JY = -89.34
    DELTA_Y_MAX = np.pi / 6
    DELTA_Z_MAX = np.pi / 6
    D_DELTA_Y = np.pi / 30
    D_DELTA_Z = np.pi / 30
    K = 6
    ALPHA_MAX = 35 * np.pi / 180
    BETA_MAX = ALPHA_MAX
    S_REF = 0.146
    MF = 490
    M_INTER = 10
    GRAVITY = 9.8

    # Beta 参数
    BETA_PARAMS = {
        '21': (10, 0.95, 0.1),
        '41': (6, 0.95, 0.1),
        '61': (7, 0.92, 0.1),
        '81': (9, 0.97, 0.1),
        '101': (7, 0.92, 0.1),
        '121': (6, 0.95, 0.1),
    }

    # 提取变量
    inner_state = m_pde[6:13]  # [alpha, beta, theta, gamma, wx, wy, wz]
    thrust = m_alg[33]
    vtx, vty, vtz = t_alg
    xt, yt, zt = t_pde[3:6]
    command = np.array([0, m_alg[16], m_alg[17]])  # [_, delta_ym, delta_zm]
    vm, theta_m, psi_m = m_pde[0:3]
    xm, ym, zm = m_pde[3:6]
    z11, z21, z31, z41, z51, z61 = m_pde[13], m_pde[15], m_pde[17], m_pde[19], m_pde[21], m_pde[23]
    z12, z22, z32, z42, z52, z62 = m_pde[14], m_pde[16], m_pde[18], m_pde[20], m_pde[22], m_pde[24]
    m_dd = m_pde[25]

    m_dot = (734 - MF) / M_INTER

    # 计算空气动力学系数
    rou, v_sound = airdensity(ym)
    c_lafa = table_aerocoeff_ma_lookup(vm / v_sound, 4)
    c_k = table_aerocoeff_ma_lookup(vm / v_sound, 3)
    c_d = table_aerocoeff_ma_lookup(vm / v_sound, 2)
    c_beta = -c_lafa

    vmx = vm * cos(theta_m) * cos(psi_m)
    vmy = vm * sin(theta_m)
    vmz = -vm * cos(theta_m) * sin(psi_m)

    # 计算相对位置和速度
    pos_m = np.array([xm, ym, zm])
    pos_t = np.array([xt, yt, zt])
    vel_m = np.array([vmx, vmy, vmz])
    vel_t = np.array([vtx, vty, vtz])
    r_vec = pos_t - pos_m
    vr_vec = vel_t - vel_m
    vr_proj = np.dot(r_vec, vr_vec) * r_vec / np.linalg.norm(r_vec)**2
    vn = vr_vec - vr_proj
    tbl_tgo = np.linalg.norm(r_vec) / np.linalg.norm(vr_proj)
    tbl_vec = vn * tbl_tgo

    # 外环控制
    u_ogl = K * 2 * tbl_vec[1] / (tbl_tgo**2)
    u_ogl = np.clip(u_ogl, -40 * GRAVITY, 40 * GRAVITY)
    alpha_c = (u_ogl + GRAVITY * cos(theta_m)) * m_dd / (thrust + c_lafa * 0.5 * rou * vm**2 * S_REF)
    alpha_c = np.clip(alpha_c, -ALPHA_MAX, ALPHA_MAX)

    pu_ogl = K * 2 * tbl_vec[2] / (tbl_tgo**2)
    pu_ogl = np.clip(pu_ogl, -40 * GRAVITY, 40 * GRAVITY)
    beta_c = pu_ogl * m_dd / (-thrust * cos(alpha_c) + c_beta * 0.5 * rou * vm**2 * S_REF)
    beta_c = np.clip(beta_c, -BETA_MAX, BETA_MAX)

    # 内环控制
    coeff_alpha = 0.23 if inner_state[0] > 0 else -0.23
    coeff_malpha_jz = 11.58 if inner_state[0] > 0 else 10.43
    coeff_beta = 0.23 if inner_state[1] > 0 else -0.23
    coeff_mbetay_jy = 11.58 if inner_state[1] > 0 else 10.43

    e3 = z21 - inner_state[0]
    e4 = alpha_c - inner_state[0]
    u_alph = BETA_PARAMS['41'][0] * fal(e4, BETA_PARAMS['41'][1], BETA_PARAMS['41'][2])
    f_alpha = -COEFF_CX * sin(inner_state[0]) - (coeff_alpha * inner_state[0] + COEFF_CDZ * command[2]) * cos(inner_state[0])
    w_zc = u_alph - f_alpha - z22

    e1 = z11 - inner_state[6]
    e2 = w_zc - z11
    u_wc = BETA_PARAMS['21'][0] * fal(e2, BETA_PARAMS['21'][1], BETA_PARAMS['21'][2])
    fwz = coeff_malpha_jz * inner_state[0] + COEFF_MWZ_JZ * inner_state[6]
    u_alpha = u_wc - fwz - z12

    e7 = z41 - inner_state[4]
    gamma_c = 0
    e8 = gamma_c - inner_state[3]
    u_gam = BETA_PARAMS['81'][0] * fal(e8, BETA_PARAMS['81'][1], BETA_PARAMS['81'][2])
    f_gamma = -tan(inner_state[2]) * (inner_state[5] * cos(inner_state[3]) - inner_state[6] * sin(inner_state[3]))
    w_xc = u_gam - f_gamma - z42

    e5 = z31 - inner_state[4]
    e6 = w_xc - z31
    u_wxc = BETA_PARAMS['61'][0] * fal(e6, BETA_PARAMS['61'][1], BETA_PARAMS['61'][2])
    fwx = COEFF_MWX_JX * inner_state[4]
    u_gamma = u_wxc - fwx - z32

    e11 = z61 - inner_state[1]
    e12 = beta_c - inner_state[1]
    u_bet = BETA_PARAMS['121'][0] * fal(e12, BETA_PARAMS['121'][1], BETA_PARAMS['121'][2])
    f_beta = -COEFF_CX * sin(inner_state[1]) + (coeff_beta * inner_state[1] + COEFF_CDY * command[1]) * cos(inner_state[1])
    w_yc = u_bet - f_beta - z62

    e9 = z51 - inner_state[5]
    e10 = w_yc - z51
    u_wyc = BETA_PARAMS['101'][0] * fal(e10, BETA_PARAMS['101'][1], BETA_PARAMS['101'][2])
    fwy = coeff_mbetay_jy * inner_state[1] + COEFF_MWY_JY * inner_state[5]
    u_beta = u_wyc - fwy - z52

    # 计算控制输入
    delta_x = u_gamma / COEFF_MDX_JX
    delta_zn = u_alpha / COEFF_MDZ_JZ
    delta_zm = (DELTA_Z_MAX - D_DELTA_Z if delta_zn > DELTA_Z_MAX else
                -DELTA_Z_MAX + D_DELTA_Z if delta_zn < -DELTA_Z_MAX else delta_zn)
    ty = (u_alpha - COEFF_MDZ_JZ * delta_zm) / (D / JZ)
    delta_yn = u_beta / COEFF_MDY_JY
    delta_ym = (DELTA_Y_MAX - D_DELTA_Y if delta_yn > DELTA_Y_MAX else
                -DELTA_Y_MAX + D_DELTA_Y if delta_yn < -DELTA_Y_MAX else delta_yn)
    tz = (u_beta - COEFF_MDY_JY * delta_ym) / (D / JY)
    mzt = ty * D
    myt = tz * D

    # 更新推力
    if (i - 1) * t_inter >= M_INTER:
        thrust = 0

    # 更新 m_alg
    m_alg[0] = e1
    m_alg[1] = e3
    m_alg[2] = e5
    m_alg[3] = e7
    m_alg[4] = e9
    m_alg[5] = e11
    m_alg[6] = f_alpha
    m_alg[7] = f_beta
    m_alg[8] = f_gamma
    m_alg[9] = fwx
    m_alg[10] = fwy
    m_alg[11] = fwz
    m_alg[12] = u_alpha
    m_alg[13] = u_beta
    m_alg[14] = u_gamma
    m_alg[15] = delta_x
    m_alg[16] = delta_ym
    m_alg[17] = delta_zm
    m_alg[18] = ty
    m_alg[19] = tz
    m_alg[20] = myt
    m_alg[21] = mzt
    m_alg[22] = u_ogl
    m_alg[23] = pu_ogl
    m_alg[24] = c_lafa
    m_alg[25] = c_k
    m_alg[26] = c_d
    m_alg[27] = c_beta
    m_alg[28] = rou
    m_alg[29] = vmx
    m_alg[30] = vmy
    m_alg[31] = vmz
    m_alg[32] = m_dot
    m_alg[33] = thrust

    return m_alg

def airdensity(h_km_in):
    """
    根据高度计算空气密度和声速。

    参数:
        h_km_in (float): 输入高度 (m)

    返回:
        tuple: (空气密度 rou, 声速 v_sound)
    """
    R_EARTH = 6356.766  # 地球半径 (km)
    RCZ = 287.0529  # 气体常数 (J/(kg·K))
    GN = 9.80665  # 标准重力 (m/s^2)

    h_km = h_km_in / 1000
    h = R_EARTH * h_km / (R_EARTH + h_km)
    h = max(h, -2)  # 限制最低高度

    if -2 <= h < 0:
        ts = 301.15 - 6.5 * (h + 2)
        ps = 13029.3 * GN * (1 - 2.158393 * 0.01 * (h + 2))**5.255880
    elif 0 <= h < 11:
        ts = 288.15 - 6.5 * h
        ps = 10332.3 * GN * (1 - 2.25577 * 0.01 * h)**5.255880
    elif 11 <= h < 20:
        ts = 216.65
        ps = 2307.83 * GN * np.exp(-0.1576885 * (h - 11))
    elif 20 <= h < 32:
        ts = 216.65 + (h - 20)
        ps = 558.28 * GN * (1 + 4.61574 * 0.001 * (h - 20))**(-34.16322)
    elif 32 <= h < 47:
        ts = 228.65 + 2.8 * (h - 32)
        ps = 88.51 * GN * (1 + 1.224579 * 0.01 * (h - 32))**(-12.20115)
    elif 47 <= h < 51:
        ts = 270.65
        ps = 11.31 * GN * np.exp(-0.126227 * (h - 47))
    elif 51 <= h < 71:
        ts = 270.65 - 2.8 * (h - 51)
        ps = 6.83 * GN * (1 - 1.034546 * 0.01 * (h - 51))**12.20115
    else:  # h >= 71
        ts = 214.65 - 2 * (h - 71)
        ps = 0.4 * GN * (1 - 9.317494 * 0.001 * (h - 71))**17.08161

    rou = ps / ts / RCZ
    v_sound = 20.0468 * np.sqrt(ts)
    return rou, v_sound

def table_aerocoeff_ma_lookup(t, n):
    """
    根据马赫数查找空气动力学系数。

    参数:
        t (float): 马赫数
        n (int): 查找的列号 (MATLAB 风格，从 1 开始: 2=C_D, 3=C_K, 4=C_Lafa)

    返回:
        float: 插值得到的空气动力学系数
    """
    # 空气动力学系数表 [Ma, C_D, C_K, C_Lafa]
    AERO_TABLE = np.array([
        [0.2, 0.241, 0.11, 9.15],
        [0.78, 0.213, 0.136, 7.51],
        [0.94, 0.258, 0.135, 7.61],
        [1.07, 0.407, 0.109, 9.63],
        [1.32, 0.445, 0.108, 9.89],
        [1.61, 0.372, 0.115, 9.18],
        [2.43, 0.255, 0.121, 8.91],
        [3.5, 0.19, 0.134, 7.94],
        [5.0, 0.15, 0.154, 7.03],
        [6.1, 0.145, 0.16, 6.93],
    ])

    n -= 1  # 转换为 Python 索引 (0-based)
    if t >= AERO_TABLE[-1, 0]:
        return AERO_TABLE[-1, n]
    elif t <= AERO_TABLE[0, 0]:
        return AERO_TABLE[0, n]

    n0, n1 = 0, len(AERO_TABLE) - 1
    while n1 - n0 > 1:
        i_inter = int(np.round((n1 - n0) / 2)) + n0
        if t < AERO_TABLE[i_inter, 0]:
            n1 = i_inter
        else:
            n0 = i_inter

    x1, x2 = AERO_TABLE[n0, 0], AERO_TABLE[n1, 0]
    y1, y2 = AERO_TABLE[n0, n], AERO_TABLE[n1, n]
    return y1 + (y2 - y1) / (x2 - x1) * (t - x1)

def fal(x, a, delta):
    """非线性函数 fal"""
    return abs(x)**a * np.sign(x) if abs(x) > delta else x / (delta**(a - 1))

# 主程序
if __name__ == "__main__":
    sim = MissileSimulation()
    sim.run_simulation()
    print("Simulation completed.")