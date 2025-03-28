import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 参数设置
m = 1.0      # 小球质量 (kg)
g = 9.8      # 重力加速度 (m/s^2)
k = 0.1      # 空气阻力系数
ks = 2.0     # 弹簧刚度 (N/m)
xe = 0.0     # 弹簧平衡位置 (m)
dt = 0.01    # 时间步长 (s)
n_steps = 1000  # 仿真步数

# 初始条件
x0, y0, z0 = 0.0, 0.0, 0.0  # 初始位置
vx0, vy0, vz0 = 5.0, 5.0, 10.0  # 初速度

# 定义数组
state_pde = np.zeros((6, n_steps))  # 微分变量 [x, y, z, vx, vy, vz]
state_alg = np.zeros((3, n_steps))  # 代数变量 [ax, ay, az]
state_pde[:, 0] = [x0, y0, z0, vx0, vy0, vz0]

# 代数方程：计算加速度
def eq_alg(state_pde):
    x, y, z, vx, vy, vz = state_pde
    v = np.sqrt(vx**2 + vy**2 + vz**2)  # 速度大小
    ax = -ks/m * (x - xe) - k/m * v * vx
    ay = -k/m * v * vy
    az = -g - k/m * v * vz
    return np.array([ax, ay, az])

# 微分方程：计算位置和速度的变化率
def eq_pde(state_alg, state_pde):
    x, y, z, vx, vy, vz = state_pde
    ax, ay, az = state_alg
    return np.array([vx, vy, vz, ax, ay, az])

# 预测-修正法仿真
for i in range(n_steps - 1):
    # 当前状态
    current_pde = state_pde[:, i]
    
    # 代数方程：计算当前加速度
    state_alg[:, i] = eq_alg(current_pde)
    
    # 预测步
    state_pde_tilde = current_pde + dt * eq_pde(state_alg[:, i], current_pde)
    
    # 代数方程：计算预测状态下的加速度
    state_alg_tilde = eq_alg(state_pde_tilde)
    
    # 修正步
    state_pde[:, i + 1] = current_pde + 0.5 * dt * (
        eq_pde(state_alg[:, i], current_pde) + 
        eq_pde(state_alg_tilde, state_pde_tilde)
    )

# 提取位置和速度
x, y, z = state_pde[0, :], state_pde[1, :], state_pde[2, :]
vx, vy, vz = state_pde[3, :], state_pde[4, :], state_pde[5, :]
t = np.arange(n_steps) * dt

# 绘制 3D 轨迹图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label='Trajectory')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Trajectory of the Ball')
ax.legend()
plt.show()

# 绘制 6 个自由度随时间变化图
fig, axs = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle('Position and Velocity vs Time')

# 位置
axs[0, 0].plot(t, x, 'b-', label='x')
axs[0, 0].set_title('X Position')
axs[0, 0].set_xlabel('Time (s)')
axs[0, 0].set_ylabel('X (m)')
axs[0, 0].legend()

axs[1, 0].plot(t, y, 'g-', label='y')
axs[1, 0].set_title('Y Position')
axs[1, 0].set_xlabel('Time (s)')
axs[1, 0].set_ylabel('Y (m)')
axs[1, 0].legend()

axs[2, 0].plot(t, z, 'r-', label='z')
axs[2, 0].set_title('Z Position')
axs[2, 0].set_xlabel('Time (s)')
axs[2, 0].set_ylabel('Z (m)')
axs[2, 0].legend()

# 速度
axs[0, 1].plot(t, vx, 'b-', label='vx')
axs[0, 1].set_title('X Velocity')
axs[0, 1].set_xlabel('Time (s)')
axs[0, 1].set_ylabel('Vx (m/s)')
axs[0, 1].legend()

axs[1, 1].plot(t, vy, 'g-', label='vy')
axs[1, 1].set_title('Y Velocity')
axs[1, 1].set_xlabel('Time (s)')
axs[1, 1].set_ylabel('Vy (m/s)')
axs[1, 1].legend()

axs[2, 1].plot(t, vz, 'r-', label='vz')
axs[2, 1].set_title('Z Velocity')
axs[2, 1].set_xlabel('Time (s)')
axs[2, 1].set_ylabel('Vz (m/s)')
axs[2, 1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()