import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 参数范围
min_values = torch.tensor([1, -1, 0, 0, -5, 0, -10.0], dtype=torch.float32)  # [m, x0, y0, z0, vx0, vy0, vz0]
max_values = torch.tensor([1, 1, 7, -80, 5, 5, 10.0], dtype=torch.float32)     # [m, x0, y0, z0, vx0, vy0, vz0]
# min_values = torch.tensor([1, -2, 0, 0, -10, 0, -20.0], dtype=torch.float32)  # [m, x0, y0, z0, vx0, vy0, vz0]
# max_values = torch.tensor([1, 2, 14, -100, 10, 10, 20.0], dtype=torch.float32)     # [m, x0, y0, z0, vx0, vy0, vz0]
# min_values = torch.tensor([1, -5, -5, 0, -20, 0, -40.0], dtype=torch.float32)  # [m, x0, y0, z0, vx0, vy0, vz0]
# max_values = torch.tensor([1, 5, 20, -100, 20, 20, 40.0], dtype=torch.float32)     # [m, x0, y0, z0, vx0, vy0, vz0]

# 模型定义
class ODEFunc(nn.Module):
    def __init__(self, state_dim=6, hidden_dim=2048):  # state_dim 改为 6
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 512),
            nn.Tanh(),
            nn.Linear(512, state_dim)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)

# 仿真器
class BallSimulatorGPU:
    def __init__(self, t_end, n_step, batch_size, device):
        self.t_end = t_end
        self.n_step = n_step
        self.dt = t_end / (n_step - 1)
        self.batch_size = batch_size
        self.device = device
        self.g = 9.8
        self.k = 0.1
        self.ks = 20.0
        self.xe = 0.0

    def simulate(self, params):
        m = params[:, 0:1]
        state_pde = torch.zeros(self.batch_size, self.n_step, 6, device=self.device)
        state_pde[:, 0, :3] = params[:, 1:4]
        state_pde[:, 0, 3:] = params[:, 4:7]
        state_alg = torch.zeros(self.batch_size, self.n_step, 3, device=self.device)

        def eq_alg(state_pde, m):
            x, y, z, vx, vy, vz = state_pde[:, 0], state_pde[:, 1], state_pde[:, 2], state_pde[:, 3], state_pde[:, 4], state_pde[:, 5]
            v = torch.sqrt(vx**2 + vy**2 + vz**2)
            x = x.unsqueeze(1)
            v = v.unsqueeze(1)
            vx = vx.unsqueeze(1)
            vy = vy.unsqueeze(1)
            vz = vz.unsqueeze(1)
            ax = -self.ks / m * (x - self.xe) - self.k / m * v * vx
            ay = -self.k / m * v * vy
            az = -self.g - self.k / m * v * vz
            return torch.cat([ax, ay, az], dim=1)

        def eq_pde(state_alg, state_pde):
            return torch.cat([state_pde[:, 3:], state_alg], dim=1)

        for i in range(self.n_step - 1):
            current_pde = state_pde[:, i, :]
            state_alg[:, i, :] = eq_alg(current_pde, m)
            state_pde_tilde = current_pde + self.dt * eq_pde(state_alg[:, i, :], current_pde)
            state_alg_tilde = eq_alg(state_pde_tilde, m)
            state_pde[:, i + 1, :] = current_pde + 0.5 * self.dt * (
                eq_pde(state_alg[:, i, :], current_pde) + eq_pde(state_alg_tilde, state_pde_tilde)
            )

        return state_pde

# 测试函数
def test_model(model_path, output_dir, num_samples=1, t_end=0.1, n_step=10, device='cpu'):
    device = torch.device(device)
    model = ODEFunc(state_dim=6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")

    simulator = BallSimulatorGPU(t_end=t_end, n_step=n_step, batch_size=num_samples, device=device)

    # 使用固定的初始条件 [m, x0, y0, z0, vx0, vy0, vz0]
    fixed_params = torch.tensor([[1.0, 0.0, 0.0, 0.0, 5.0, 5.0, 10.0]], dtype=torch.float32, device=device)
    test_params = fixed_params.repeat(num_samples, 1)
    print(f"Using fixed initial condition: {fixed_params.tolist()}")

    with torch.no_grad():
        true_y = simulator.simulate(test_params)
    
    min_vals = min_values.to(device)
    max_vals = max_values.to(device)
    true_y0 = test_params[:, 1:]
    true_y0_norm = (true_y0 - min_vals[1:]) / (max_vals[1:] - min_vals[1:])
    true_y_norm = (true_y - min_vals[1:]) / (max_vals[1:] - min_vals[1:])

    t = torch.linspace(0., t_end, n_step).to(device)
    with torch.no_grad():
        pred_y_norm = odeint(model, true_y0_norm, t, method='rk4')
        pred_y_norm = pred_y_norm.permute(1, 0, 2)

    pred_y = pred_y_norm * (max_vals[1:] - min_vals[1:]) + min_vals[1:]
    true_y = true_y

    visualize_results(true_y, pred_y, t, output_dir)

# 可视化函数
def visualize_results(true_y, pred_y, t, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    true_y = true_y.cpu().numpy()
    pred_y = pred_y.cpu().numpy()
    t = t.cpu().numpy()

    for i in range(true_y.shape[0]):
        true_sample = true_y[i]
        pred_sample = pred_y[i]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(true_sample[:, 0], true_sample[:, 1], true_sample[:, 2], 'g-', label='True Trajectory')
        ax.plot(pred_sample[:, 0], pred_sample[:, 1], pred_sample[:, 2], 'b--', label='Predicted Trajectory')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'3D Trajectory Comparison (Sample {i+1})')
        ax.legend()
        plt.savefig(f'{output_dir}/trajectory_3d_sample_{i+1}.png')
        plt.close()

        fig, axs = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle(f'Position and Velocity vs Time (Sample {i+1})')

        axs[0, 0].plot(t, true_sample[:, 0], 'g-', label='True x')
        axs[0, 0].plot(t, pred_sample[:, 0], 'b--', label='Pred x')
        axs[0, 0].set_title('X Position')
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('X (m)')
        axs[0, 0].legend()

        axs[1, 0].plot(t, true_sample[:, 1], 'g-', label='True y')
        axs[1, 0].plot(t, pred_sample[:, 1], 'b--', label='Pred y')
        axs[1, 0].set_title('Y Position')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('Y (m)')
        axs[1, 0].legend()

        axs[2, 0].plot(t, true_sample[:, 2], 'g-', label='True z')
        axs[2, 0].plot(t, pred_sample[:, 2], 'b--', label='Pred z')
        axs[2, 0].set_title('Z Position')
        axs[2, 0].set_xlabel('Time (s)')
        axs[2, 0].set_ylabel('Z (m)')
        axs[2, 0].legend()

        axs[0, 1].plot(t, true_sample[:, 3], 'g-', label='True vx')
        axs[0, 1].plot(t, pred_sample[:, 3], 'b--', label='Pred vx')
        axs[0, 1].set_title('X Velocity')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Vx (m/s)')
        axs[0, 1].legend()

        axs[1, 1].plot(t, true_sample[:, 4], 'g-', label='True vy')
        axs[1, 1].plot(t, pred_sample[:, 4], 'b--', label='Pred vy')
        axs[1, 1].set_title('Y Velocity')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('Vy (m/s)')
        axs[1, 1].legend()

        axs[2, 1].plot(t, true_sample[:, 5], 'g-', label='True vz')
        axs[2, 1].plot(t, pred_sample[:, 5], 'b--', label='Pred vz')
        axs[2, 1].set_title('Z Velocity')
        axs[2, 1].set_xlabel('Time (s)')
        axs[2, 1].set_ylabel('Vz (m/s)')
        axs[2, 1].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'{output_dir}/freedom_sample_{i+1}.png')
        plt.close()

    print(f"Visualization saved to {output_dir}")

if __name__ == "__main__":
    model_path = './output/model_full_tr2500_bs50_niters10000.pth'  # 替换为你的模型路径
    output_dir = './test_output_fixed'
    num_samples = 1  # 只测试一个样本
    t_end = 10
    n_step = 1000
    device = 'cpu'

    test_model(model_path, output_dir, num_samples=num_samples, t_end=t_end, n_step=n_step, device=device)