import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import argparse
import os

# 参数范围
min_values = torch.tensor([1, -1, 0, 0, -5, 0, -10.0], dtype=torch.float32)  # [m, x0, y0, z0, vx0, vy0, vz0]
max_values = torch.tensor([1, 1, 7, -80, 5, 5, 10.0], dtype=torch.float32)     # [m, x0, y0, z0, vx0, vy0, vz0]

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
        m = params[:, 0:1]  # (batch_size, 1)
        state_pde = torch.zeros(self.batch_size, self.n_step, 6, device=self.device)
        state_pde[:, 0, :3] = params[:, 1:4]
        state_pde[:, 0, 3:] = params[:, 4:7]
        state_alg = torch.zeros(self.batch_size, self.n_step, 3, device=self.device)

        def eq_alg(state_pde, m):
            x, y, z, vx, vy, vz = state_pde[:, 0], state_pde[:, 1], state_pde[:, 2], state_pde[:, 3], state_pde[:, 4], state_pde[:, 5]
            v = torch.sqrt(vx**2 + vy**2 + vz**2)
            # 显式调整维度以避免广播问题
            x = x.unsqueeze(1)  # (batch_size, 1)
            v = v.unsqueeze(1)  # (batch_size, 1)
            vx = vx.unsqueeze(1)
            vy = vy.unsqueeze(1)
            vz = vz.unsqueeze(1)
            ax = -self.ks / m * (x - self.xe) - self.k / m * v * vx  # (batch_size, 1)
            ay = -self.k / m * v * vy  # (batch_size, 1)
            az = -self.g - self.k / m * v * vz  # (batch_size, 1)
            return torch.cat([ax, ay, az], dim=1)  # (batch_size, 3)

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

# 数据生成函数
def generate_simulation_data(simulator, num_samples, device):
    print("=== Starting Simulation Data Generation ===")
    print(f"Generating {num_samples} samples on {device}...")
    samples = torch.rand(num_samples, 7, device=device) * (max_values - min_values).to(device) + min_values.to(device)
    with torch.no_grad():
        state_pde = simulator.simulate(samples)
    simulation_results = state_pde
    true_y = simulation_results.reshape(-1, 6)
    true_y0 = samples
    print("=== Simulation Data Generated ===")
    print(f"true_y shape: {true_y.shape}")
    print(f"true_y0 shape: {true_y0.shape}")
    return true_y, true_y0

# 模型定义（调整状态维度）
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

# 配置类（调整状态维度）
class Config:
    def __init__(self):
        self.data_size = 10
        self.t_end = 0.1
        self.total_init_rows = 10000  # 减少样本量以加快测试
        self.state_dim = 6  # 斜抛小球的 6 个自由度
        self.batch_size = 10
        self.niters = 2000
        self.test_freq = 200
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.optimizer_type = 'AdamW'
        self.model_type = 'full'
        self.viz = False
        self.gpu = 0
        self.use_gpu = torch.cuda.is_available()
        self.output_dir_base = './output'
        self.model_save_path = 'model.pth'
        self.device = torch.device(f'cuda:{self.gpu}' if self.use_gpu else 'cpu')

    def update_from_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        self.use_gpu = torch.cuda.is_available() and self.gpu >= 0
        self.device = torch.device(f'cuda:{self.gpu}' if self.use_gpu else 'cpu')
        self.total_sim_rows = self.total_init_rows * self.data_size
        self.num_simulations = self.total_init_rows
        self.output_dir_full = f'{self.output_dir_base}/{self.model_type}_png'
        self.model_save_path = f'{self.output_dir_base}/model_{self.model_type}.pth'
        os.makedirs(self.output_dir_full, exist_ok=True)
        if self.use_gpu:
            self.viz = False
            print("Running on GPU, visualization disabled.")
        
        print("=== Configuration ===")
        print(f"Model type: {self.model_type}")
        print(f"Simulation end time (t_end): {self.t_end}")
        print(f"Number of time steps (data_size): {self.data_size}")
        print(f"Total initial samples: {self.total_init_rows}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of iterations: {self.niters}")
        print(f"Learning rate: {self.lr}")
        print(f"Optimizer: {self.optimizer_type}")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir_base}")

# 训练类（略作调整以适配新状态维度）
class ODETrainer:
    def __init__(self, config, true_y, true_y0, min_vals, max_vals, model_type, total_init_rows, batch_size, niters):
        print("=== Initializing ODETrainer ===")
        self.config = config
        self.true_y = true_y
        self.true_y0 = true_y0[:, 1:]  # 只取 [x0, y0, z0, vx0, vy0, vz0]，忽略 m
        self.min_vals = min_vals[1:]   # 调整维度
        self.max_vals = max_vals[1:]
        self.t = torch.linspace(0., config.t_end, config.data_size).to(config.device)
        self.model = ODEFunc(config.state_dim).to(config.device)
        self.optimizer = self._get_optimizer()
        self.train_loss_history = []
        self.model_type = model_type
        self.total_init_rows = total_init_rows
        self.batch_size = batch_size
        self.niters = niters
        print(f"Training time steps: {self.t.shape}")
        print(f"Model moved to device: {self.config.device}")

    def _get_optimizer(self):
        if self.config.optimizer_type == 'AdamW':
            return optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        elif self.config.optimizer_type == 'RMSprop':
            return optim.RMSprop(self.model.parameters(), lr=self.config.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")

    def get_batch(self):
        b = torch.randperm(self.config.num_simulations, device=self.config.device)[:self.config.batch_size] * self.config.data_size
        batch_y0 = self.true_y[b]
        batch_t = self.t
        offsets = torch.arange(self.config.data_size, device=self.config.device)
        batch_y = self.true_y[b.unsqueeze(1) + offsets]
        return batch_y0, batch_t, batch_y

    def denormalize(self, data):
        min_vals = self.min_vals.cpu().numpy()
        max_vals = self.max_vals.cpu().numpy()
        denormalized_data = data * (max_vals - min_vals) + min_vals
        return torch.tensor(denormalized_data, dtype=torch.float32).to(self.config.device)

    def train(self):
        print("=== Starting Training ===")
        loss_fn = nn.MSELoss()
        loss_history = torch.zeros(self.config.niters, device=self.config.device)
        
        # 记录训练总开始时间和第一次迭代时间
        total_start_time = time.time()
        first_iter_time = None
        
        for itr in range(self.config.niters):
            iter_start_time = time.time()
            # 重置显存峰值统计（仅GPU）
            if self.config.use_gpu:
                torch.cuda.reset_peak_memory_stats()
            
            self.optimizer.zero_grad()
            batch_y0, batch_t, batch_y = self.get_batch()
            pred_y = odeint(self.model, batch_y0, batch_t, method='rk4')
            pred_y = pred_y.transpose(0, 1)
            loss = loss_fn(pred_y, batch_y)
            loss_history[itr] = loss
            self.train_loss_history.append((itr, loss.item()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 显存监控与清理（仅GPU）
            if self.config.use_gpu:
                # 获取显存统计信息
                current_mem = torch.cuda.memory_allocated(self.config.device) / 1024**2  # MB
                peak_mem = torch.cuda.max_memory_allocated(self.config.device) / 1024**2
                
                # 按频率打印显存信息
                if itr % self.config.test_freq == 0 or itr == 1:
                    print(f'Iter {itr:04d} | Train Loss {loss.item():.6f} | '
                        f'GPU Mem: Current {current_mem:.2f}MB / Peak {peak_mem:.2f}MB')
                
                # 释放未使用的显存缓存
                torch.cuda.empty_cache()

            # 计算并打印第一次迭代的时间估计
            if itr == 0:
                first_iter_time = time.time() - total_start_time
                estimated_total = first_iter_time * self.config.niters
                print(f"First iteration took {first_iter_time:.2f} seconds.")
                print(f"Estimated total training time: {estimated_total:.2f} seconds | {estimated_total / 60:.2f} minutes | {estimated_total / 3600:.2f} hours")

            # 原有的日志和可视化逻辑
            if itr % self.config.test_freq == 0 or itr == 1:
                with torch.no_grad():
                    idx = torch.randperm(self.config.num_simulations, device=self.config.device)
                    new_true_y0 = self.true_y0[idx]
                    pred_y = odeint(self.model, new_true_y0[0], self.t, method='rk4').reshape(self.config.data_size, self.config.state_dim)
                    y_idx = idx[0] * self.config.data_size
                    new_true_y = self.true_y[y_idx:y_idx + self.config.data_size]
                    del idx, new_true_y0
                    torch.cuda.empty_cache()
                    print(f'Iter {itr:04d}/{self.config.niters} | Train Loss {loss.item():.6f}')
                    denorm_pred_y = self.denormalize(pred_y.cpu().numpy())
                    denorm_true_y = self.denormalize(new_true_y.cpu().numpy())
                    self.visualize_trajectory(denorm_true_y, denorm_pred_y, itr)
                    self.plot_loss_history(itr)

        # 保存模型和损失
        model_save_dir = self.config.output_dir_base
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, f"model_{self.model_type}_tr{self.total_init_rows}_bs{self.batch_size}_niters{self.niters}.pth")
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

        # loss_history_cpu = loss_history.cpu().numpy()
        loss_history_cpu = loss_history.detach().cpu().numpy()
        loss_save_path = os.path.join(model_save_dir, f"loss_history_{self.model_type}_tr{self.total_init_rows}_bs{self.batch_size}_niters{self.niters}.txt")
        with open(loss_save_path, 'w') as f:
            f.write("Iteration\tLoss\n")
            for i, loss_val in enumerate(loss_history_cpu):
                f.write(f"{i + 1}\t{loss_val:.6f}\n")
        print(f"Loss history saved to {loss_save_path}")

        # 计算并打印实际总训练时间
        total_time = time.time() - total_start_time
        print(f"=== Training Completed ===")
        print(f"Actual total training time: {total_time:.2f} seconds | {total_time / 60:.2f} minutes | {total_time / 3600:.2f} hours")

    def visualize_trajectory(self, true_y, pred_y, itr):
        """绘制真实和预测的 3D 轨迹及 6 个自由度随时间变化图（GPU兼容版）"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # 确保数据在CPU并转换为numpy（带梯度分离）
        def to_numpy(tensor):
            return tensor.detach().cpu().numpy()

        # 转换所有输入张量
        true_y = to_numpy(true_y)  # (data_size, 6)
        pred_y = to_numpy(pred_y)  # (data_size, 6)
        t = to_numpy(self.t)       # 时间序列

        # 3D 轨迹图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(true_y[:, 0], true_y[:, 1], true_y[:, 2], 'g-', label='True Trajectory')
        ax.plot(pred_y[:, 0], pred_y[:, 1], pred_y[:, 2], 'b--', label='Predicted Trajectory')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'3D Trajectory Comparison (Iter {itr:04d})')
        ax.legend()
        plt.savefig(f'{self.config.output_dir_full}/trajectory_3d_{itr:04d}.png')
        plt.close()

        # 6 个自由度随时间变化图
        fig, axs = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle(f'Position and Velocity vs Time (Iter {itr:04d})')

        # 位置可视化
        for i, (dim, label) in enumerate(zip(['X', 'Y', 'Z'], ['x', 'y', 'z'])):
            axs[i, 0].plot(t, true_y[:, i], 'g-', label=f'True {label}')
            axs[i, 0].plot(t, pred_y[:, i], 'b--', label=f'Pred {label}')
            axs[i, 0].set_title(f'{dim} Position')
            axs[i, 0].set_xlabel('Time (s)')
            axs[i, 0].set_ylabel(f'{dim} (m)')
            axs[i, 0].legend()

        # 速度可视化
        for i, (dim, label) in enumerate(zip(['X', 'Y', 'Z'], ['vx', 'vy', 'vz'])):
            axs[i, 1].plot(t, true_y[:, i+3], 'g-', label=f'True {label}')
            axs[i, 1].plot(t, pred_y[:, i+3], 'b--', label=f'Pred {label}')
            axs[i, 1].set_title(f'{dim} Velocity')
            axs[i, 1].set_xlabel('Time (s)')
            axs[i, 1].set_ylabel(f'{dim} (m/s)')
            axs[i, 1].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'{self.config.output_dir_full}/freedom_{itr:04d}.png')
        plt.close()

    def plot_loss_history(self, itr):
        """绘制并更新损失曲线（GPU兼容版）"""
        import matplotlib.pyplot as plt

        # 确保数据在CPU
        train_iters, train_losses = zip(*[
            (i, loss.detach().cpu().numpy()) if isinstance(loss, torch.Tensor) else (i, loss)
            for i, loss in self.train_loss_history
        ])

        plt.figure(figsize=(10, 6))
        plt.plot(train_iters, train_losses, 'b-', label='Train Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss History')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{self.config.output_dir_full}/loss_history_{itr:04d}.png')
        plt.close()

# 主程序
if __name__ == "__main__":
    print("=== Program Started ===")
    parser = argparse.ArgumentParser('ODE Training with Ball Simulation')
    parser.add_argument('--model_type', type=str, choices=['full'], default='full', help='Model type: full only')
    parser.add_argument('--data_size', type=int, default=10, help='Number of time steps per simulation')
    parser.add_argument('--t_end', type=float, default=0.1, help='Simulation end time')
    parser.add_argument('--total_init_rows', type=int, default=2500, help='Number of initial samples')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--niters', type=int, default=10000, help='Number of training iterations')
    parser.add_argument('--test_freq', type=int, default=500, help='Frequency of logging')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--optimizer_type', type=str, choices=['AdamW', 'RMSprop'], default='AdamW', help='Optimizer: AdamW or RMSprop')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index (use -1 for CPU)')
    parser.add_argument('--output_dir_base', type=str, default='./output', help='Base directory for outputs')
    args = parser.parse_args()

    # 配置
    config = Config()
    config.update_from_args(args)

    import time
    start_time = time.time()

    # 创建仿真器
    simulator = BallSimulatorGPU(t_end=config.t_end, n_step=config.data_size, batch_size=config.total_init_rows, device=config.device)

    # 生成仿真数据
    true_y, true_y0 = generate_simulation_data(simulator, config.total_init_rows, config.device)

    end_time = time.time()
    print(f"生成 {config.total_init_rows} 组数据耗时: {end_time - start_time:.2f}秒")

    # 标准化
    min_vals = min_values.to(config.device)
    max_vals = max_values.to(config.device)
    true_y = (true_y - min_vals[1:]) / (max_vals[1:] - min_vals[1:])  # 只标准化 [x, y, z, vx, vy, vz]
    true_y0 = (true_y0 - min_vals) / (max_vals - min_vals)            # 初始条件包括 m

    # 训练
    trainer = ODETrainer(config, true_y.to(config.device), true_y0.to(config.device), min_vals, max_vals, args.model_type, args.total_init_rows, args.batch_size, args.niters)
    trainer.train()