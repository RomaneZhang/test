import torch

class MissileParams:
    Jx = 3.03
    Jy = 306.3
    Jz = 306.3
    coeff_cx = 0.3
    coeff_cdz = 0.082
    coeff_cdy = 0.082
    coeff_mwz_Jz = -0.32
    coeff_mwy_Jy = -0.32
    coeff_mwx_Jx = 1.278
    coeff_mdx_Jx = 9641
    coeff_mdz_Jz = -89.34
    coeff_mdy_Jy = -89.34
    coeff_alpha = 0.23  # 默认值，动态更新
    coeff_beta = 0.23   # 默认值，动态更新
    coeff_malpha_Jz = 11.58  # 默认值，动态更新
    coeff_mbetay_Jy = 11.58  # 默认值，动态更新

class SimulatorGPU:
    def __init__(self, t_end, n_step, batch_size, device):
        self.t_end = torch.tensor(t_end, device=device)
        self.n_step = n_step
        self.batch_size = batch_size
        self.t_inter = self.t_end / (n_step - 1)
        self.device = device
        self.time_steps = torch.linspace(0, t_end, n_step, device=device).view(1, -1).expand(batch_size, -1)

    def fal(self, x, a, delta):
        return torch.where(
            torch.abs(x) > delta,
            torch.sign(x) * torch.abs(x)**a,
            x / (delta**(a - 1))
        )

    def eq_pde_target(self, T_alg, T_pde):
        batch_size = T_pde.shape[0]
        dT_pde = torch.zeros_like(T_pde, device=self.device)
        vt, theta_t = T_pde[:, 0], T_pde[:, 1]
        at_axis = torch.full((batch_size,), -3 * 9.8, device=self.device)
        at_ynormal = torch.full((batch_size,), -2 * 9.8, device=self.device)
        at_znormal = torch.full((batch_size,), -2 * 9.8, device=self.device)
        vtx, vty, vtz = T_alg[:, 0], T_alg[:, 1], T_alg[:, 2]
        dtheta_t = at_ynormal / vt
        dpsi_t = -at_znormal / (vt * torch.cos(theta_t))
        dvt = at_axis
        dxt, dyt, dzt = vtx, vty, vtz
        dT_pde[:, 0] = dvt
        dT_pde[:, 1] = dtheta_t
        dT_pde[:, 2] = dpsi_t
        dT_pde[:, 3] = dxt
        dT_pde[:, 4] = dyt
        dT_pde[:, 5] = dzt
        return dT_pde

    def eq_alg_target(self, T_pde):
        theta_t, psi_t, vt = T_pde[:, 1], T_pde[:, 2], T_pde[:, 0]
        vtx = vt * torch.cos(theta_t) * torch.cos(psi_t)
        vty = vt * torch.sin(theta_t)
        vtz = -vt * torch.cos(theta_t) * torch.sin(psi_t)
        return torch.stack([vtx, vty, vtz], dim=1)

    def airdensity(self, H_km_in):
        H = H_km_in / 1000
        Rcz = 287.0529
        gn = 9.80665
        r_earth = 6356.766
        H = r_earth * H / (r_earth + H)
        H = torch.clamp(H, min=-2.0)

        conditions = [
            (-2 <= H) & (H < 0),
            (0 <= H) & (H < 11),
            (11 <= H) & (H < 20),
            (20 <= H) & (H < 32),
            (32 <= H) & (H < 47),
            (47 <= H) & (H < 51),
            (51 <= H) & (H < 71),
            (H >= 71)
        ]
        params = torch.tensor([
            [-2, -6.5, 301.15, 13029.3, 2.158393 * 0.01],
            [0, -6.5, 288.15, 10332.3, 2.25577 * 0.01],
            [11, 0, 216.65, 2307.83, 0.1576885],
            [20, 1, 216.65, 558.28, 4.61574],
            [32, 2.8, 228.65, 88.51, 1.224579 * 0.01],
            [47, 0, 270.65, 11.31, 0.126227],
            [51, -2.8, 270.65, 6.83, -1.034546 * 0.01],
            [71, -2, 214.65, 0.4, -9.317494 * 0.001]
        ], device=self.device)

        Ts = torch.zeros_like(H, device=self.device)
        Ps = torch.zeros_like(H, device=self.device)
        for i in range(len(conditions)):
            H_range = params[i, :2]
            Ts_base, Ps_base, exp_coeff = params[i, 2], params[i, 3], params[i, 4]
            is_exp = (H_range[1] == 0).float()
            delta_H = H - H_range[0]
            Ts_temp = Ts_base + (1 - is_exp) * H_range[1] * delta_H
            Ps_exp = Ps_base * gn * torch.exp(-exp_coeff * delta_H)
            Ps_power = Ps_base * gn * (1 + exp_coeff * 0.001 * delta_H)**(-34.16322 if H_range[1] > 0 else 12.20115)
            Ps_temp = Ps_exp * is_exp + Ps_power * (1 - is_exp)
            Ts = torch.where(conditions[i], Ts_temp, Ts)
            Ps = torch.where(conditions[i], Ps_temp, Ps)

        rou = Ps / Ts / Rcz
        v_sound = 20.0468 * torch.sqrt(Ts)
        return rou, v_sound

    def table_aerocoeff_Ma_lookup(self, t, n):
        Table_aerocoeff = torch.tensor([
            [0.2, 0.241, 0.11, 9.15],
            [0.78, 0.213, 0.136, 7.51],
            [0.94, 0.258, 0.135, 7.61],
            [1.07, 0.407, 0.109, 9.63],
            [1.32, 0.445, 0.108, 9.89],
            [1.61, 0.372, 0.115, 9.18],
            [2.43, 0.255, 0.121, 8.91],
            [3.5, 0.19, 0.134, 7.94],
            [5.0, 0.15, 0.154, 7.03],
            [6.1, 0.145, 0.16, 6.93]
        ], device=self.device)
        t = t.unsqueeze(-1)
        mask_below = t <= Table_aerocoeff[:, 0]
        mask_above = t >= Table_aerocoeff[:, 0]
        idx_low = torch.where(mask_below, torch.arange(len(Table_aerocoeff), device=self.device), 0).max(dim=-1)[1]
        idx_high = torch.where(mask_above, torch.arange(len(Table_aerocoeff), device=self.device), len(Table_aerocoeff) - 1).min(dim=-1)[1]
        
        x1 = Table_aerocoeff[idx_low, 0]
        x2 = Table_aerocoeff[idx_high, 0]
        y1 = Table_aerocoeff[idx_low, n]
        y2 = Table_aerocoeff[idx_high, n]
        interp = torch.where(
            idx_low == idx_high,
            y1,
            y1 + (y2 - y1) / (x2 - x1) * (t.squeeze(-1) - x1)
        )
        return interp

    def eq_pde_missile(self, M_alg, M_pde):
        p = MissileParams
        dM_pde = torch.zeros_like(M_pde, device=self.device)
        
        beta_11, beta_12 = 10, 12
        a1, d1 = 0.95, 0.1
        beta_31, beta_32 = 8, 15
        a3, d3 = 0.95, 0.1
        beta_51, beta_52 = 20, 23
        a5, d5 = 0.92, 0.1
        beta_71, beta_72 = 30, 26
        a7, d7 = 0.97, 0.1
        beta_91, beta_92 = 10, 12
        a9, d9 = 0.95, 0.1
        beta_111, beta_112 = 8, 15
        a11, d11 = 0.95, 0.1
        S_ref = 0.146

        e1, e3, e5, e7, e9, e11 = M_alg[:, :6].T
        f_alpha, f_beta, f_gamma = M_alg[:, 6:9].T
        fwx, fwy, fwz = M_alg[:, 9:12].T
        u_alpha, u_beta, u_gamma = M_alg[:, 12:15].T
        delta_x, delta_ym, delta_zm, Ty, Tz, Myt, Mzt = M_alg[:, 15:22].T
        u_OGL, Pu_OGL = M_alg[:, 22:24].T
        C_Lafa, C_K, C_D = M_alg[:, 24:27].T
        rou, vmx, vmy, vmz, m_dot, thrust = M_alg[:, 28:].T
        
        vm, theta_m = M_pde[:, 0], M_pde[:, 1]
        z12, z22, z32, z42, z52, z62 = M_pde[:, 14], M_pde[:, 16], M_pde[:, 18], M_pde[:, 20], M_pde[:, 22], M_pde[:, 24]
        m_DD = M_pde[:, 25]
        s = M_pde[:, 6:13]

        dz21 = z22 - beta_31 * e3 + f_alpha + s[:, 6]
        dz22 = -beta_32 * self.fal(e3, a3, d3)
        dz11 = z12 - beta_11 * e1 + fwz + u_alpha
        dz12 = -beta_12 * self.fal(e1, a1, d1)
        dz41 = z42 - beta_71 * e7 + f_gamma + s[:, 4]
        dz42 = -beta_72 * self.fal(e7, a7, d7)
        dz31 = z32 - beta_51 * e5 + fwx + u_gamma
        dz32 = -beta_52 * self.fal(e5, a5, d5)
        dz61 = z62 - beta_111 * e11 + f_beta + s[:, 5]
        dz62 = -beta_112 * self.fal(e11, a11, d11)
        dz51 = z52 - beta_91 * e9 + fwy + u_beta
        dz52 = -beta_92 * self.fal(e9, a9, d9)

        g = 9.8
        dalpha = (s[:, 6] - s[:, 4] * torch.cos(s[:, 0]) * torch.tan(s[:, 1]) + s[:, 5] * torch.sin(s[:, 0]) * torch.tan(s[:, 1]) -
                  p.coeff_cx * torch.sin(s[:, 0]) / torch.cos(s[:, 1]) + g * torch.sin(s[:, 2]) * torch.sin(s[:, 0]) / (vm * torch.cos(s[:, 1])) -
                  (p.coeff_alpha * s[:, 0] + p.coeff_cdz * delta_zm) * torch.cos(s[:, 0]) / torch.cos(s[:, 1]) -
                  Ty * torch.cos(s[:, 0]) / (m_DD * vm * torch.cos(s[:, 1])) +
                  g * torch.cos(s[:, 3]) * torch.cos(s[:, 2]) * torch.cos(s[:, 0]) / (vm * torch.cos(s[:, 1])) -
                  thrust * torch.sin(s[:, 0]) / (m_DD * vm * torch.cos(s[:, 1])))

        dbeta = (s[:, 4] * torch.sin(s[:, 0]) + s[:, 5] * torch.cos(s[:, 0]) - p.coeff_cx * torch.cos(s[:, 0]) * torch.sin(s[:, 1]) +
                 g * torch.sin(s[:, 2]) * torch.cos(s[:, 0]) * torch.sin(s[:, 1]) / vm +
                 (p.coeff_alpha * s[:, 0] + p.coeff_cdz * delta_zm) * torch.sin(s[:, 0]) * torch.sin(s[:, 1]) +
                 Ty * torch.sin(s[:, 0]) * torch.sin(s[:, 1]) / (m_DD * vm) +
                 (p.coeff_beta * s[:, 1] + p.coeff_cdy * delta_ym) * torch.cos(s[:, 1]) +
                 Tz * torch.cos(s[:, 1]) / (m_DD * vm) + g * torch.cos(s[:, 2]) * torch.sin(s[:, 3]) * torch.cos(s[:, 1]) / vm -
                 thrust / (m_DD * vm))

        dtheta_var = s[:, 5] * torch.sin(s[:, 3]) + s[:, 6] * torch.cos(s[:, 3])
        dgamma = s[:, 4] - torch.tan(s[:, 2]) * (s[:, 5] * torch.cos(s[:, 3]) - s[:, 6] * torch.sin(s[:, 3]))

        dwx = p.coeff_mdx_Jx * delta_x + p.coeff_mwx_Jx * s[:, 4]
        dwy = (p.Jz - p.Jy) / p.Jy * s[:, 6] * s[:, 4] + p.coeff_mbetay_Jy * s[:, 1] + p.coeff_mdy_Jy * delta_ym + p.coeff_mwy_Jy * s[:, 5] + Myt / p.Jy
        dwz = (p.Jx - p.Jy) / p.Jz * s[:, 4] * s[:, 5] + p.coeff_malpha_Jz * s[:, 0] + p.coeff_mdz_Jz * delta_zm + p.coeff_mwz_Jz * s[:, 6] + Mzt / p.Jz

        dtheta_m = u_OGL / vm
        dpsi_m = -Pu_OGL / (vm * torch.cos(theta_m))
        dvm = (thrust * torch.cos(s[:, 0]) * torch.cos(s[:, 1]) - (C_D + C_K * C_Lafa**2 * s[:, 0]**2) * 0.5 * rou * vm**2 * S_ref) / m_DD - 9.8 * torch.sin(theta_m)
        dxm, dym, dzm = vmx, vmy, vmz
        dm = torch.where(thrust != 0, -m_dot, torch.zeros_like(m_dot, device=self.device))

        dM_pde[:, 0] = dvm
        dM_pde[:, 1] = dtheta_m
        dM_pde[:, 2] = dpsi_m
        dM_pde[:, 3:6] = torch.stack([dxm, dym, dzm], dim=1)
        dM_pde[:, 6:13] = torch.stack([dalpha, dbeta, dtheta_var, dgamma, dwx, dwy, dwz], dim=1)
        dM_pde[:, 13:25] = torch.stack([dz11, dz12, dz21, dz22, dz31, dz32, dz41, dz42, dz51, dz52, dz61, dz62], dim=1)
        dM_pde[:, 25] = dm
        return dM_pde

    def eq_alg_missile(self, M_alg, M_pde, T_alg, T_pde, t):
        p = MissileParams
        M_alg_new = torch.zeros_like(M_alg, device=self.device)
        d = 1.26
        delta_y_max = torch.pi / 6
        delta_z_max = torch.pi / 6
        d_delta_y = torch.pi / 30
        d_delta_z = torch.pi / 30
        beta_21, a2, d2 = 10, 0.95, 0.1
        beta_41, a4, d4 = 6, 0.95, 0.1
        beta_61, a6, d6 = 7, 0.92, 0.1
        beta_81, a8, d8 = 9, 0.97, 0.1
        beta_101, a10, d10 = 7, 0.92, 0.1
        beta_121, a12, d12 = 6, 0.95, 0.1
        K = 6
        alpha_max = 35 * torch.pi / 180
        beta_max = alpha_max
        S_ref = 0.146
        mf = 490
        m_inter = 10

        thrust = M_alg[:, 33]
        vtx, vty, vtz = T_alg[:, 0], T_alg[:, 1], T_alg[:, 2]
        xt, yt, zt = T_pde[:, 3], T_pde[:, 4], T_pde[:, 5]
        vm, theta_m, psi_m = M_pde[:, 0], M_pde[:, 1], M_pde[:, 2]
        xm, ym, zm = M_pde[:, 3], M_pde[:, 4], M_pde[:, 5]
        inner_state = M_pde[:, 6:13]
        z11, z21, z31, z41, z51, z61 = M_pde[:, 13], M_pde[:, 15], M_pde[:, 17], M_pde[:, 19], M_pde[:, 21], M_pde[:, 23]
        z12, z22, z32, z42, z52, z62 = M_pde[:, 14], M_pde[:, 16], M_pde[:, 18], M_pde[:, 20], M_pde[:, 22], M_pde[:, 24]
        m_DD = M_pde[:, 25]
        command = M_alg[:, 15:22]
        m_dot = (734 - mf) / m_inter

        rou, v_sound = self.airdensity(ym)
        C_Lafa = self.table_aerocoeff_Ma_lookup(vm / v_sound, 3)
        C_K = self.table_aerocoeff_Ma_lookup(vm / v_sound, 2)
        C_D = self.table_aerocoeff_Ma_lookup(vm / v_sound, 1)
        C_Beta = -C_Lafa

        vmx = vm * torch.cos(theta_m) * torch.cos(psi_m)
        vmy = vm * torch.sin(theta_m)
        vmz = -vm * torch.cos(theta_m) * torch.sin(psi_m)

        pos_m = torch.stack([xm, ym, zm], dim=1)
        pos_t = torch.stack([xt, yt, zt], dim=1)
        vel_m = torch.stack([vmx, vmy, vmz], dim=1)
        vel_t = T_alg
        r_vec = pos_t - pos_m
        vr_vec = vel_t - vel_m
        r_norm = torch.norm(r_vec, dim=1)
        vr_proj = torch.where(
            (r_norm > 0)[:, None],
            torch.einsum('bi,bi->b', r_vec, vr_vec)[:, None] * r_vec / (r_norm**2)[:, None],
            torch.zeros_like(r_vec)
        )
        vn = vr_vec - vr_proj
        tbl_tgo = torch.where(
            torch.norm(vr_proj, dim=1) > 0,
            r_norm / torch.norm(vr_proj, dim=1),
            torch.full_like(r_norm, float('inf'))
        )
        tbl_vec = vn * tbl_tgo[:, None]

        u_OGL = torch.clamp(
            torch.where(tbl_tgo != float('inf'), K * 2 * tbl_vec[:, 1] / (tbl_tgo**2), 0),
            -40 * 9.8, 40 * 9.8
        )
        denom = thrust + C_Lafa * 0.5 * rou * vm**2 * S_ref
        alpha_c = torch.clamp(
            torch.where(denom != 0, (u_OGL + 9.8 * torch.cos(theta_m)) * m_DD / denom, 0),
            -alpha_max, alpha_max
        )

        Pu_OGL = torch.clamp(
            torch.where(tbl_tgo != float('inf'), K * 2 * tbl_vec[:, 2] / (tbl_tgo**2), 0),
            -40 * 9.8, 40 * 9.8
        )
        denom = -thrust * torch.cos(alpha_c) + C_Beta * 0.5 * rou * vm**2 * S_ref
        beta_c = torch.clamp(
            torch.where(denom != 0, Pu_OGL * m_DD / denom, 0),
            -beta_max, beta_max
        )

        p.coeff_alpha = torch.where(inner_state[:, 0] > 0, 0.23, -0.23)
        p.coeff_malpha_Jz = torch.where(inner_state[:, 0] > 0, 11.58, 10.43)
        p.coeff_beta = torch.where(inner_state[:, 1] > 0, 0.23, -0.23)
        p.coeff_mbetay_Jy = torch.where(inner_state[:, 1] > 0, 11.58, 10.43)

        e3 = z21 - inner_state[:, 0]
        e4 = alpha_c - inner_state[:, 0]
        u_alph = beta_41 * self.fal(e4, a4, d4)
        f_alpha = -p.coeff_cx * torch.sin(inner_state[:, 0]) - (p.coeff_alpha * inner_state[:, 0] + p.coeff_cdz * command[:, 2]) * torch.cos(inner_state[:, 0])
        w_zc = u_alph - f_alpha - z22

        e1 = z11 - inner_state[:, 6]
        e2 = w_zc - z11
        u_wc = beta_21 * self.fal(e2, a2, d2)
        fwz = p.coeff_malpha_Jz * inner_state[:, 0] + p.coeff_mwz_Jz * inner_state[:, 6]
        u_alpha = u_wc - fwz - z12

        e7 = z41 - inner_state[:, 4]
        gamma_c = torch.zeros_like(e7)
        e8 = gamma_c - inner_state[:, 3]
        u_gam = beta_81 * self.fal(e8, a8, d8)
        f_gamma = -torch.tan(inner_state[:, 2]) * (inner_state[:, 5] * torch.cos(inner_state[:, 3]) - inner_state[:, 6] * torch.sin(inner_state[:, 3]))
        w_xc = u_gam - f_gamma - z42

        e5 = z31 - inner_state[:, 4]
        e6 = w_xc - z31
        u_wxc = beta_61 * self.fal(e6, a6, d6)
        fwx = p.coeff_mwx_Jx * inner_state[:, 4]
        u_gamma = u_wxc - fwx - z32

        e11 = z61 - inner_state[:, 1]
        e12 = beta_c - inner_state[:, 1]
        u_bet = beta_121 * self.fal(e12, a12, d12)
        f_beta = -p.coeff_cx * torch.sin(inner_state[:, 1]) + (p.coeff_beta * inner_state[:, 1] + p.coeff_cdy * command[:, 1]) * torch.cos(inner_state[:, 1])
        w_yc = u_bet - f_beta - z62

        e9 = z51 - inner_state[:, 5]
        e10 = w_yc - z51
        u_wyc = beta_101 * self.fal(e10, a10, d10)
        fwy = p.coeff_mbetay_Jy * inner_state[:, 1] + p.coeff_mwy_Jy * inner_state[:, 5]
        u_beta = u_wyc - fwy - z52

        delta_x = u_gamma / p.coeff_mdx_Jx
        delta_zn = u_alpha / p.coeff_mdz_Jz
        delta_zm = torch.clamp(delta_zn, -delta_z_max + d_delta_z, delta_z_max - d_delta_z)
        Ty = (u_alpha - p.coeff_mdz_Jz * delta_zm) / (d / p.Jz)
        delta_yn = u_beta / p.coeff_mdy_Jy
        delta_ym = torch.clamp(delta_yn, -delta_y_max + d_delta_y, delta_y_max - d_delta_y)
        Tz = (u_beta - p.coeff_mdy_Jy * delta_ym) / (d / p.Jy)
        Mzt = Ty * d
        Myt = Tz * d

        thrust_new = torch.where((t - 1) * self.t_inter >= m_inter, torch.zeros_like(thrust), thrust)

        M_alg_new[:, :6] = torch.stack([e1, e3, e5, e7, e9, e11], dim=1)
        M_alg_new[:, 6:12] = torch.stack([f_alpha, f_beta, f_gamma, fwx, fwy, fwz], dim=1)
        M_alg_new[:, 12:15] = torch.stack([u_alpha, u_beta, u_gamma], dim=1)
        M_alg_new[:, 15:22] = torch.stack([delta_x, delta_ym, delta_zm, Ty, Tz, Myt, Mzt], dim=1)
        M_alg_new[:, 22:24] = torch.stack([u_OGL, Pu_OGL], dim=1)
        M_alg_new[:, 24:27] = torch.stack([C_Lafa, C_K, C_D], dim=1)
        M_alg_new[:, 27] = C_Beta
        M_alg_new[:, 28:32] = torch.stack([rou, vmx, vmy, vmz], dim=1)
        M_alg_new[:, 32] = m_dot
        M_alg_new[:, 33] = thrust_new
        return M_alg_new

    def simulate(self, samples):
        batch_size = samples.shape[0]
        T_pde = torch.zeros(batch_size, 6, self.n_step, device=self.device)
        M_pde = torch.zeros(batch_size, 26, self.n_step, device=self.device)
        T_alg = torch.zeros(batch_size, 3, self.n_step, device=self.device)
        M_alg = torch.zeros(batch_size, 34, self.n_step, device=self.device)

        T_pde[:, :, 0] = samples[:, 0:6]
        M_pde[:, :, 0] = samples[:, 6:32]
        vmx01 = samples[:, 6] * torch.cos(samples[:, 7]) * torch.cos(samples[:, 8])
        vmy01 = samples[:, 6] * torch.sin(samples[:, 7])
        vmz01 = -samples[:, 6] * torch.cos(samples[:, 7]) * torch.sin(samples[:, 8])
        M_alg[:, 29:32, 0] = torch.stack([vmx01, vmy01, vmz01], dim=1)
        M_alg[:, 33, 0] = torch.full((batch_size,), 110 * 1000, device=self.device)

        for t in range(self.n_step - 1):
            T_alg_tilde = self.eq_alg_target(T_pde[:, :, t])
            M_alg_tilde = self.eq_alg_missile(M_alg[:, :, t], M_pde[:, :, t], T_alg_tilde, T_pde[:, :, t], self.time_steps[:, t])

            T_pde_tilde = T_pde[:, :, t] + self.t_inter * self.eq_pde_target(T_alg_tilde, T_pde[:, :, t])
            M_pde_tilde = M_pde[:, :, t] + self.t_inter * self.eq_pde_missile(M_alg_tilde, M_pde[:, :, t])

            T_alg_next = self.eq_alg_target(T_pde_tilde)
            M_alg_next = self.eq_alg_missile(M_alg_tilde, M_pde_tilde, T_alg_next, T_pde_tilde, self.time_steps[:, t])

            T_pde[:, :, t + 1] = T_pde[:, :, t] + 0.5 * self.t_inter * (
                self.eq_pde_target(T_alg_tilde, T_pde[:, :, t]) + self.eq_pde_target(T_alg_next, T_pde_tilde)
            )
            M_pde[:, :, t + 1] = M_pde[:, :, t] + 0.5 * self.t_inter * (
                self.eq_pde_missile(M_alg_tilde, M_pde[:, :, t]) + self.eq_pde_missile(M_alg_next, M_pde_tilde)
            )
            T_alg[:, :, t + 1] = T_alg_next
            M_alg[:, :, t + 1] = M_alg_next

        return T_pde, M_pde, T_alg, M_alg
    
    def simulate_fixed_T_pde(self, samples):
        batch_size = samples.shape[0]
        T_pde = torch.zeros(batch_size, 6, self.n_step, device=self.device)
        M_pde = torch.zeros(batch_size, 26, self.n_step, device=self.device)
        T_alg = torch.zeros(batch_size, 3, self.n_step, device=self.device)
        M_alg = torch.zeros(batch_size, 34, self.n_step, device=self.device)

        # 初始化，只设置初始值，T_pde 在整个仿真中保持不变
        T_pde[:, :, :] = samples[:, 0:6].unsqueeze(-1).expand(-1, -1, self.n_step)  # T_pde 保持初始值
        M_pde[:, :, 0] = samples[:, 6:32]
        vmx01 = samples[:, 6] * torch.cos(samples[:, 7]) * torch.cos(samples[:, 8])
        vmy01 = samples[:, 6] * torch.sin(samples[:, 7])
        vmz01 = -samples[:, 6] * torch.cos(samples[:, 7]) * torch.sin(samples[:, 8])
        M_alg[:, 29:32, 0] = torch.stack([vmx01, vmy01, vmz01], dim=1)
        M_alg[:, 33, 0] = torch.full((batch_size,), 110 * 1000, device=self.device)

        # 仿真循环，只更新 M_pde 和相关的 M_alg
        for t in range(self.n_step - 1):
            # 使用固定的 T_pde[:, :, 0] 计算 T_alg
            T_alg_tilde = self.eq_alg_target(T_pde[:, :, 0])  # T_pde 固定为初始值
            M_alg_tilde = self.eq_alg_missile(M_alg[:, :, t], M_pde[:, :, t], T_alg_tilde, T_pde[:, :, 0], self.time_steps[:, t])

            # 只更新 M_pde，不更新 T_pde
            M_pde_tilde = M_pde[:, :, t] + self.t_inter * self.eq_pde_missile(M_alg_tilde, M_pde[:, :, t])

            # 计算下一时刻的 T_alg 和 M_alg
            T_alg_next = self.eq_alg_target(T_pde[:, :, 0])  # T_pde 仍然固定为初始值
            M_alg_next = self.eq_alg_missile(M_alg_tilde, M_pde_tilde, T_alg_next, T_pde[:, :, 0], self.time_steps[:, t])

            # 更新 M_pde，使用中点法
            M_pde[:, :, t + 1] = M_pde[:, :, t] + 0.5 * self.t_inter * (
                self.eq_pde_missile(M_alg_tilde, M_pde[:, :, t]) + self.eq_pde_missile(M_alg_next, M_pde_tilde)
            )
            T_alg[:, :, t + 1] = T_alg_next
            M_alg[:, :, t + 1] = M_alg_next

        return T_pde, M_pde  # 只返回 T_pde 和 M_pde，与原方法一致

import matplotlib.pyplot as plt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 初始化仿真器
t_end = 20.0  # 20 秒
n_step = 2000  # 2000 步
batch_size = 1  # 单次仿真
simulator = SimulatorGPU(t_end, n_step, batch_size, device)

# 设置初始条件
samples = torch.tensor([
        3471.31099154196, -0.763399365336313, -3.06176266787756, 80000, 75000, -11000,
        269.258240356725, 0.523598775598299, 0.872664625997165, 2000, 1000, -1000, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 734
    ], device=device).view(1, -1)

# 运行仿真
T_pde, M_pde, T_alg, M_alg = simulator.simulate(samples)

# 将结果移到 CPU 并转为 numpy 用于绘图
T_pde = T_pde.cpu().numpy()[0]  # [6, 2000]
M_pde = M_pde.cpu().numpy()[0]  # [26, 2000]
time = simulator.time_steps.cpu().numpy()[0]  # [2000]

# T_pde 变量名称
T_pde_labels = ['vt', 'theta_t', 'psi_t', 'xt', 'yt', 'zt']
# M_pde 变量名称
M_pde_labels = ['vm', 'theta_m', 'psi_m', 'xm', 'ym', 'zm', 'alpha', 'beta', 
                'theta_var', 'gamma', 'wx', 'wy', 'wz', 'z11', 'z12', 'z21', 
                'z22', 'z31', 'z32', 'z41', 'z42', 'z51', 'z52', 'z61', 'z62', 'm_DD']

# 第一张图：前 16 个变量 (T_pde 的 6 个 + M_pde 的前 10 个)
plt.figure(figsize=(15, 10))
for i in range(6):
    plt.subplot(4, 4, i+1)
    plt.plot(time, T_pde[i], label=T_pde_labels[i])
    plt.xlabel('Time (s)')
    plt.legend()
for i in range(10):
    plt.subplot(4, 4, i+7)
    plt.plot(time, M_pde[i], label=M_pde_labels[i])
    plt.xlabel('Time (s)')
    plt.legend()
plt.tight_layout()
plt.suptitle('First 16 Variables (T_pde: 6, M_pde: 10)', y=1.02)
plt.savefig('plot1.png')

# 第二张图：后 16 个变量 (M_pde 的后 16 个)
plt.figure(figsize=(15, 10))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.plot(time, M_pde[i+10], label=M_pde_labels[i+10])
    plt.xlabel('Time (s)')
    plt.legend()
plt.tight_layout()
plt.suptitle('Last 16 Variables (M_pde: 16)', y=1.02)
plt.savefig('plot2.png')

print("仿真完成，图表已保存为 'plot1.png' 和 'plot2.png'")