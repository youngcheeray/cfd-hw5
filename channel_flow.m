%% ========================================================================
%  二维不可压缩明渠流 —— Final assignment
%  方程:  u_t + u u_x + v u_y = -(1/rho) p_x + (mu/rho) lap(u) + F
%         v_t + u v_x + v v_y = -(1/rho) p_y + (mu/rho) lap(v)
%  参数:  rho=2, mu=0.2, F=1;   区域 x∈[0,2], y∈[0,1];  t∈[0,0.5]
%  初值:  u=v=p=0
%  边界:  左右 x 方向周期;  上下 u=v=0, dp/dy=0
%
%  方法:  压力修正型投影法 (incremental pressure-correction / Chorin)
%         空间: 同位网格, 对流项一阶迎风, 扩散项/梯度项二阶中心差分
%         时间: 显式 Euler
%         压力 Poisson 方程: Jacobi 迭代, 收敛准则 max|phi^{k+1}-phi^k| < tol
%  需要 MATLAB R2016b 及以上 (脚本内含局部函数)
%% ========================================================================
clear; clc; close all;

%% ---------------- 1. 物理参数 ----------------
rho = 2;  mu = 0.2;  nu = mu/rho;  F = 1;
Lx  = 2;  Ly = 1;    T  = 0.5;

%% ---------------- 2. 数值参数 (可自行调整) ----------------
Nx  = 80;                 % x 方向网格数 (周期, 不重复存储 x=2)
Ny  = 41;                 % y 方向节点数 (含上下壁面)
dx  = Lx/Nx;              % x 方向步长
dy  = Ly/(Ny-1);          % y 方向步长

% 时间步长: 取显式扩散稳定极限的 50%, 并检查对流 CFL
dt_diff = 0.5/(nu*(1/dx^2 + 1/dy^2));
dt      = 0.5*dt_diff;
nt      = ceil(T/dt);
dt      = T/nt;           % 调整使得恰好到达 t = T

tol_p   = 1e-8;           % 压力迭代收敛准则
maxit_p = 20000;          % 压力迭代最大次数

x = (0:Nx-1)'*dx;         % 列向量 (周期网格)
y = (0:Ny-1)*dy;          % 行向量
fprintf('Nx=%d, Ny=%d, dx=%.4f, dy=%.4f, dt=%.3e, nt=%d\n', Nx,Ny,dx,dy,dt,nt);

%% ---------------- 3. 初始条件 ----------------
u = zeros(Nx,Ny);  v = zeros(Nx,Ny);  p = zeros(Nx,Ny);

% 记录用: 在 t=0.1, 0.25, 0.5 保存 x=Lx/2 处的速度剖面
t_save  = [0.1 0.25 0.5];
prof    = nan(numel(t_save), Ny);
err_hist = zeros(nt,1);  div_hist = zeros(nt,1);  it_hist = zeros(nt,1);

%% ---------------- 4. 时间推进 ----------------
for n = 1:nt
    t = n*dt;

    % ---- 4.1 预测步: 含对流, 扩散, 体积力 F, 旧时刻压力梯度 ----
    [dpdx, dpdy] = grad_c(p, dx, dy);
    [Au, Lu] = adv_diff(u, u, v, dx, dy);
    [Av, Lv] = adv_diff(v, u, v, dx, dy);
    us = u + dt*(-Au + nu*Lu + F - dpdx/rho);
    vs = v + dt*(-Av + nu*Lv       - dpdy/rho);
    us(:,[1 Ny]) = 0;  vs(:,[1 Ny]) = 0;           % 壁面无滑移

    % ---- 4.2 压力修正 Poisson 方程: lap(phi) = rho/dt * div(u*) ----
    div_s = divergence_c(us, vs, dx, dy);
    rhs   = rho/dt*div_s;
    rhs   = rhs - mean(rhs(:));                    % 纯 Neumann/周期问题的相容条件

    phi = zeros(Nx,Ny);
    for it = 1:maxit_p
        pe = circshift(phi,[-1 0]);  pw = circshift(phi,[1 0]);
        pn = [phi(:,2:end), phi(:,end)];
        ps = [phi(:,1),     phi(:,1:end-1)];
        pn(:,Ny) = phi(:,Ny-1);                    % dphi/dy=0 (上壁, 镜像虚点)
        ps(:,1 ) = phi(:,2);                       % dphi/dy=0 (下壁)
        phi_new = ((pe+pw)*dy^2 + (pn+ps)*dx^2 - rhs*dx^2*dy^2) ...
                  / (2*(dx^2+dy^2));
        res = max(abs(phi_new(:)-phi(:)));
        phi = phi_new;
        if res < tol_p, break; end
    end
    phi = phi - mean(phi(:));
    it_hist(n) = it;

    % ---- 4.3 速度/压力修正 ----
    [gx, gy] = grad_c(phi, dx, dy);
    u = us - dt/rho*gx;
    v = vs - dt/rho*gy;
    u(:,[1 Ny]) = 0;  v(:,[1 Ny]) = 0;
    p = p + phi;

    % ---- 4.4 监控量 ----
    ue = u_exact(y, t, nu, F);
    err_hist(n) = max(abs(u(Nx/2,:) - ue));
    div_hist(n) = max(max(abs(divergence_c(u, v, dx, dy))));

    k = find(abs(t - t_save) < 0.5*dt, 1);
    if ~isempty(k), prof(k,:) = u(Nx/2,:); end

    if mod(n,100)==0 || n==nt
        fprintf('step %4d/%d  t=%.3f  max|u|=%.4f  max|div|=%.2e  Poisson its=%d\n', ...
                n, nt, t, max(abs(u(:))), div_hist(n), it);
    end
end
fprintf('\n与解析解的最大误差 (x=1 剖面, t=%.2f): %.3e\n', T, err_hist(end));

%% ---------------- 5. 可视化 ----------------
% 补上 x=Lx 的周期列, 便于作图
xp = [x; Lx];  up = [u; u(1,:)];  vp = [v; v(1,:)];  pp = [p; p(1,:)];
[X, Y] = ndgrid(xp, y);

% 图1: 速度矢量场 (对应题目"结果示例")
figure('Color','w','Position',[100 100 700 450]);
sx = 3; sy = 2;                                   % 抽稀箭头
quiver(X(1:sx:end,1:sy:end), Y(1:sx:end,1:sy:end), ...
       up(1:sx:end,1:sy:end), vp(1:sx:end,1:sy:end), 1.2, 'b');
axis([0 Lx 0 Ly]); xlabel('x'); ylabel('y');
title(sprintf('速度矢量场 (t = %.1f)', T));
saveas(gcf, 'fig1_quiver.png');

% 图2: u 云图 + 流线
figure('Color','w','Position',[100 100 700 450]);
contourf(X, Y, up, 20, 'LineColor','none'); colorbar; hold on;
[sxx, syy] = meshgrid(0.05, 0.05:0.1:0.95);
h = streamline(stream2(X', Y', up', vp', sxx, syy)); set(h,'Color','k');
xlabel('x'); ylabel('y'); title('u 分量云图及流线 (t = 0.5)');
saveas(gcf, 'fig2_u_contour.png');

% 图3: 速度剖面 数值解 vs 解析解
figure('Color','w','Position',[100 100 600 450]); hold on; grid on;
cols = lines(numel(t_save));
for k = 1:numel(t_save)
    plot(prof(k,:), y, 'o', 'Color', cols(k,:), 'MarkerSize', 4);
    plot(u_exact(y, t_save(k), nu, F), y, '-', 'Color', cols(k,:), 'LineWidth', 1.3);
end
ys = plot(5*y.*(1-y), y, 'k--');                  % 稳态抛物线解 F/(2nu)*y(1-y)
xlabel('u'); ylabel('y');
legend([arrayfun(@(t) sprintf('t=%.2f 数值 / 解析', t), t_save, 'UniformOutput', false), ...
        {'稳态解'}], 'Location','southeast');
legend('boxoff');
title('x = 1 处 u(y) 剖面: 点=数值解, 实线=解析解');
saveas(gcf, 'fig3_profile.png');

% 图4: 压力场 + 散度/误差/迭代历史
figure('Color','w','Position',[100 100 900 600]);
subplot(2,2,1); contourf(X, Y, pp, 20, 'LineColor','none'); colorbar;
title('压力场 p (t = 0.5)'); xlabel('x'); ylabel('y');
tt = (1:nt)*dt;
subplot(2,2,2); semilogy(tt, div_hist); grid on;
xlabel('t'); title('max|\nabla\cdot u|');
subplot(2,2,3); semilogy(tt, err_hist); grid on;
xlabel('t'); title('与解析解的最大误差');
subplot(2,2,4); plot(tt, it_hist); grid on;
xlabel('t'); title('每个时间步的压力迭代次数');
saveas(gcf, 'fig4_diagnostics.png');

%% ======================== 局部函数 ========================
function [A, L] = adv_diff(q, u, v, dx, dy)
% 返回对流项 A = u q_x + v q_y (一阶迎风) 与 Laplace 项 L (二阶中心)
    qe = circshift(q,[-1 0]);  qw = circshift(q,[1 0]);   % x 方向周期
    qn = [q(:,2:end), q(:,end)];                          % y 方向 (壁面行之后会被覆盖)
    qs = [q(:,1),     q(:,1:end-1)];
    qx = (u>0).*(q-qw)/dx + (u<=0).*(qe-q)/dx;
    qy = (v>0).*(q-qs)/dy + (v<=0).*(qn-q)/dy;
    A  = u.*qx + v.*qy;
    L  = (qe-2*q+qw)/dx^2 + (qn-2*q+qs)/dy^2;
end

function [gx, gy] = grad_c(f, dx, dy)
% 中心差分梯度; 上下壁面 df/dy = 0
    fe = circshift(f,[-1 0]);  fw = circshift(f,[1 0]);
    fn = [f(:,2:end), f(:,end)];
    fs = [f(:,1),     f(:,1:end-1)];
    gx = (fe-fw)/(2*dx);
    gy = (fn-fs)/(2*dy);
    gy(:,[1 end]) = 0;
end

function d = divergence_c(u, v, dx, dy)
% 散度: x 方向中心差分; y 方向内点中心差分, 壁面单侧差分
    ue = circshift(u,[-1 0]);  uw = circshift(u,[1 0]);
    vn = [v(:,2:end), v(:,end)];
    vs = [v(:,1),     v(:,1:end-1)];
    d  = (ue-uw)/(2*dx) + (vn-vs)/(2*dy);
    d(:,1)   = (ue(:,1)-uw(:,1))/(2*dx)     + (v(:,2)-v(:,1))/dy;
    d(:,end) = (ue(:,end)-uw(:,end))/(2*dx) + (v(:,end)-v(:,end-1))/dy;
end

function ue = u_exact(y, t, nu, F)
% 解析解 (u=u(y,t), v=0, p=0):  u_t = nu u_yy + F, u(0)=u(1)=0, u(y,0)=0
% u = F/(2nu) y(1-y) - sum_{n odd} 4F/(nu (n pi)^3) exp(-nu (n pi)^2 t) sin(n pi y)
    ue = F/(2*nu)*y.*(1-y);
    for n = 1:2:199
        ue = ue - 4*F/(nu*(n*pi)^3) * exp(-nu*(n*pi)^2*t) .* sin(n*pi*y);
    end
end
