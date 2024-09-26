# 说明：
非牛顿流体模型：

引入了剪切速率的计算和粘度的动态更新。compute_shear_rate 函数计算每个粒子的剪切速率，update_viscosity 函数根据剪切速率更新粘度。
使用幂律流体模型，其中 mu = k_flow * (shear_rate)^(n-1)。
液滴碰撞和溅射模拟：

通过计算韦伯数（We）来判断是否发生溅射。当韦伯数超过临界值 We_critical 时，在撞击点附近生成新的溅射粒子。
溅射粒子的生成方向和速度是随机的，以模拟真实的飞溅效果。
数据可视化：

每隔一定步数（如每10步）保存当前粒子的状态为图片，颜色表示粘度的变化。
可选地，使用 create_animation 函数将保存的图片生成动画（需要安装 imagemagick）。
参数说明：

h：平滑长度，决定了粒子之间相互作用的范围。
rho0：参考密度。
k：压力系数，用于计算粒子的压力。
k_flow 和 n：用于幂律流体模型，控制粘度的变化。
mu：基础粘度系数。
g：重力加速度。
We_critical：临界韦伯数，超过此值则发生溅射。
sigma：表面张力系数。
diameter：液滴直径。
dt：时间步长。
steps：仿真总步数。
1.2 液滴碰撞和溅射模拟
在上面的代码中，simulate_splash 函数已经实现了液滴碰撞和溅射的模拟。当液滴撞击画布（假设画布在 y=0）时，计算韦伯数并判断是否发生溅射。如果发生溅射，则在撞击点附近生成新的粒子，模拟颜料的飞溅。

1.3 数据可视化
仿真过程中，粒子的位置信息和粘度会被实时保存为图片。您可以使用这些图片生成动画，以更直观地观察滴画过程。

### 完整代码

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os

# 定义粒子类
class Particle:
    def __init__(self, position, velocity, mass=1.0):
        self.position = position  # 位置向量 [x, y]
        self.velocity = velocity  # 速度向量 [vx, vy]
        self.mass = mass          # 质量
        self.density = 0.0        # 密度
        self.pressure = 0.0       # 压力
        self.shear_rate = 0.0     # 剪切速率
        self.viscosity = 0.1      # 粘度

# 核函数及其梯度（Cubic Spline）
def W(r, h):
    q = r / h
    alpha = 10 / (7 * np.pi * h ** 2)
    if q <= 1:
        return alpha * (1 - 1.5 * q ** 2 + 0.75 * q ** 3)
    elif q <= 2:
        return alpha * 0.25 * (2 - q) ** 3
    else:
        return 0

def grad_W(rij, h):
    r = np.linalg.norm(rij)
    if r == 0:
        return np.array([0.0, 0.0])
    q = r / h
    alpha = 10 / (7 * np.pi * h ** 2)
    if q <= 1:
        grad = alpha * (-3 * q + 2.25 * q ** 2) / h
    elif q <= 2:
        grad = -alpha * 0.75 * (2 - q) ** 2 / h
    else:
        grad = 0
    return grad * (rij / r) if r != 0 else np.array([0.0, 0.0])

# 计算密度和压力
def compute_density_and_pressure(particles, h, rho0, k):
    for pi in particles:
        density = 0.0
        for pj in particles:
            r = np.linalg.norm(pi.position - pj.position)
            density += pj.mass * W(r, h)
        pi.density = density
        pi.pressure = k * (pi.density - rho0)

# 计算剪切速率
def compute_shear_rate(particles, h):
    for pi in particles:
        shear_rate = 0.0
        for pj in particles:
            if pi != pj:
                rij = pi.position - pj.position
                r = np.linalg.norm(rij)
                if r < h:
                    vij = pi.velocity - pj.velocity
                    grad = grad_W(rij, h)
                    shear_rate += np.dot(vij, grad) / (r + 1e-5)
        pi.shear_rate = abs(shear_rate)

# 更新粘度
def update_viscosity(particles, k_flow, n):
    for p in particles:
        p.viscosity = k_flow * (p.shear_rate ** (n - 1)) if p.shear_rate != 0 else p.viscosity

# 计算加速度
def compute_forces(particles, h, mu, g, We_critical, sigma, diameter):
    new_particles = []
    for pi in particles:
        force_pressure = np.array([0.0, 0.0])
        force_viscosity = np.array([0.0, 0.0])
        for pj in particles:
            if pi != pj:
                rij = pi.position - pj.position
                r = np.linalg.norm(rij)
                if r < h:
                    grad = grad_W(rij, h)
                    force_pressure += -pj.mass * (pi.pressure + pj.pressure) / (2 * pj.density) * grad
                    vij = pi.velocity - pj.velocity
                    force_viscosity += mu * pj.mass * vij / pj.density * W(r, h)
        # 重力
        force_gravity = np.array(g) * pi.mass
        # 总加速度
        acceleration = (force_pressure + force_viscosity + force_gravity) / pi.mass
        # 更新速度和位置（半隐式欧拉法）
        pi.velocity += acceleration * dt
        pi.position += pi.velocity * dt
    # 液滴碰撞和溅射
    new_particles = simulate_splash(particles, We_critical, sigma, diameter)
    particles.extend(new_particles)

# 模拟液滴碰撞和溅射
def simulate_splash(particles, We_critical, sigma, diameter):
    new_particles = []
    particles_to_remove = []
    for p in particles:
        if p.position[1] <= 0:  # 假设画布在 y=0
            We = (p.density * p.velocity[1] ** 2 * diameter) / sigma
            if We > We_critical:
                # 生成溅射粒子
                num_splashes = np.random.randint(5, 10)  # 增加生成粒子的数量
                for _ in range(num_splashes):
                    angle = np.random.uniform(-np.pi / 2, np.pi / 2)
                    speed = np.random.uniform(0.5, 1.0) * np.linalg.norm(p.velocity)  # 增加速度范围
                    vx = speed * np.cos(angle)
                    vy = speed * np.sin(angle)
                    position = p.position.copy()
                    new_particle = Particle(position, np.array([vx, vy]))
                    new_particles.append(new_particle)
            # 移除已撞击的粒子
            particles_to_remove.append(p)
    for p in particles_to_remove:
        particles.remove(p)
    return new_particles

# 可视化函数
def visualize_simulation(particles, step):
    # 确保目录存在
    os.makedirs("simulation_frames", exist_ok=True)
    positions = np.array([p.position for p in particles])
    viscosities = np.array([p.viscosity for p in particles])
    plt.figure(figsize=(6,6))
    plt.scatter(positions[:, 0], positions[:, 1], c=viscosities, cmap='viridis', s=5)  # 增加点的大小
    plt.colorbar(label='Viscosity')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.title(f"Step {step}")
    plt.savefig(f"simulation_frames/step_{step}.png")
    plt.close()

# 仿真主循环
def simulate(particles, h, rho0, k, k_flow, n, mu, g, We_critical, sigma, diameter, steps, dt):
    for step in range(steps):
        compute_density_and_pressure(particles, h, rho0, k)
        compute_shear_rate(particles, h)
        update_viscosity(particles, k_flow, n)
        compute_forces(particles, h, mu, g, We_critical, sigma, diameter)
        # 可视化或数据记录
        if step % 10 == 0:
            visualize_simulation(particles, step)
            print(f"Step {step} completed.")

# 参数设置
h = 5.0         # 平滑长度
rho0 = 1000.0   # 参考密度
k = 1000.0      # 压力系数
k_flow = 0.1    # 流动一致性系数
n = 0.8         # 流变指数（n < 1 表示假塑性流体）
mu = 0.1        # 基础粘度系数
g = [0.0, -9.81] # 重力加速度
We_critical = 10.0 # 临界韦伯数
sigma = 0.072   # 表面张力系数（例如水的表面张力）
diameter = 1.0  # 液滴直径
dt = 0.001      # 时间步长
steps = 1000    # 仿真步数

# 初始化粒子
particles = []
num_particles = 200  # 增加粒子数量以产生更好的效果
for _ in range(num_particles):
    position = np.array([np.random.uniform(45, 55), np.random.uniform(80, 90)])  # 随机初始化位置
    velocity = np.array([0.0, 0.0])  # 初始速度
    particles.append(Particle(position, velocity))

# 运行仿真
simulate(particles, h, rho0, k, k_flow, n, mu, g, We_critical, sigma, diameter, steps, dt)

# 使用动画展示仿真过程（可选）
def create_animation(frames_dir, output_file, steps, interval=100):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    scat = ax.scatter([], [], s=5)

    def init

():
        scat.set_offsets([])
        return scat,

    def update(frame):
        img = plt.imread(f"{frames_dir}/step_{frame}.png")
        ax.clear()
        ax.imshow(img)
        ax.axis('off')
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=range(0, steps, 10),
                                  init_func=init, blit=True)
    ani.save(output_file, writer='imagemagick', fps=10)

# 创建动画
# create_animation('simulation_frames', 'simulation.gif', steps)
```

### 关键改进点

1. **粒子初始化**：
   - 增加了粒子数量，以提高滴画效果的复杂性和丰富性。
   - 使用随机位置初始化粒子，使得它们分布更加均匀。

2. **溅射粒子的生成**：
   - 增加了溅射粒子的数量范围，从 `3-6` 改为 `5-10`。
   - 通过调整速度范围，使得生成的溅射粒子更具动态感。

3. **可视化设置**：
   - 调整可视化中的粒子大小，以提高显示效果。

### 注意事项

- **可视化效果**：根据不同的参数设置和粒子数量，您可能需要反复实验，以获得与波洛克滴画风格更接近的效果。
- **动画生成**：在需要的情况下，可以解除注释以生成动画。确保已经安装了 `imagemagick` 以便保存动画。
