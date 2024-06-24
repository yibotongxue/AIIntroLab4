from typing import List
import numpy as np
from utils import Particle

### 可以在这里写下一些你需要的变量和函数 ###
COLLISION_DISTANCE = 1
MAX_ERROR = 50000

def is_in_walls(point, walls):
    return np.any(np.all(np.abs(point - walls) < np.array([0.75, 0.75]), axis=1))

def is_out_of_range(point, t_range):
    x_min, y_min, x_max, y_max = t_range
    return float(point[0]) < x_min or float(point[0]) >= x_max or float(point[1]) < y_min or float(point[1]) >= y_max

t_range = None
t_walls = None

k = 0.42
scale = 0.096

### 可以在这里写下一些你需要的变量和函数 ###


def generate_uniform_particles(walls, N):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    N: int, 采样点数量
    输出：
    particles: List[Particle], 返回在空地上均匀采样出的N个采样点的列表，每个点的权重都是1/N
    """
    all_particles: List[Particle] = []
    for _ in range(N):
        all_particles.append(Particle(1.0, 1.0, 1.0, 0.0))
    ### 你的代码 ###
    x_min, y_min = walls.min(axis=0)
    x_max, y_max = walls.max(axis=0)
    global t_range
    global t_walls
    t_range = (x_min, y_min, x_max, y_max)
    t_walls = walls.copy()
    for i in range(N):
        while True:
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            if not is_in_walls(np.array([x, y]), walls):
                all_particles[i].position = np.array([x, y]).copy()
                all_particles[i].theta = np.random.uniform(-np.pi, np.pi)
                all_particles[i].weight = 1.0 / N
                break
    
    ### 你的代码 ###
    return all_particles


def calculate_particle_weight(estimated, gt):
    """
    输入：
    estimated: np.array, 该采样点的距离传感器数据
    gt: np.array, Pacman实际位置的距离传感器数据
    输出：
    weight, float, 该采样点的权重
    """
    weight : float = 1.0
    ### 你的代码 ###
    global k

    weight = float(np.exp(-k * np.linalg.norm(estimated - gt)))
    
    ### 你的代码 ###
    return weight


def resample_particles(walls, particles: List[Particle]):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    particles: List[Particle], 上一次采样得到的粒子，注意是按权重从大到小排列的
    输出：
    particles: List[Particle], 返回重采样后的N个采样点的列表
    """
    resampled_particles: List[Particle] = []
    for _ in range(len(particles)):
        resampled_particles.append(Particle(1.0, 1.0, 1.0, 0.0))
    ### 你的代码 ###
    global scale
    x_min, y_min = walls.min(axis=0)
    x_max, y_max = walls.max(axis=0)
    global t_range
    global t_walls
    t_range = (float(x_min), float(y_min), float(x_max), float(y_max))
    t_walls = walls.copy()
    weights = [particle.weight for particle in particles]
    prefix_sum = np.cumsum(weights)
    for i in range(len(particles)):
        while True:
            weight = np.random.uniform(0, prefix_sum[-1])
            pos = np.searchsorted(prefix_sum, weight)
            resampled_particles[i].position = particles[pos].position + np.random.normal(0, scale, size=(2,))
            resampled_particles[i].theta = particles[pos].theta + np.random.normal(0, scale)
            if resampled_particles[i].theta < -np.pi:
                resampled_particles[i].theta += 2 * np.pi
            if resampled_particles[i].theta >= np.pi:
                resampled_particles[i].theta -= 2 * np.pi
            resampled_particles[i].weight = particles[pos].weight + 0.0
            break
                
    ### 你的代码 ###
    return resampled_particles

def apply_state_transition(p: Particle, traveled_distance, dtheta):
    """
    输入：
    p: 采样的粒子
    traveled_distance, dtheta: ground truth的Pacman这一步相对于上一步运动方向改变了dtheta，并移动了traveled_distance的距离
    particle: 按照相同方式进行移动后的粒子
    """
    ### 你的代码 ###

    p.theta += dtheta
    if p.theta < -np.pi:
        p.theta += 2 * np.pi
    if p.theta >= np.pi:
        p.theta -= 2 * np.pi
    global t_range
    global t_walls
    t_x = p.position[0] + traveled_distance * np.cos(p.theta)
    t_y = p.position[1] + traveled_distance * np.sin(p.theta)
    while is_in_walls(np.array([t_x, t_y]), t_walls) or is_out_of_range(np.array([t_x, t_y]), t_range):
        return p
    p.position[0] += traveled_distance * np.cos(p.theta)
    p.position[1] += traveled_distance * np.sin(p.theta)

    ### 你的代码 ###
    return p

def get_estimate_result(particles: List[Particle]):
    """
    输入：
    particles: List[Particle], 全部采样粒子
    输出：
    final_result: Particle, 最终的猜测结果
    """
    final_result = Particle()
    ### 你的代码 ###

    final_result = max(particles, key=lambda x : x.get_weight())
    
    ### 你的代码 ###
    return final_result
