import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# === Angle Modulo Utility ===
def angle_mod(x, zero_2_2pi=False, degree=False):
    if degree:
        x = math.radians(x)
    if zero_2_2pi:
        res = x % (2 * math.pi)
    else:
        res = (x + math.pi) % (2 * math.pi) - math.pi
    if degree:
        res = math.degrees(res)
    return res

# === Tuning Parameters ===
k = 0.1
Lfc = 2.0
Kp = 1.0
dt = 0.1
WB = 2.9

LENGTH = WB + 1.0
WIDTH = 2.0
WHEEL_LEN = 0.6
WHEEL_WIDTH = 0.2
MAX_STEER = math.pi / 4

# === State of Vehicle ===
class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.recalc_rear()

    def recalc_rear(self):
        self.rear_x = self.x - (WB / 2) * math.cos(self.yaw)
        self.rear_y = self.y - (WB / 2) * math.sin(self.yaw)

    def update(self, a, delta):
        delta = np.clip(delta, -MAX_STEER, MAX_STEER)
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / WB * math.tan(delta) * dt
        self.yaw = angle_mod(self.yaw)
        self.v += a * dt
        self.recalc_rear()

    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return math.hypot(dx, dy)

# === State Recorder for Visualization ===
class States:
    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.t = []

    def append(self, t, state):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.t.append(t)

# === PID Speed Controller ===
def proportional_control(target, current):
    return Kp * (target - current)

# === Pure Pursuit Target Finder ===
class TargetCourse:
    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_index(self, state):
        if self.old_nearest_point_index is None:
            dx = [state.rear_x - icx for icx in self.cx]
            dy = [state.rear_y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = state.calc_distance(self.cx[ind], self.cy[ind])
            while True:
                if (ind + 1) >= len(self.cx):
                    break
                distance_next_index = state.calc_distance(self.cx[ind + 1], self.cy[ind + 1])
                if distance_this_index < distance_next_index:
                    break
                ind += 1
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        Lf = Lfc + k * state.v
        look_ahead_sum = 0.0
        target_ind = ind
        while look_ahead_sum < Lf and (target_ind + 1) < len(self.cx):
            dx = self.cx[target_ind + 1] - self.cx[target_ind]
            dy = self.cy[target_ind + 1] - self.cy[target_ind]
            look_ahead_sum += math.hypot(dx, dy)
            target_ind += 1
        return target_ind, Lf

# === Pure Pursuit Steering Control ===
def pure_pursuit_steer_control(state, trajectory, prevind):
    target_ind, Lf = trajectory.search_target_index(state)
    if target_ind < len(trajectory.cx):
        tx = trajectory.cx[target_ind]
        ty = trajectory.cy[target_ind]
    else:
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        target_ind = len(trajectory.cx) - 1

    dx = tx - state.rear_x
    dy = ty - state.rear_y
    alpha = math.atan2(dy, dx) - state.yaw
    alpha = angle_mod(alpha)
    if state.v < 0.01:
        return 0.0, target_ind
    delta = math.atan2(2.0 * WB * math.sin(alpha) / Lf, 1.0)
    return delta, target_ind

# === Visualization Utilities ===
def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
              head_width=width, head_length=width, fc=fc, ec=ec)

def plot_vehicle(x, y, yaw, steer=0.0, color='blue'):
    def plot_wheel(wheel_x, wheel_y, wheel_yaw, steer=0.0, color=color):
        wheel = np.array([
            [-WHEEL_LEN/2, WHEEL_WIDTH/2],
            [WHEEL_LEN/2, WHEEL_WIDTH/2],
            [WHEEL_LEN/2, -WHEEL_WIDTH/2],
            [-WHEEL_LEN/2, -WHEEL_WIDTH/2],
            [-WHEEL_LEN/2, WHEEL_WIDTH/2]
        ])
        if steer != 0:
            c, s = np.cos(steer), np.sin(steer)
            rot_steer = np.array([[c, -s], [s, c]])
            wheel = wheel @ rot_steer.T
        c, s = np.cos(wheel_yaw), np.sin(wheel_yaw)
        rot_yaw = np.array([[c, -s], [s, c]])
        wheel = wheel @ rot_yaw.T
        wheel[:, 0] += wheel_x
        wheel[:, 1] += wheel_y
        plt.plot(wheel[:, 0], wheel[:, 1], color=color)

    corners = np.array([
        [-LENGTH/2, WIDTH/2],
        [LENGTH/2, WIDTH/2],
        [LENGTH/2, -WIDTH/2],
        [-LENGTH/2, -WIDTH/2],
        [-LENGTH/2, WIDTH/2]
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    body = corners @ rot.T
    body[:, 0] += x
    body[:, 1] += y
    plt.plot(body[:, 0], body[:, 1], color=color)
    front_x_offset = LENGTH / 4
    rear_x_offset = -LENGTH / 4
    half_width = WIDTH / 2
    plot_wheel(x + front_x_offset * c - half_width * s, y + front_x_offset * s + half_width * c, yaw, steer, 'black')
    plot_wheel(x + front_x_offset * c + half_width * s, y + front_x_offset * s - half_width * c, yaw, steer, 'black')
    plot_wheel(x + rear_x_offset * c - half_width * s, y + rear_x_offset * s + half_width * c, yaw, 0.0, 'black')
    plot_wheel(x + rear_x_offset * c + half_width * s, y + rear_x_offset * s - half_width * c, yaw, 0.0, 'black')
    arrow_length = LENGTH / 2
    plt.arrow(x, y, arrow_length * np.cos(yaw), arrow_length * np.sin(yaw),
              head_width=0.3, head_length=0.4, fc='r', ec='r')

# === Simulation and Animation ===
def main():
    # If needed, set ffmpeg path explicitly (uncomment and edit below)
    # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'  # Linux/Mac
    # plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'  # Windows

    cx = np.arange(0, 50, 0.5)
    cy = [math.sin(ix / 5.0) * ix / 2.0 for ix in cx]
    target_speed = 10.0 / 3.6
    T = 100.0
    state = State(x=0.0, y=-3.0, yaw=0.0, v=0.0)
    lastIndex = len(cx) - 1
    time = 0.0
    states = States()
    states.append(time, state)
    target_course = TargetCourse(cx, cy)
    target_ind, _ = target_course.search_target_index(state)

    # Precompute all states for animation
    traj_x, traj_y, traj_yaw, traj_v, traj_di, traj_target_ind = [], [], [], [], [], []
    while T >= time and lastIndex > target_ind:
        ai = proportional_control(target_speed, state.v)
        di, target_ind = pure_pursuit_steer_control(state, target_course, target_ind)
        state.update(ai, di)
        time += dt
        states.append(time, state)
        traj_x.append(state.x)
        traj_y.append(state.y)
        traj_yaw.append(state.yaw)
        traj_v.append(state.v)
        traj_di.append(di)
        traj_target_ind.append(target_ind)

    # Prepare for animation
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.grid(True)
    plt.axis("equal")
    ax.set_xlim(min(cx)-5, max(cx)+5)
    ax.set_ylim(min(cy)-10, max(cy)+10)
    line_course, = ax.plot(cx, cy, "-r", label="Course")
    line_traj, = ax.plot([], [], "-b", label="Trajectory")
    point_target, = ax.plot([], [], "xg", label="Target")
    vehicle_patch, = ax.plot([], [], color='blue')
    ax.legend()

    def animate(i):
        ax.cla()
        ax.plot(cx, cy, "-r", label="Course")
        ax.plot(traj_x[:i+1], traj_y[:i+1], "-b", label="Trajectory")
        if traj_target_ind[i] < len(cx):
            ax.plot(cx[traj_target_ind[i]], cy[traj_target_ind[i]], "xg", label="Target")
        plot_vehicle(traj_x[i], traj_y[i], traj_yaw[i], traj_di[i])
        ax.set_xlim(min(cx)-5, max(cx)+5)
        ax.set_ylim(min(cy)-10, max(cy)+10)
        ax.grid(True)
        ax.set_title("Speed [km/h]: {:.2f}".format(traj_v[i] * 3.6))
        ax.legend()
        return ax.patches + ax.lines

    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=len(traj_x), interval=50, blit=False)

   # Save animation as GIF using Pillow
    ani.save('pure_pursuit_simulation.gif', writer='pillow', fps=20)
    print("Video saved as 'pure_pursuit_simulation.gif'.")


    # Show final trajectory and speed profile
    plt.figure()
    plt.plot(cx, cy, ".r", label="Course")
    plt.plot(states.x, states.y, "-b", label="Trajectory")
    plt.legend()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.grid(True)

    plt.figure()
    plt.plot(states.t, [v * 3.6 for v in states.v], "-r")
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [km/h]")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    print("=== Pure Pursuit + PID Speed Control Simulation ===")
    main()
