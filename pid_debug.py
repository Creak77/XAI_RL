import numpy as np
import random
import copy
from scipy.integrate import solve_ivp
from ddpg_modules import Car_Actor, Car_Critic, ReplayBuffer
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
import tqdm


class CarEnv:
    def __init__(self, initial_distance, initial_velocity, dt):
        # d is center of mass of the car, car length is 5m
        self.initial_distance = initial_distance
        self.initial_velocity = initial_velocity
        self.dt = dt
        self.state = None
        self.t = 0
        self.history = []
        # self.reset()

        self.fig, self.ax = plt.subplots(figsize=(10, 2))
        self.car_length = 5.0
        self.car_height = 0.2
        self.colors = ['red', 'blue', 'green']  # Colors for Car1, Car2, Car3
        self.init_plot()

    def init_plot(self):
        self.ax.set_ylim(-1, self.car_height + 2)
        self.ax.set_xlim(-10, 40)  # Initial range for positioning
        self.ax.set_facecolor('#f2f2f2')  # Light background color for the road
        
        # Draw road background
        self.ax.fill_betweenx([-1, self.car_height + 1], -100, 10000, color='#e0e0e0', alpha=0.8)
        for i in range(-100, 10000, 5):
            # Draw dashed lines on the road
            self.ax.plot([i, i + 2], [-0.15, -0.15], color='white', lw=2, alpha=0.8)
            # Draw the second dashed line on top
            self.ax.plot([i, i + 2], [self.car_height + 0.15, self.car_height + 0.15], color='white', lw=2, alpha=0.8)


        self.ax.set_xlabel('Position (m)', fontsize=12)
        self.ax.set_yticks([])
        self.ax.set_title('Three-Car Simulation', fontsize=16, fontweight='bold')

        self.cars = []
        self.labels = []
        for i in range(3):
            car_patch = Rectangle((0, 0), self.car_length, self.car_height, color=self.colors[i], alpha=0.8)
            self.ax.add_patch(car_patch)
            self.cars.append(car_patch)

            label = self.ax.text(0, self.car_height + 0.5, f'Car {i+1}', ha='center', va='bottom', fontsize=10, color=self.colors[i])
            self.labels.append(label)
        
        plt.ion()
        plt.show()

    def reset(self):
        d1, v1 = self.initial_distance[0], self.initial_velocity[0]
        d2, v2 = self.initial_distance[1], self.initial_velocity[1]
        d3, v3 = self.initial_distance[2], self.initial_velocity[2]
        self.state = np.array([d1, v1, d2, v2, d3, v3])
        self.t = 0
        # print(self.state)
        return self.state
    
    def observe(self, car_id):
        if car_id == 0:
            # rear car observes its own speed and the distance to the middle car
            distance_to_middle = self.state[2] - self.state[0]
            velocity = self.state[1]
            observation = np.array([distance_to_middle, velocity])
        elif car_id == 2:
            # front car observes its own speed and the distance to the middle car
            distance_to_middle = self.state[4] - self.state[2]
            velocity = self.state[5]
            observation = np.array([distance_to_middle, velocity])
        else:
            observation = None
        return observation

    # Define the dynamical system
    def car_dynamics(self, x, u):
        # x = [d1, v1, d2, v2, d3, v3]
        # u = [a1, a2, a3]
        A = np.array([[0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0]])
        B = np.array([[0, 0, 0],
                        [1, 0, 0],
                        [0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0],
                        [0, 0, 1]])
        C = np.eye(6)
        D = np.zeros((6, 3))
        
        dx = A @ x + B @ u
        y = C @ x + D @ u
        return dx, y

    # Define a function to simulate the car dynamics
    def simulate_car_dynamics(self, x0, u, dt):
        sol = solve_ivp(lambda t, x: self.car_dynamics(x, u)[0], [0, dt], x0, t_eval=[dt], method='RK45')
        return sol.y[:, -1]
    
    def step(self, actions):
        # actions = [a1, a2, a3]
        # print(actions)
        next_state = self.simulate_car_dynamics(self.state, actions, self.dt)
        self.state = next_state
        self.t += self.dt
        # Get observations for front and rear cars
        obs_rear = self.observe(car_id=0)
        obs_front = self.observe(car_id=2)
        # Rewards for front and rear agents
        reward_front = self.reward_front()
        reward_rear = self.reward_rear()
        done = self.done()
        return obs_front, obs_rear, reward_front, reward_rear, done

    
    def reward_front(self):
        collision_with_middle = (self.state[4] - self.state[2]) < 4.9
        stopped = self.state[1] < 0.5
        reward = 0
        if stopped:
            reward += 1
        if collision_with_middle:
            reward += 2
        else:
            reward -= 10
        return reward
    
    def reward_rear(self):
        detect_front_stopped = self.state[1] < 0.5
        collision_with_middle = (self.state[2] - self.state[0]) < 4.9
        slowing_down = self.state[5] < self.initial_velocity[2]
        reward = 0
        if detect_front_stopped and slowing_down:
            reward += 2
        if detect_front_stopped and not slowing_down:
            reward -= 1
        if collision_with_middle:
            reward += 3
        else:
            reward -= 10
        return reward
        
    def done(self):
        if self.t >= 20 or self.check_collision():
            return True
        return False
        
    def check_collision(self):
        if self.state[2] - self.state[0] < 4.9 or self.state[4] - self.state[2] < 4.9:
            return True
        return False
    
    def render(self):
        for i, car_patch in enumerate(self.cars):
            d = self.state[0 + i * 2]
            car_patch.set_xy((d, 0))
            self.labels[i].set_x(d)
            # self.labels[i].set_text(f'Car {i+1} | Pos: {d:.2f}m | Vel: {self.state[1 + i * 2]:.2f} m/s')
            self.labels[i].set_text(f'Car {i+1}|{d:.2f}m|{self.state[1 + i * 2]:.2f}m/s')
        
        min_pos = min(self.state[0], self.state[2], self.state[4]) - self.car_length - 10
        max_pos = max(self.state[0], self.state[2], self.state[4]) + self.car_length + 10
        self.ax.set_xlim(min_pos, max_pos)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        

# Define the control algorithm for the middle car
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def control(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        u = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        # acceleration limit of +-3 m/s^2
        u = np.clip(u, -3, 3)
        return u
    
if __name__ == "__main__":
    initial_distance = [10, 50, 90]
    initial_velocity = [50, 60, 50]
    dt = 0.1
    env = CarEnv(initial_distance, initial_velocity, dt)
    state = env.reset()
    pid = PIDController(10, 0.01, 10.5)
    for _ in range(200):
        error = state[4] - state[2] - 30
        action = pid.control(error, dt)
        print(f'Error: {error:.2f} | Action: {action:.2f}')
        obs_front, obs_rear, reward_front, reward_rear, done = env.step([0, action, 0])
        env.render()
        state = env.state
        if done:
            break
        time.sleep(0.1)
    print("Simulation done")