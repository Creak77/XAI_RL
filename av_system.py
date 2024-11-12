'''
roadmap
1) Model the dynamical system for the 3 cars. Model the cars as point masses where acceleration is input. The output of the dynamical system should be the distances and velocities of the cars.
2) Decide on the control algorithm of the middle vehicle
3) Search literature on highway acceleration data to identify the bounds of the RL action
4) Train a RL agent (DDPG or similar) to crash the the cars.
'''
'''
Scenario 1A.
 
Autonomous vehicle driving between two cars in one lane.
 
Failure: Car crash.
 
Impact score: The risk of driving between two cars, one at the front and the other at the rear, of the autonomous vehicle.
 
Classification: High risk, medium risk, low risk of crashing.
 
RL Agents:

 Observations:

 (1), (2) Distances to front and rear vehicles.

 (3), (4) Velocities of front and rear vehicles.
 
RL Actions:

 (1), (2) Acceleration of front and rear vehicles.
 
Chain of events:

 (1) Front car decelerates hard

 (2) Middle car decelerates according to its own algorithms (can be simple if-else, or PID)

 (3) Rear car decelerates
 
But cars are too close to avoid a crash.
 
Considerations:

 (1) The RL action (acceleration) of the front vehicle should be limited to max deceleration or acceleration possible in an average passenger car. Deceleration cannot be instantaneous. There is a minimum deceleration rate related to tire-skid, brakes, etc.

 (2) The deceleration of the rear vehicle is also limited by lower tail of average deceleration rates on highways. The rear car has to decelerate to avoid the crash as well; else the scenarios becomes one that cannot be avoided. 

 However, the scenario should be: if the front car stops suddenly, and the rear car makes a good effort to stop as well, is the state of the system (consisting of speeds and distances between vehicles) safe enough to avoid a crash?
 
Environment/dynamical model: Just use simple point masses for the vehicles. 
'''

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
        print(self.state)
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
        print(actions)
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

# define the DDPG RL agent
class DDPG_Car:
    def __init__(self, state_dim, action_dim):
        self.actor = Car_Actor(state_dim, action_dim, max_action=3)
        self.critic = Car_Critic(state_dim, action_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        self.memory = ReplayBuffer(capacity=100000)
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).detach().cpu().numpy()[0]
        # acceleration limit of +-3 m/s^2
        action = np.clip(action, -3, 3)
        return action[0]
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Compute target Q-value
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + (1 - dones) * self.gamma * next_Q

        # Get current Q-value estimates
        current_Q = self.critic(states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def save_model(self, actor_path='actor.pth', critic_path='critic.pth'):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

def train(num_episodes=100):
    front_car = DDPG_Car(state_dim=2, action_dim=1)
    rear_car = DDPG_Car(state_dim=2, action_dim=1)
    safe_distance = 30
    # use tqdm for progress bar
    for episode in tqdm.tqdm(range(num_episodes)):
        #initial_distance = [_ for _ in np.random.randint(0, 170, 3)]
        rear_initial_distance = np.random.randint(-10, 10)
        middle_initial_distance = np.random.randint(30, 50)
        front_initial_distance = np.random.randint(70, 90)
        initial_distance = [rear_initial_distance, middle_initial_distance, front_initial_distance]
        initial_velocity = [_ for _ in np.random.randint(50, 100, 3)]
        dt = 0.1
        env = CarEnv(initial_distance, initial_velocity, dt)
        pid = PIDController(1, 0.1, 0.01)
        
        done = False
        front_car_reward = 0
        rear_car_reward = 0
        state = env.reset()
        while not done:
            obs_rear = env.observe(car_id=0)
            obs_front = env.observe(car_id=2)
            distance_from_front = state[4] - state[2]
            error = distance_from_front - safe_distance
            a1 = front_car.select_action(obs_front)
            a2 = pid.control(error, dt)
            a3 = rear_car.select_action(obs_rear)
            actions = [a1, a2, a3]

            obs_front, obs_rear, reward_front, reward_rear, done = env.step(actions)
            front_car_reward += reward_front
            rear_car_reward += reward_rear
            front_car.remember(obs_front, a1, reward_front, obs_front, done)
            rear_car.remember(obs_rear, a3, reward_rear, obs_rear, done)
            front_car.update()
            rear_car.update()
            env.render()
            state = env.state
            env.history.append(env.state)
            time.sleep(dt)
        if (episode + 1) % 50 == 0:
            front_car_actor_path = f'front_car_actor_{episode+1}.pth'
            front_car_critic_path = f'front_car_critic_{episode+1}.pth'
            rear_car_actor_path = f'rear_car_actor_{episode+1}.pth'
            rear_car_critic_path = f'rear_car_critic_{episode+1}.pth'
            front_car.save_model(front_car_actor_path, front_car_critic_path)
            rear_car.save_model(rear_car_actor_path, rear_car_critic_path)
        print(f'Episode {episode+1}, Front car reward: {front_car_reward}, Rear car reward: {rear_car_reward}')
        time.sleep(2)



if __name__ == '__main__':
    train()
    # initial_distance = [0.0, 40.0, 100.0]
    # initial_velocity = [50, 50, 50]
    # dt = 0.1
    # env = CarEnv(initial_distance, initial_velocity, dt)

    # a1, a2, a3 = 3, 0, 0
    # state = env.reset()
    # done = False
    # while not done:
    #     # print(state)
    #     action = [a1, a2, a3]
    #     obs_front, obs_rear, reward_front, reward_rear, done = env.step(action)
    #     env.render()
    #     state = env.state
    #     env.history.append(env.state)
    #     time.sleep(dt)