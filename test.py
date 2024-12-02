import torch
import numpy as np
import matplotlib.pyplot as plt
from av_system import DDPG_Car, CarEnv, PIDController
import time

def test_car_ddpg(mode, initial_distance=[-10, 30, 90], initial_velocity=[22, 24, 24], render=False):
    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Metal Performance Shaders) on Mac GPU.")
    else:
        device = torch.device('cpu')
        print("MPS not available. Using CPU.")

    # Load trained DDPG agent
    ddpg_agent = DDPG_Car(state_dim=5, action_dim=2, device=device)
    actor_state_dict = torch.load(r'/Users/creak77/RL_env/XAI_RL/car_ddpg_actor.pth', map_location=device, weights_only=True)
    critic_state_dict = torch.load(r'/Users/creak77/RL_env/XAI_RL/car_ddpg_critic.pth', map_location=device, weights_only=True)
    ddpg_agent.actor.load_state_dict(actor_state_dict)
    ddpg_agent.critic.load_state_dict(critic_state_dict)
    ddpg_agent.actor.eval()
    ddpg_agent.critic.eval()

    # Initialize environment
    safe_distance = initial_velocity[1] * 2
    dt = 0.1
    env = CarEnv(initial_distance, initial_velocity, dt, render=render)
    pid = PIDController(10, 0.01, 10.5)
    state = env.reset()
    done = False
    episode_reward = 0
    acc = []
    # Run simulation
    while not done:
        distance_from_front = state[4] - state[2]
        error = distance_from_front - safe_distance
        obs = env.observe()
        a1, a3 = ddpg_agent.select_action(obs, mode)
        a2 = pid.control(error, dt)
        actions = [a1, a2, a3]
        acc.append(actions)
        # print(f'Actions: {actions}')
        next_state, front_reward, rear_reward, done = env.step(actions)
        episode_reward += front_reward + rear_reward
        state = env.state
        env.history.append(env.state)
        if render:
            env.render()
            time.sleep(dt)
    
    # Plot history
    history = np.array(env.history)
    acc = np.array(acc).T
    plot_history(history, acc, dt, mode)

def plot_history(history, acc, dt, mode):
    t = np.arange(0, len(history) * dt, dt)
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    axs[0].plot(t, history[:, 2] - history[:, 0], label='rear_middle_distance', color='g')
    axs[0].plot(t, history[:, 4] - history[:, 2], label='middle_front_distance', color='b')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Position (m)')
    axs[0].legend()
    axs[1].plot(t, history[:, 1], label='Rear Car Velocity', color='r')
    axs[1].plot(t, history[:, 3], label='Middle Car Velocity', color='g')
    axs[1].plot(t, history[:, 5], label='Front Car Velocity', color='b')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Velocity (m/s)')
    axs[1].legend()
    axs[2].plot(t, acc[0], label='Rear Car Acceleration', color='r')
    axs[2].plot(t, acc[1], label='Middle Car Acceleration', color='g')
    axs[2].plot(t, acc[2], label='Front Car Acceleration', color='b')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Acceleration (m/s^2)')
    axs[2].legend()
    if mode == 'normal':
        plt.savefig('test_-4_rear_acc.png')
    else:
        plt.savefig('test_-6_rear_acc.png')
    plt.show()

if __name__ == '__main__':
    rear_initial_distance = -10
    middle_initial_distance = 30
    front_initial_distance = 90
    initial_distance = [rear_initial_distance, middle_initial_distance, front_initial_distance]
    initial_velocity = [22, 24, 24]
    test_car_ddpg(mode='normal', initial_distance=initial_distance, initial_velocity=initial_velocity, render=False)