import torch
import numpy as np
import matplotlib.pyplot as plt
from av_system import DDPG_Car, CarEnv, PIDController
import time

def test_car_ddpg(num_episodes=10):
    # Set device to GPU if available
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Metal Performance Shaders) on Mac GPU.")
    else:
        device = torch.device('cpu')
        print("MPS not available. Using CPU.")

    # Initialize the DDPG agent
    ddpg_agent = DDPG_Car(state_dim=5, action_dim=2, device=device)

    # Load the trained model weights
    
    actor_state_dict = torch.load(r'/Users/creak77/RL_env/XAI_RL/car_ddpg_actor_1000.pth', map_location=device, weights_only=True)
    critic_state_dict = torch.load(r'/Users/creak77/RL_env/XAI_RL/car_ddpg_critic_1000.pth', map_location=device, weights_only=True)
    ddpg_agent.actor.load_state_dict(actor_state_dict)
    ddpg_agent.critic.load_state_dict(critic_state_dict)

    # Set the agent to evaluation mode
    ddpg_agent.actor.eval()
    ddpg_agent.critic.eval()

    for episode in range(num_episodes):
        rear_initial_distance = np.random.randint(-10, 0)
        middle_initial_distance = np.random.randint(40, 50)
        front_initial_distance = np.random.randint(90, 100)
        initial_distance = [rear_initial_distance, middle_initial_distance, front_initial_distance]
        initial_velocity = np.random.randint(22, 28, 3)
        safe_distance = initial_velocity[1] * 2
        dt = 0.1
        env = CarEnv(initial_distance, initial_velocity, dt)
        pid = PIDController(10, 0.01, 10.5)
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            distance_from_front = state[4] - state[2]
            error = distance_from_front - safe_distance
            obs = env.observe()
            a1, a3 = ddpg_agent.select_action(obs)
            a2 = pid.control(error, dt)
            actions = [a1, a2, a3]
            print(f'Actions: {actions}')
            next_obs, reward_front, reward_rear, done = env.step(actions)
            reward = reward_front + reward_rear
            episode_reward += reward
            state = env.state
            env.history.append(env.state)
            env.render()
            time.sleep(dt)

if __name__ == '__main__':
    test_car_ddpg(num_episodes=10)