
from av_system import CarEnv, PIDController, DDPG_Car
import time
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch

def train_car_ddpg(num_episodes=1000):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Metal Performance Shaders) on Mac GPU.")
    else:
        device = torch.device('cpu')
        print("MPS not available. Using CPU.")

    ddpg_agent = DDPG_Car(state_dim=5, action_dim=2, device=device)
    loss = []
    reward_plot = []
    logging.basicConfig(level=logging.INFO)
    
    # Replay buffer warm-up
    initial_warmup_steps = 10000
    rear_initial_distance = np.random.randint(-10, 0)
    middle_initial_distance = np.random.randint(40, 50)
    front_initial_distance = np.random.randint(90, 100)
    initial_distance = [rear_initial_distance, middle_initial_distance, front_initial_distance]
    initial_velocity = [_ for _ in np.random.randint(22, 28, 3)]
    safe_distance = initial_velocity[1] * 2
    dt = 0.1
    env = CarEnv(initial_distance, initial_velocity, dt)
    state = env.reset()
    print("Populating replay buffer...")
    for _ in range(initial_warmup_steps):
        obs = env.observe()
        a1, a3 = ddpg_agent.select_action(obs)
        error = (state[4] - state[2]) - safe_distance
        a2 = PIDController(10, 0.01, 10.5).control(error, dt)
        actions = [a1, a2, a3]
        next_obs, reward_front, reward_rear, done = env.step(actions)
        reward = (reward_front + reward_rear) / 10.0  # Reward normalization
        ddpg_agent.remember(obs, [a1, a3], reward, next_obs, done)
        if done:
            state = env.reset()
        else:
            state = next_obs

    print("Replay buffer warm-up complete. Starting training...")
    
    # Training loop
    for episode in tqdm.tqdm(range(num_episodes), desc="Training Episodes"):
        c_loss = 0
        a_loss = 0
        total_reward = 0
        rear_initial_distance = np.random.randint(-10, 0)
        middle_initial_distance = np.random.randint(40, 50)
        front_initial_distance = np.random.randint(90, 100)
        initial_distance = [rear_initial_distance, middle_initial_distance, front_initial_distance]
        initial_velocity = [_ for _ in np.random.randint(22, 28, 3)]
        safe_distance = initial_velocity[1] * 2
        env = CarEnv(initial_distance, initial_velocity, dt)
        pid = PIDController(10, 0.01, 10.5)
        done = False
        state = env.reset()
        while not done:
            distance_from_front = state[4] - state[2]
            error = distance_from_front - safe_distance
            obs = env.observe()
            # a1 is acceleration of rear car, a2 is acceleration of middle car, a3 is acceleration of front car
            a1, a3 = ddpg_agent.select_action(obs)
            a2 = pid.control(error, dt)
            actions = [a1, a2, a3]
            if (episode + 1) > 1900:
                logging.info(f'Actions: {actions}')
            next_obs, reward_front, reward_rear, done = env.step(actions)
            reward = (reward_front + reward_rear) / 10.0  # Reward normalization
            total_reward += reward
            ddpg_agent.remember(obs, [a1, a3], reward, next_obs, done)
            critic_loss, actor_loss = ddpg_agent.update()
            if critic_loss is not None and actor_loss is not None:
                c_loss += critic_loss
                a_loss += actor_loss
            state = env.state
            env.history.append(env.state)
            time.sleep(dt)
        if (episode + 1) % 100 == 0 and (episode + 1) >= 1500:
            ddpg_agent.save_model(f'car_ddpg_actor_{episode+1}.pth', f'car_ddpg_critic_{episode+1}.pth')
        logging.info(f'Episode {episode+1}, Critic loss: {c_loss/len(env.history):.4f}, Actor loss: {a_loss/len(env.history):.4f}')
        logging.info(f'Episode {episode+1}, Reward: {total_reward:.4f}')
        loss.append([c_loss / len(env.history), a_loss / len(env.history)])
        reward_plot.append(total_reward)
        time.sleep(1)
    # Plot loss
    loss = np.array(loss)
    plt.figure()
    plt.plot(loss[:, 0], label='Critic loss')
    plt.plot(loss[:, 1], label='Actor loss')
    plt.legend()
    plt.savefig('car_ddpg_loss.png')
    # Plot reward
    plt.figure()
    plt.plot(reward_plot)
    plt.savefig('car_ddpg_reward.png')


if __name__ == '__main__':
    train_car_ddpg(2000)
