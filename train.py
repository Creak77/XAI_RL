from av_system import CarEnv, Car_PIDController, DDPG_Car
from water_system import WaterEnv, Water_PIDController, DDPG_Water
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
    safe_distance_fm = initial_velocity[1] * 2
    safe_distance_mr = initial_velocity[0] * 1.5
    safe_distance = initial_velocity[1] * 2
    dt = 0.1
    env = CarEnv(initial_distance, initial_velocity, dt)
    state = env.reset()
    print("Populating replay buffer...")
    for _ in range(initial_warmup_steps):
        obs = env.observe()
        a1, a3 = ddpg_agent.select_action(obs)
        error = (state[4] - state[2]) - safe_distance_fm
        a2 = Car_PIDController(10, 0.01, 10.5).control(error, dt)
        actions = [a1, a2, a3]
        next_obs, front_reward, rear_reward, done = env.step(actions)
        #next_obs, reward, done = env.step(actions)
        reward = front_reward + rear_reward
        ddpg_agent.remember(obs, [a1, a3], reward, next_obs, done)
        if done:
            state = env.reset()
        else:
            state = env.state

    print("Replay buffer warm-up complete. Starting training...")
    # use tqdm for progress bar
    for episode in tqdm.tqdm(range(num_episodes), desc="Training Episodes"):
        c_loss = 0
        a_loss = 0
        total_reward = 0
        rear_initial_distance = np.random.randint(-10, 0)
        middle_initial_distance = np.random.randint(40, 50)
        front_initial_distance = np.random.randint(90, 100)
        initial_distance = [rear_initial_distance, middle_initial_distance, front_initial_distance]
        initial_velocity = [_ for _ in np.random.randint(22, 28, 3)]
        safe_distance_fm = initial_velocity[1] * 2
        safe_distance_mr = initial_velocity[0] * 1.5
        dt = 0.1
        env = CarEnv(initial_distance, initial_velocity, dt, render=False)
        pid = Car_PIDController(10, 0.01, 10.5)
        done = False
        state = env.reset()
        while not done:
            distance_from_front = state[4] - state[2]
            error = distance_from_front - safe_distance_fm
            obs = env.observe()
            # a1 is accleartion of rear car, a2 is acceleration of middle car, a3 is acceleration of front car
            a1, a3 = ddpg_agent.select_action(obs, mode='normal')
            a2 = pid.control(error, dt)
            actions = [a1, a2, a3]
            logging.info(f'Actions: {actions}')
            next_obs, front_reward, rear_reward, done = env.step(actions)
            #next_obs, reward, done = env.step(actions)
            reward = front_reward + rear_reward
            total_reward += reward
            ddpg_agent.remember(obs, [a1, a3], reward, next_obs, done)
            critic_loss, actor_loss = ddpg_agent.update()
            if critic_loss is not None and actor_loss is not None:
                # print(f'Episode {episode+1}, Critic loss: {critic_loss}, Actor loss: {actor_loss}')
                c_loss += critic_loss
                a_loss += actor_loss
            #env.render()
            state = env.state
            env.history.append(env.state)
            time.sleep(dt)
        if (episode + 1) % 100 == 0:
            ddpg_agent.save_model(f'car_ddpg_actor_{episode+1}.pth', f'car_ddpg_critic_{episode+1}.pth')
        logging.info(f'Episode {episode+1}, Critic loss: {c_loss/len(env.history)}, Actor loss: {a_loss/len(env.history)}')
        logging.info(f'Episode {episode+1}, Reward: {total_reward}')
        loss.append([c_loss/len(env.history), a_loss/len(env.history)])
        reward_plot.append(total_reward)
        time.sleep(1)
    # plot loss
    loss = np.array(loss)
    plt.plot(loss[:, 0], label='Critic loss')
    plt.plot(loss[:, 1], label='Actor loss')
    plt.legend()
    plt.savefig('car_ddpg_loss.png')
    # plot reward
    plt.plot(reward_plot)
    plt.savefig('car_ddpg_reward.png')

def train_water_ddpg(num_episodes=1000):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Metal Performance Shaders) on Mac GPU.")
    else:
        device = torch.device('cpu')
        print("MPS not available. Using CPU.")

    ddpg_agent = DDPG_Water(state_dim=3, action_dim=2, device=device)
    loss = []
    reward_plot = []
    logging.basicConfig(level=logging.INFO)

    # Replay buffer warm-up
    initial_warmup_steps = 10000
    initial_flow_rate = np.random.randint(6, 11)
    initial_pressure = np.random.randint(3, 5)
    initial_viscosity = round(np.random.uniform(0.9, 1.3), 2)
    initial_state = [initial_flow_rate, initial_pressure, initial_viscosity]
    initial_temperature = np.random.randint(60, 70)
    env = WaterEnv(initial_state, initial_temperature, dt=1)
    state = env.reset()

    pid = Water_PIDController(kp=1.1, ki=0.18, kd=0.01, setpoint=initial_flow_rate)
    print("Populating replay buffer...")
    for _ in range(initial_warmup_steps):
        obs = env.observe()
        pump_speed = pid.compute(state[0], dt=env.dt)
        chemical_addition, temperature_control = ddpg_agent.select_action(obs)
        action = [pump_speed, chemical_addition, temperature_control]
        next_obs, reward, done = env.step(action)
        ddpg_agent.remember(obs, [chemical_addition, temperature_control], reward, next_obs, done)
        if done:
            state = env.reset()
        else:
            state = env.state

    print("Replay buffer warm-up complete. Starting training...")
    # use tqdm for progress bar
    for episode in tqdm.tqdm(range(num_episodes), desc="Training Episodes"):
        c_loss = 0
        a_loss = 0
        total_reward = 0
        initial_flow_rate = np.random.randint(6, 11)
        initial_pressure = np.random.randint(3, 5)
        initial_viscosity = round(np.random.uniform(0.9, 1.3), 2)
        initial_state = [initial_flow_rate, initial_pressure, initial_viscosity]
        initial_temperature = np.random.randint(60, 70)
        env = WaterEnv(initial_state, initial_temperature, dt=1)
        pid = Water_PIDController(kp=1.1, ki=0.18, kd=0.01, setpoint=initial_flow_rate)
        done = False
        state = env.reset()
        while not done:
            obs = env.observe()
            pump_speed = pid.compute(state[0], dt=env.dt)
            chemical_addition, temperature_control = ddpg_agent.select_action(obs, mode='normal')
            action = [pump_speed, chemical_addition, temperature_control]
            print(f'pump_speed: {pump_speed}, chemical_addition: {chemical_addition}, temperature_control: {temperature_control}')
            print(f'environemnt state: {state}')
            next_obs, reward, done = env.step(action)
            total_reward += reward
            ddpg_agent.remember(obs, [chemical_addition, temperature_control], reward, next_obs, done)
            critic_loss, actor_loss = ddpg_agent.update()
            if critic_loss is not None and actor_loss is not None:
                c_loss += critic_loss
                a_loss += actor_loss
            state = env.state
            env.history.append(env.state)
            time.sleep(env.dt)
        if (episode + 1) % 100 == 0:
            ddpg_agent.save_model(f'water_ddpg_actor_{episode+1}.pth', f'water_ddpg_critic_{episode+1}.pth')
        logging.info(f'Episode {episode+1}, Critic loss: {c_loss/len(env.history)}, Actor loss: {a_loss/len(env.history)}')
        logging.info(f'Episode {episode+1}, Reward: {total_reward}')
        loss.append([c_loss/len(env.history), a_loss/len(env.history)])
        reward_plot.append(total_reward)
        time.sleep(1)

if __name__ == '__main__':
    #train_car_ddpg()
    train_water_ddpg()