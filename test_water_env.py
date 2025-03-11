import torch
import numpy as np
import matplotlib.pyplot as plt
from water_system import DDPG_Water, WaterEnv, Water_PIDController
import time

def test_car_ddpg(initial_state, initial_temperature):
    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Metal Performance Shaders) on Mac GPU.")
    else:
        device = torch.device('cpu')
        print("MPS not available. Using CPU.")

    # Load trained DDPG agent
    ddpg_agent = DDPG_Water(state_dim=3, action_dim=2, device=device)
    actor_state_dict = torch.load(r'/Users/creak77/RL_env/XAI_RL/water_ddpg_actor_500.pth', map_location=device, weights_only=True)
    critic_state_dict = torch.load(r'/Users/creak77/RL_env/XAI_RL/water_ddpg_critic_500.pth', map_location=device, weights_only=True)
    ddpg_agent.actor.load_state_dict(actor_state_dict)
    ddpg_agent.critic.load_state_dict(critic_state_dict)
    ddpg_agent.actor.eval()
    ddpg_agent.critic.eval()

    # Initialize environment
    env = WaterEnv(initial_state, initial_temperature)
    pid = Water_PIDController(kp=1.6, ki=0.18, kd=0.01, setpoint=initial_state[0])
    state = env.reset()
    done = False
    episode_reward = 0
    acc = []
    q = []
    # Run simulation
    while not done:
        obs = env.observe()
        pump_speed = pid.compute(state[0], env.dt)
        chemical_addition, temperature_control = ddpg_agent.select_action(obs)
        action = [pump_speed, chemical_addition, temperature_control]
        _a = np.array([chemical_addition, temperature_control])
        print('chemical_addition:', chemical_addition)
        print('temperature_control:', temperature_control)
        acc.append(action)
        Q = ddpg_agent.critic(torch.FloatTensor(obs).unsqueeze(0).to(device), torch.FloatTensor(_a).unsqueeze(0).to(device))
        q.append(np.exp(Q.item()))
        next_state, reward, done = env.step(action)
        episode_reward += reward
        state = env.state
        env.history.append(env.state)
    
    # Plot history
    history = np.array(env.history)
    acc = np.array(acc).T
    plot_history(history, acc, env.dt, q)
    #plot_Q_values(q)

def plot_history(history, acc, dt, q):
    t = np.arange(0, len(history)*dt, dt)
    fig, axs = plt.subplots(5, 1, figsize=(10, 10))
    axs[0].plot(t, history[:, 0], label='Flow Rate')
    axs[0].set_ylabel('Flow Rate')
    axs[0].legend()
    axs[1].plot(t, history[:, 1], label='Pressure')
    axs[1].set_ylabel('Pressure')
    axs[1].legend()
    axs[2].plot(t, history[:, 2], label='Viscosity')
    axs[2].set_ylabel('Viscosity')
    axs[2].legend()
    axs[3].plot(t, acc[1], label='Chemical Addition')
    axs[3].set_ylabel('Chemical Addition')
    axs[3].legend()
    axs[4].plot(t, acc[2], label='Temperature Control')
    axs[4].set_ylabel('Temperature Control')
    axs[4].legend()
    plt.tight_layout()
    plt.savefig('water_ddpg_history.png')
    plt.show()

def plot_Q_values(q):
    plt.plot(q)
    plt.xlabel('Time')
    plt.ylabel('Q-value')
    plt.savefig('water_ddpg_q_values.png')
    plt.show()

def plot_Q():
    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Metal Performance Shaders) on Mac GPU.")
    else:
        device = torch.device('cpu')
        print("MPS not available. Using CPU.")

    # Load trained DDPG agent
    ddpg_agent = DDPG_Water(state_dim=3, action_dim=2, device=device)
    actor_state_dict = torch.load(r'/Users/creak77/RL_env/XAI_RL/water_ddpg_actor_500.pth', map_location=device, weights_only=True)
    critic_state_dict = torch.load(r'/Users/creak77/RL_env/XAI_RL/water_ddpg_critic_500.pth', map_location=device, weights_only=True)
    ddpg_agent.actor.load_state_dict(actor_state_dict)
    ddpg_agent.critic.load_state_dict(critic_state_dict)
    ddpg_agent.actor.eval()
    ddpg_agent.critic.eval()

    # Initialize environment
    #state = [10, 4, 0] # initial state
    state = [9.875, 5.5, 1.5]
    #state = [9.9, 9, 3]
    temp = 65 # initial temperature
    chemical_addition = np.linspace(-0.1, 0.1, 10)
    temperature_control = np.linspace(-5, 5, 10)
    C, T = np.meshgrid(chemical_addition, temperature_control, indexing='ij')
    Q_values = np.zeros_like(C, dtype=np.float32)
    for i in range(len(C)):
        for j in range(len(T)):
            action = [chemical_addition[i], temperature_control[j]]
            Q_values[i, j] = ddpg_agent.critic(torch.FloatTensor(state).unsqueeze(0).to(device), torch.FloatTensor(action).unsqueeze(0).to(device)).item()

    Z = Q_values #if np.max(Q_values) < 0 else -Q_values 
    Z = np.exp(Z)
    Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))

    plt.contourf(C, T, Z, cmap='viridis')
    plt.colorbar()
    plt.xlabel('Chemical Addition')
    plt.ylabel('Temperature Control')
    # plt.savefig('water_ddpg_q_values.png')
    plt.show()

if __name__ == '__main__':
    # initial_state = [10, 4, 0]
    # initial_temperature = 65
    # test_car_ddpg(initial_state, initial_temperature)
    plot_Q()