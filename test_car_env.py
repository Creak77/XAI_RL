import torch
import numpy as np
import matplotlib.pyplot as plt
from av_system import DDPG_Car, CarEnv, Car_PIDController
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
    pid = Car_PIDController(10, 0.01, 10.5)
    state = env.reset()
    done = False
    episode_reward = 0
    acc = []
    q = []
    # Run simulation
    while not done:
        distance_from_front = state[4] - state[2]
        error = distance_from_front - safe_distance
        obs = env.observe()
        #print(f'Observation: {obs}')
        a1, a3 = ddpg_agent.select_action(obs, mode)
        _a = np.array([a1, a3])
        Q = ddpg_agent.critic(torch.FloatTensor(obs).unsqueeze(0).to(device), torch.FloatTensor(_a).unsqueeze(0).to(device))
        a2 = pid.control(error, dt)
        actions = [a1, a2, a3]
        acc.append(actions)
        # q.append(np.exp(Q.item()))
        q.append(np.exp(Q.item()))
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
    q = np.array(q)
    print(f'q: {q}')
    plot_history(history, acc, dt, mode, q)

def plot_history(history, acc, dt, mode, q):
    t = np.arange(0, len(history) * dt, dt)
    fig, axs = plt.subplots(4, 1, figsize=(12, 12))
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
    axs[3].plot(t, q, label='impact_value', color='r')
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('impact')
    axs[3].legend()
    if mode == 'normal':
        plt.savefig('test_-4_rear_acc.png')
    else:
        plt.savefig('test_-6_rear_acc.png')
    plt.show()

def plot_Q(mode='acc'):
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

    if mode == 'acc':
        state = np.array([60, 40, 24, 24, 24]) #0 sec
        #state = np.array([50, 38, 15, 22, 18]) #2 sec
        #state = np.array([40, 25, 0, 10, 6]) #4 sec
        a1 = np.linspace(-6, 2, 7)
        a3 = np.linspace(-8, 2, 11)
        A1, A3 = np.meshgrid(a1, a3, indexing='ij')
        Q_values = np.zeros_like(A1, dtype=np.float32)
        for i in range(len(a1)):
            for j in range(len(a3)):
                _a = np.array([a1[i], a3[j]])
                Q = ddpg_agent.critic(torch.FloatTensor(state).unsqueeze(0).to(device), torch.FloatTensor(_a).unsqueeze(0).to(device))
                Q_values[i, j] = Q.item()

        Z = Q_values #if np.max(Q_values) < 0 else -Q_values 
        Z = np.exp(Z)
        Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))

        plt.figure(figsize=(10, 6))
        contour = plt.contourf(A1, A3, Z, levels=50, cmap="viridis")
        plt.colorbar(contour, label="Impact Value")
        plt.xlabel("Rear acc", fontsize=12)
        plt.ylabel("Front acc", fontsize=12)
        plt.title("Critic Value Function Visualization", fontsize=14)
        plt.grid(False)
        plt.savefig('impact_value_acc_0s_final.png')
        plt.show()
    
    if mode == 'dis':
        state = np.array([0, 0, 24, 22, 24]) #0 sec
        acc = np.array([2, -8]) #0 sec
        # state = np.array([0, 0, 15, 22, 18]) #2 sec
        # acc = np.array([-6, -8]) #2 sec
        # state = np.array([0, 0, 0, 10, 6]) #4 sec
        # acc = np.array([-6, -8]) #4 sec
        front_distance = np.linspace(5, 50, 20)
        rear_distance = np.linspace(5, 50, 20)
        D1, D3 = np.meshgrid(front_distance, rear_distance, indexing='ij')
        Q_values = np.zeros_like(D1, dtype=np.float32)
        for i in range(len(front_distance)):
            for j in range(len(rear_distance)):
                state[0] = front_distance[i]
                state[1] = rear_distance[j]
                Q = ddpg_agent.critic(torch.FloatTensor(state).unsqueeze(0).to(device), torch.FloatTensor(acc).unsqueeze(0).to(device))
                Q_values[i, j] = Q.item()
                #print(f'Q: {Q.item()} when front_distance: {front_distance[i]}, rear_distance: {rear_distance[j]}')
        
        Z = (Q_values - np.min(Q_values)) / (np.max(Q_values) - np.min(Q_values))
        Z = np.exp(-Z)
        Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
        #Z = 1 - Z
        

        plt.figure(figsize=(10, 6))
        contour = plt.contourf(D1, D3, Z, levels=50, cmap="viridis")
        plt.colorbar(contour, label="Impact Value")
        plt.xlabel("Middle to Front", fontsize=12)
        plt.ylabel("Middle to Rear", fontsize=12)
        plt.title("Critic Value Function Visualization", fontsize=14)
        plt.grid(False)
        plt.savefig('impact_value_dis_0s_final.png')
        plt.show()

if __name__ == '__main__':
    rear_initial_distance = -10
    middle_initial_distance = 30
    front_initial_distance = 90
    initial_distance = [rear_initial_distance, middle_initial_distance, front_initial_distance]
    initial_velocity = [22, 24, 24]
    test_car_ddpg(mode='aggresive', initial_distance=initial_distance, initial_velocity=initial_velocity, render=False)
    #plot_Q(mode='acc')
    #plot_Q(mode='dis')