from av_system import CarEnv, PIDController, DDPG_Car
import time
import tqdm
import numpy as np

def train_car_ddpg(num_episodes=1000):
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
        pid = PIDController(10, 0.01, 10.5)
        done = False
        front_car_reward = 0
        rear_car_reward = 0
        state = env.reset()
        while not done:
            obs_rear = env.observe(car_id=0)
            obs_front = env.observe(car_id=2)
            distance_from_front = state[4] - state[2]
            error = distance_from_front - safe_distance
            # print(f'Error: {error:.2f}')
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
            # env.render()
            state = env.state
            env.history.append(env.state)
            time.sleep(dt)
        if (episode + 1) % 100 == 0:
            front_car_actor_path = f'front_car_actor_{episode+1}.pth'
            front_car_critic_path = f'front_car_critic_{episode+1}.pth'
            rear_car_actor_path = f'rear_car_actor_{episode+1}.pth'
            rear_car_critic_path = f'rear_car_critic_{episode+1}.pth'
            front_car.save_model(front_car_actor_path, front_car_critic_path)
            rear_car.save_model(rear_car_actor_path, rear_car_critic_path)
        print(f'Episode {episode+1}, Front car reward: {front_car_reward}, Rear car reward: {rear_car_reward}')
        time.sleep(2)


if __name__ == '__main__':
    train_car_ddpg()