'''
roadmap
1) Model the dynamical system for the water system.
broken pump, insuffient water quality as potential failure modes
input: pump speed, chemical addition, ultrafiltration
output: flow rate, composite water quality (pH + turbidity)
2) Implement a simple MPC controller to maintain the water quality within safe limits by 
passing chemical addition and ultrafiltration unit
3) Implement a PID controller to maintain the flow rate within safe limits.
4) Train a DDPG agent to crash the water system.
'''
'''
Scenario 1: Water System Dynamics

Objective: Model the dynamics of a water system with flow rate, pH, and turbidity as state variables.

Failure: no sufficient water quality and broken pump

Impact score: depend on how long the system water can be used until empty tank

RL Agents:

 Observations:
    (1) Flow rate
    (2) pH
    (3) Turbidity

RL Actions:
    (1) Pump speed
    (2) Chemical addition
    (3) ultrafiltration
 
Chain of events:
    (1) Flow rate decreases due to broken pump
    (2) pH increases due to insufficient water quality
    (3) Turbidity increases due to unsufficient ultrafiltration
 
Considerations:
    (1) The system should be able to maintain the flow rate within safe limits.
    (2) The system should be able to maintain the water quality within safe limits.
    (3) The system should be able to maintain the turbidity within safe limits.

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from ddpg_modules import ReplayBuffer, Actor, Critic
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as nn_utils

class WaterEnv:
    def __init__(self, initial_state, initial_temperature, dt=1):
        self.initial_state = initial_state
        self.temperature = initial_temperature # Water temperature in °C
        self.dt = dt
        self.time = 0.0
        #self.last_action = np.array([0, 0, 0])
        self.last_state = self.initial_state.copy()
        self.pressure_limit = 10.0  # Safety limit for pressure
        self.history = []

    def reset(self):
        self.state = self.initial_state.copy()  # [Flow (L/s), Pressure (bar), Viscosity]
        return self.state    

    def observe(self):
        return self.state  

    def compute_pump_pressure(self):
        """
        Compute dynamic pressure using Bernoulli's equation.
        Convert flow_rate from L/s to m³/s.
        """
        Q_m3 = self.state[0] / 1000.0  # Convert L/s to m³/s
        rho = 1000  # kg/m³
        D = 0.2  # m
        A = (np.pi * D**2) / 4  # m²
        v = Q_m3 / A  # Flow velocity in m/s
        dynamic_pressure = 0.5 * rho * v**2 / 1e5  # Convert Pa -> bar
        return dynamic_pressure

    def compute_flow_decay(self):
        """
        Compute the pressure drop due to friction (Darcy-Weisbach).
        Convert flow_rate from L/s to m³/s.
        """
        Q_m3 = self.state[0] / 1000.0  # m³/s
        rho = 1000  # kg/m³
        f = 0.02  # friction factor
        L = 10    # m
        D = 0.2   # m
        A = (np.pi * D**2) / 4  # m²
        v = Q_m3 / A  # m/s
        # Pressure drop: ΔP = f*(L/D)*(rho*v^2/2)
        pressure_drop = f * (L / D) * (rho * v**2 / 2) / 1e5  # Convert Pa -> bar
        return pressure_drop
    
    def compute_chemical_effect(self):
        """
        Apply the Jones–Dole equation.
        chemical_dosing: mass flow rate of NaCl added (in g/s)
        Uses the current water flow (in L/s) from self.state for dilution.
        """
        A = 0.075  # Charge-charge interaction coefficient
        B = 0.1    # Solute-solvent interaction coefficient
        M_NaCl = 58.44  # g/mol
        
        # Use the current water flow (avoid division by zero)
        Q = self.state[0] if self.state[0] != 0 else 1.0  
        # Calculate molarity in mol/L (g/s divided by (g/mol * L/s))
        molarity = (1 / Q) / M_NaCl  
        relative_viscosity = 1 + A * np.sqrt(molarity) + B * molarity
        
        return relative_viscosity

    def water_dynamic(self, x, u):
        """
        Simplified state dynamics.
        x: [Flow rate (L/s), Pressure, Viscosity]
        u: [pump_speed, chemical_dosing (g/s), water_temp (°C)]
        """
        pump_speed, chemical_dosing, water_temp = u
        flow_rate, pressure, viscosity = x

        # Compute individual effects with proper unit conversions
        pressure_increase = self.compute_pump_pressure()     # Increase in pressure due to pump
        #print(f'pressure_increase: {pressure_increase}')
        friction_loss = self.compute_flow_decay()            # Pressure drop due to friction
        chem_effect = self.compute_chemical_effect()  # Target viscosity multiplier
        # Constants defining system behavior
        viscosity_factor = 0.2  # Viscosity effect on flow rate
        pump_flow_ratio = 1.0  # Pump flow contribution
        pump_pressure_ratio = 0.5  # Pump pressure contribution
        temp_effect = -0.01                     # Temperature effect on viscosity

        # Updated A Matrix
        A = np.array([
            [-friction_loss, 0, -viscosity_factor],
            [0, pressure_increase, 0],
            [0, 0, 0]
        ])

        # Updated B Matrix
        B = np.array([
            [pump_flow_ratio, 0, 0],
            [pump_pressure_ratio, 0, 0],
            [0, chem_effect, temp_effect]
        ])

        C = np.eye(3)
        D = np.zeros((3, 3))

        dxdt = A @ x + B @ u
        dy = C @ x + D @ u
        return dxdt, dy

    def simulate_water_dynamics(self, u, dt):
        ode_func = lambda t, x: self.water_dynamic(x, u)[0]
        t_span = (0, dt)
        sol = solve_ivp(ode_func, t_span, self.state, method='RK45', t_eval=[dt])
        new_state = sol.y[:, -1]
        new_state[0] = max(new_state[0], 0.0)  # Ensure nonnegative flow rate
        self.state = new_state
        self.time += dt
        return new_state

    def step(self, actions):
        #self.last_action = actions
        self.last_state = self.state
        if self.temperature < 5:
            actions[2] = 0  # Stop cooling if temperature is too low
        self.temperature += actions[2] * self.dt
        self.state = self.simulate_water_dynamics(actions, self.dt)
        reward = self.reward()
        done = self.done()
        return self.state, reward, done

    def reward(self):
        # reward based on viscosity and pressure increase
        reward = self.state[2] - self.last_state[2] + self.state[1] - self.last_state[1]
        return reward

    def done(self):
        flow_rate, pressure, viscosity = self.state
        if flow_rate < 5 or flow_rate > 25 or pressure > self.pressure_limit or pressure < 0 or viscosity > 5 or viscosity < 0:
            return True
        return False

    def render(self):
        flow_rate, pressure, viscosity = self.state
        print(f"Time: {self.time:.2f} sec | Flow Rate: {flow_rate:.2f} L/s | Pressure: {pressure:.2f} | Viscosity: {viscosity:.2f}")

class Water_PIDController:
    def __init__(self, kp, ki, kd, setpoint, output_limits=(0, 10)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self._integral = 0
        self._prev_error = 0
        self.output_limits = output_limits

    def compute(self, measurement, dt):
        error = self.setpoint - measurement
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        self._prev_error = error
        return np.clip(output, *self.output_limits)
    
class DDPG_Water:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.0001)
        self.memory = ReplayBuffer(capacity=100000)
        self.batch_size = 128
        self.gamma = 0.99
        self.tau = 0.005

    def select_action(self, state, mode='normal'):
        # print(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state_tensor).detach().cpu().numpy()[0]
        # Scale chemical_addition from [-1, 1] to [-0.1, 0.1]
        chemical_addition = action[0] * 0.1
        # Scale temperature_control from [-1, 1] to [-5, 5]
        temperature_control = action[1] * 5
        return chemical_addition, temperature_control
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return None, None
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
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
        nn_utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        # Compute actor loss
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn_utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        # Update target networks
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return critic_loss.item(), actor_loss.item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def save_model(self, actor_path='actor.pth', critic_path='critic.pth'):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)


if __name__ == "__main__":
    # Initial state: [Flow rate (L/s), Pressure (bar), Viscosity (cP)]
    initial_state = np.array([10.0, 4.0, 0])
    initial_temperature = 60.0
    water_system = WaterEnv(initial_state, initial_temperature, dt=1)
    pid_controller = Water_PIDController(kp=1.6, ki=0.18, kd=0.01, setpoint=10)
    water_system.reset()

    states = []
    for _ in range(100):
        print("Current state:", water_system.state)
        pump_speed = pid_controller.compute(water_system.state[0], water_system.dt)
        print("Pump speed:", pump_speed)

        # Suppose we add 0.1 g/s chemical, -5.0 degC from normal
        chemical_addition = 0.1
        temperature_control = -5.0
        actions = np.array([pump_speed, chemical_addition, temperature_control])
        obs, reward, done = water_system.step(actions)
        states.append(obs)
        water_system.render()
        if done:
            print("Episode done due to safety limits.")
            break

    print("Simulation complete.")

    # Plotting the results
    time_array = np.arange(len(states)) * water_system.dt
    states = np.array(states)

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time_array, states[:, 0], label='Flow Rate')
    plt.ylabel('Flow Rate (L/s)')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time_array, states[:, 1], label='Pressure')
    plt.ylabel('Pressure')
    plt.ylim(0, 15)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time_array, states[:, 2], label='Viscosity')
    plt.ylabel('Viscosity')
    plt.xlabel('Time (s)')
    plt.legend()

    plt.tight_layout()
    plt.show()