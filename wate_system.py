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

class WaterSystem:
    def __init__(self, initial_state, dt=0.1):
        self.initial_state = initial_state
        self.dt = dt
        self.time = 0.0
        self.last_action = np.array([0, 0, 0])
        self.pressure_limit = 10.0  # Safety limit for pressure

    def reset(self):
        self.state = self.initial_state.copy()  # [Flow (L/s), Pressure, Viscosity]
        return self.state      

    def compute_chemical_effect(self, chemical_dosing):
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
        molarity = (chemical_dosing / Q) / M_NaCl  
        relative_viscosity = 1 + A * np.sqrt(molarity) + B * molarity
        
        return relative_viscosity

    def compute_pump_pressure(self, flow_rate):
        """
        Compute dynamic pressure using Bernoulli's equation.
        Convert flow_rate from L/s to m³/s.
        """
        Q_m3 = flow_rate / 1000.0  # Convert L/s to m³/s
        rho = 1000  # kg/m³
        D = 0.5  # m
        A = (np.pi * D**2) / 4  # m²
        v = Q_m3 / A  # Flow velocity in m/s
        dynamic_pressure = 0.5 * rho * v**2
        return dynamic_pressure

    def compute_viscosity_factor(self):
        """
        Compute a factor related to viscosity (using Hagen-Poiseuille formula as a guide).
        """
        L = 10  # Pipe length in m
        D = 0.5  # Pipe diameter in m
        return (128 * L) / (np.pi * D**4)

    def compute_pump_effect(self, pump_speed):
        """
        Compute the effective pump contribution to flow rate.
        pump_speed: command in RPM (or an arbitrary scale).
        Returns a flow contribution in L/s. Affinity Laws
        """
        Q_max = 100  # Maximum pump flow (L/s)
        N_max = 3600  # Maximum RPM
        return Q_max * (pump_speed / N_max)

    def compute_flow_decay(self, flow_rate):
        """
        Compute the pressure drop due to friction (Darcy-Weisbach).
        Convert flow_rate from L/s to m³/s.
        """
        Q_m3 = flow_rate / 1000.0  # m³/s
        f = 0.02  # friction factor
        L = 10    # m
        D = 0.5   # m
        A = (np.pi * D**2) / 4  # m²
        rho = 1000  # kg/m³
        v = Q_m3 / A  # m/s
        # Pressure drop: ΔP = f*(L/D)*(rho*v^2/2)
        pressure_drop = f * (L / D) * (rho * v**2 / 2)
        return pressure_drop

    def water_dynamic(self, x, u):
        """
        Simplified state dynamics.
        x: [Flow rate (L/s), Pressure, Viscosity]
        u: [pump_speed, chemical_dosing (g/s), water_temp (°C)]
        """
        pump_speed, chemical_dosing, water_temp = u
        flow_rate, pressure, viscosity = x

        # Compute individual effects with proper unit conversions
        pump_flow = self.compute_pump_effect(pump_speed)      # L/s contribution from pump
        decay_term = 0.1 * flow_rate                                  # Simple decay proportional to Q
        pressure_increase = self.compute_pump_pressure(flow_rate)     # Increase in pressure due to pump
        friction_loss = self.compute_flow_decay(flow_rate)            # Pressure drop due to friction
        pump_pressure_factor = pressure_increase - friction_loss       # Net pressure effect
        chem_effect = self.compute_chemical_effect(chemical_dosing)  # Target viscosity multiplier
        viscosity_factor = self.compute_viscosity_factor()  # Viscosity effect on flow rate
        temp_effect = 1 / (water_temp + 273.15)                     # Temperature effect on viscosity

        # Compute the new state values
        A = np.array([
            [-decay_term, 0, -pump_flow],
            [0, 0, 0.1],
            [0, 0, 0]
        ])

        B = np.array([
            [pump_flow, 0, 0],
            [pump_pressure_factor, 0, 0],
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
        self.last_action = actions
        self.state = self.simulate_water_dynamics(actions, self.dt)
        reward = self.reward()
        done = self.done()
        return self.state, reward, done, {}

    def reward(self):
        pass

    def done(self):
        flow_rate, pressure, viscosity = self.state
        if flow_rate < 5 or pressure > self.pressure_limit or viscosity > 15 or pressure < 0:
            return True
        return False

    def render(self):
        flow_rate, pressure, viscosity = self.state
        print(f"Time: {self.time:.2f} sec | Flow Rate: {flow_rate:.2f} L/s | Pressure: {pressure:.2f} | Viscosity: {viscosity:.2f}")

class PIDController:
    def __init__(self, kp, ki, kd, setpoint, output_limits=(0, 100)):
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

if __name__ == "__main__":
    # Initial state: [Flow rate (L/s), Pressure, Viscosity]
    initial_state = np.array([10.0, 7.0, 0.001])
    water_system = WaterSystem(initial_state, dt=0.1)
    pid_controller = PIDController(kp=2.5, ki=1, kd=0.005, setpoint=5.0)
    water_system.reset()

    states = []
    for _ in range(100):
        print("Current state:", water_system.state)
        pump_speed = pid_controller.compute(water_system.state[0], water_system.dt)
        chemical_addition = np.random.uniform(0, 1)
        temperature_control = np.random.uniform(15, 35)
        actions = np.array([pump_speed, chemical_addition, temperature_control])
        state, reward, done, _ = water_system.step(actions)
        states.append(state)
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