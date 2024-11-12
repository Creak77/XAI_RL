'''
AV System

Objective:
Avoid colliding with a suddenly stopped front vehicle by either slowing down or 
safely switching lanes if the adjacent lane is clear

Environment Inputs:
Distance to Front Vehicle: The distance between the ego vehicle and the front vehicle
Front Vehicle Speed: The speed of the front vehicle
Distance to Next Lane Vehicle: Distance to any vehicle in the adjacent lane (y direction)
Next Lane Vehicle Speed: Speed of the vehicle in the adjacent lane
Ego Vehicle Speed: Speed of the ego vehicle
vehicle size: length and width of the vehicle

RL Actions:
Change Lane: Move to the adjacent lane if it's safe OR
Adjust Speed: Either decelerate or maintain speed based on the distance to the front vehicle

Controller:
PID controller for speed control and smooth lane change
'''

'''
AV System 2

Objective:
Act at intersections to safely navigate without traffic light signals avoiding collisions

Environment Inputs:
Intersection Type: Type of intersection (4-way, T-junction, etc.)
Traffic Density: Number of vehicles in each direction
Ego Vehicle: Speed of the ego vehicle and desired direction (straight, left, right)
Distance to Intersection: Distance to the intersection point
Intersection Angle: Angle of intersection relative to the ego vehicle
Surrounding Vehicles: Positions and speeds of vehicles around the intersection

RL Actions:
Make a turn at the intersection
Adjust speed based on traffic density and distance to the intersection

Controller:
PID controller for speed control and smooth turning at intersections
'''

'''
Water System

Objective:
Detect leaks and adjust pressure to maintain consistent water delivery across the network

Environment Inputs:
Water Pressure at nodes: Water pressure readings from multiple critical points in the network
Flow Rate: Flow rate measurements at main and branch pipelines.
Threshold Data: Baseline pressure and flow patterns, allowing detection of anomalies suggestive of leaks.

Actions:
Adjust Pump Speed: Increase or decrease pump speed to control water pressure and flow
Isolate Sections: Temporarily close off sections where a leak is suspected to minimize water loss and pinpoint the issue

Controller:
the action will change the pump speed of the system
Control for pump speed and isolation valve based on how likely a leak is detected to protect the system
'''


'''
Water System 2

Objective:
Detect external pollutants in the water supply and adjust flow rates to maintain water quality, keeping the demand

Environment Inputs:
Water Quality: Measurements of various pollutants in the water supply at multiple points to detect contamination
Flow Rate: Flow rate measurements at main and branch pipelines
Threshold Data: Baseline pollutant levels and flow patterns, allowing detection of anomalies suggestive of contamination
water demand: the amount of water required by the consumers

Actions:
Adjust flow rate: Increase or decrease flow rates to filter out pollutants at each state and maintain water quality
Open or Close Valves: Route water around contaminated areas to move backward if necessary

Controller:
Use the updated filtration rate, decision of switch position to control the system
'''

'''
Robotic Arm System

Objective:
Detect cyber-attacks on the robotic arm system and correct the system to prevent harzard movement

Environment Inputs:
System State: Position and orientation of the robotic arm
Speed: Speed of the robotic arm
Force: Force applied by the robotic arm
Desired Position: Target position for the robotic arm

Actions:
Adjust Speed: prevent sudden movements and force the robotic arm to move back to home position

Controller:
Use the updated speed and force to control the system, (probably Six-axis robot arm) dynamic model
'''