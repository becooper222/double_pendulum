import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from scipy.integrate import solve_ivp
import os

# Get initial angles from the user in degrees and convert to radians
initial_angle1_deg = 128.4 # float(input("Enter the initial angle for the first pendulum (in degrees): "))
initial_angle2_deg = 133.4 # float(input("Enter the initial angle for the second pendulum (in degrees): "))
total_time = 60 # int(input("Enter the total time for the simulation (in seconds): "))

# TODO: Manually set position of pendulum arms before starting simulation
# TODO: Add a button to start the simulation from the beginning
# TODO: Add a button to pause the simulation

# Create a folder to save the plots
if not os.path.exists(f'{os.getcwd()}/saved_pendulums'):
    os.mkdir(f'{os.getcwd()}/saved_pendulums')

# Constants
g = 9.81  # acceleration due to gravity
L1, L2 = 1.0, 1.0  # lengths of the pendulums (fixed)
m1, m2 = 0.5, 0.5  # masses of the pendulums

# Function to convert degrees to radians
def degrees_to_radians(degrees):
    return degrees * np.pi / 180

# Equations of motion
def equations(t, y, L1, L2, m1, m2):
    theta1, z1, theta2, z2 = y
    c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)

    theta1dot = z1
    z1dot = (m2 * g * np.sin(theta2) * c - m2 * s * (L1 * z1**2 * c + L2 * z2**2) -
             (m1 + m2) * g * np.sin(theta1)) / L1 / (m1 + m2 * s**2)
    theta2dot = z2
    z2dot = ((m1 + m2) * (L1 * z1**2 * s - g * np.sin(theta2) + g * np.sin(theta1) * c) +
             m2 * L2 * z2**2 * s * c) / L2 / (m1 + m2 * s**2)
    return [theta1dot, z1dot, theta2dot, z2dot]

initial_angle1 = degrees_to_radians(initial_angle1_deg)
initial_angle2 = degrees_to_radians(initial_angle2_deg)

# Initial conditions: [theta1, theta1_dot, theta2, theta2_dot]
y0 = [initial_angle1, 0, initial_angle2, 0]

# Time array
t_span = (0, total_time)
fps = 30
resolution_factor = 2  # Increase this factor to make the trace line smoother
t_eval = np.linspace(t_span[0], t_span[1], t_span[1] * fps * resolution_factor)

# Solve ODE
solution = solve_ivp(equations, t_span, y0, t_eval=t_eval, args=(L1, L2, m1, m2))

# Extract angles
theta1, theta2 = solution.y[0], solution.y[2]

# Convert to Cartesian coordinates
x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)


# Set up the figure and axis with a black background
fig, ax = plt.subplots(facecolor='black')
ax.set_facecolor('black')
# Title
ax.set_title(f'Double Pendulum Simulation\nInitial Angles: {initial_angle1_deg}°, {initial_angle2_deg}°', color='white')

ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)

# Set the line color and the trace color to be visible against the black background
line, = ax.plot([], [], 'o-', lw=2, color='white')
trace, = ax.plot([], [], 'r-', lw=1)


def init():
    line.set_data([], [])
    trace.set_data([], [])
    return line, trace

def update(frame):
    thisx = [0, x1[frame], x2[frame]]
    thisy = [0, y1[frame], y2[frame]]

    line.set_data(thisx, thisy)

    trace.set_data(x2[:frame+1], y2[:frame+1])
    return line, trace

ani = FuncAnimation(fig, update, frames=range(len(t_eval)),
                    init_func=init, blit=True, interval=1000/fps/resolution_factor, repeat=False)


# Function to break the animation
def break_animation():
    plt.close()

# Function to save the plot
def save_plot(event):
    filename = f'{os.getcwd()}/saved_pendulums/theta1_{initial_angle1_deg}_theta2_{initial_angle2_deg}_{t_span[1]}s.png'
    plt.savefig(filename)
    print(f'Plot saved as {filename}')
    break_animation()

# Add a button to save the plot
ax_button = plt.axes([0.8, 0.025, 0.1, 0.04])
btn_save = Button(ax_button, 'Save Plot', color='gray', hovercolor='lightgray')
btn_save.on_clicked(save_plot)


plt.show()
