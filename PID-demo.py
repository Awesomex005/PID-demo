#
# Copyright (c) 2021 Alex Song <zheng.song@intel.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import matplotlib.pyplot as plt

# J/kg*C
HEAT_CAPACITY_WATER = 4200
# J/s or W
INENER_HEATER = 1600
# Celsius degree
INTI_TEMP = 42
# kg          
AMOUNT_OF_WATER = 1
# J/s
INNER_HEAT_LOSE_RATE = 20
# J/rpm
HEAT_DISSIPATION_RATE_FAN = 0.20
COOLER_RPM_MAX = 20000
COOLER_RPM_MIN = 6000
COOLER_RPM_MIN_PERCENTAGE = 0.30

# then the heat balance will be hit at 7900 RPM, 0.395;
# where 0.395 * 20000 * 0.20 + 20 = 1600

# Celsius degree
SETPOINT = 40

# PID coefficients
# You may twiddle these coefficients to get an insight on how them affects the PV (Process Value)(Temperature) curve
Pcoefficient = -0.5
Icoefficient = -0.05
Dcoefficient = -0.25


def water_en2temp(J):
    return J/(HEAT_CAPACITY_WATER * AMOUNT_OF_WATER)

def cooler_coolratefunc(rpm): 
    # J/s
    return -rpm*HEAT_DISSIPATION_RATE_FAN

class PID_CONTROLLER:
    def __init__(self, P, I=0, D=0, dt=1) -> None:
        self.P = P
        self.I = I
        self.D = D
        self.dt = dt
        self.integral = 0
        self.previous_error = 0
    def __call__(self, error):
        self.integral += error * self.dt
        derivative = (error - self.previous_error) / self.dt
        output = self.P*error + self.I*self.integral + self.D*derivative
        self.previous_error = error
        return output

def time_flows(pidcontoller, sec=200):
    energy_inwhole = HEAT_CAPACITY_WATER * INTI_TEMP * AMOUNT_OF_WATER
    time_line = [0,1,2]
    water_temperature = [water_en2temp(energy_inwhole), water_en2temp(energy_inwhole), water_en2temp(energy_inwhole)]
    pid_outp = [0, 0, 0]
    rpm_outp = [0, 0, 0]
    elapsed_time = 2
    while elapsed_time < sec:
        messured_tmp = water_temperature[-3] # Roughly simulates the system latencies, such as measurement latency, rpm drive latency.
                                             # To model a pefect no latency system, change -3 to -1.
        elapsed_time += 1
        energy_inwhole += INENER_HEATER
        energy_inwhole -= INNER_HEAT_LOSE_RATE

        # where PID comes in
        error = SETPOINT - messured_tmp
        pid_outp_ = pidcontoller(error)

        if pid_outp_ > 1.0:
            pid_outp_ = 1.0
        if pid_outp_ < COOLER_RPM_MIN_PERCENTAGE:
            pid_outp_ = COOLER_RPM_MIN_PERCENTAGE

        rpm_outp_ = pid_outp_ * COOLER_RPM_MAX
        energy_inwhole += cooler_coolratefunc(rpm_outp_)
        time_line.append(elapsed_time)
        water_temperature.append(water_en2temp(energy_inwhole))
        pid_outp.append(pid_outp_)
        rpm_outp.append(rpm_outp_)

    return time_line, water_temperature, pid_outp, rpm_outp


p_controller = PID_CONTROLLER(Pcoefficient)
pi_controller = PID_CONTROLLER(Pcoefficient, Icoefficient)
pid_controller = PID_CONTROLLER(Pcoefficient, Icoefficient, Dcoefficient)

time_line, water_temperature_p, p_outp, p_rpm_outp = time_flows(p_controller)
time_line, water_temperature_pi, pi_outp, pi_rpm_outp = time_flows(pi_controller)
time_line, water_temperature_pid, pid_outp, pid_rpm_outp = time_flows(pid_controller)

# Data visualization
fig, (ax1) = plt.subplots(1, 1, figsize=(16, 8))
line_p, = ax1.plot(time_line, water_temperature_p, 'y', label='P controller')
line_pi, = ax1.plot(time_line, water_temperature_pi, 'c', linewidth=2.0, label='PI controller')
line_pid, = ax1.plot(time_line, water_temperature_pid, 'b', linewidth=2.5, label='PID controller')
ax1.plot(time_line, np.ones(len(time_line))*SETPOINT, 'r', linewidth=1.5, label='SETPOINT')
plt.xlabel('Time')
plt.ylabel('Termperature')
legend = plt.legend(loc='upper right', shadow=True, fontsize='x-small')

## Setup the annotation
annot = ax1.annotate("", xy=(0,0), xytext=(-20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=-0.01"))
annot.set_visible(False)

def update_annot(annot, line, ind, pid_outp, rpm_outp):
    x,y = line.get_data()
    annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
    text = "{}: Sec {}, Temp {:.3f}, RPM {:.3f}, PID {:.3f}".format(line.get_label(),
                                                                    annot.xy[0],
                                                                    annot.xy[1],
                                                                    rpm_outp[ind["ind"][0]],
                                                                    pid_outp[ind["ind"][0]])
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax1:
        for line, pid_outp_, rpm_outp_ in zip([line_p, line_pi, line_pid],
                                                [p_outp, pi_outp, pid_outp],
                                                [p_rpm_outp, pi_rpm_outp, pid_rpm_outp]):
            cont, ind = line.contains(event)
            if cont:
                update_annot(annot, line, ind, pid_outp_, rpm_outp_)
                annot.set_visible(True)
                fig.canvas.draw_idle()
                break
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()