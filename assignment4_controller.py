import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dm4bem

controller = True
neglect_air_glass_capacity = True # Always true in this model
imposed_time_step = False
Δt = 1800    # s, imposed time step

folder_path = './BLDG_controller/'

# Disassembled thermal circuits
TCd = dm4bem.bldg2TCd(folder_path,TC_auto_number=True)

# Assembled thermal circuit from assembly_lists.csv
ass_lists = pd.read_csv(folder_path + 'assembly_lists.csv')

ass_matrix = dm4bem.assemble_lists2matrix(ass_lists)
TC = dm4bem.assemble_TCd_matrix(TCd, ass_matrix)

# State-space
[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)

# Eigenvalues analysis
λ = np.linalg.eig(As)[0]        # eigenvalues of matrix As
# print(f'λ = {λ}')

# time step
Δtmax = 2 * min(-1 / λ)    # max time step for stability of Euler explicit
dm4bem.print_rounded_time('Δtmax', Δtmax)

if imposed_time_step:
    dt = Δt
else:
    dt = dm4bem.round_time(Δtmax)

if dt < 10:
    raise ValueError("Time step is too small. Stopping the script.")

dm4bem.print_rounded_time('dt', dt)

# settling time
t_settle = 4 * max(-1 / λ)
dm4bem.print_rounded_time('t_settle', t_settle)

# duration: next multiple of 3600 s that is larger than t_settle
duration = np.ceil(t_settle / 3600) * 3600
dm4bem.print_rounded_time('duration', duration)

# Create input_data_set u with constant values
# ---------------------
# time vector
n = int(np.floor(duration / dt))    # number of time steps

# DateTimeIndex starting at "00:00:00" with a time step of dt
time = pd.date_range(start="2025-01-01 00:00:00",
                           periods=n, freq=f"{int(dt)}s")
 
To = 10 * np.ones(n)         # outdoor temperature 10 °C
Ti1 = 20 * np.ones(n)        # indoor temperature set point 20 °C
Ti2 = 20 * np.ones(n)        # indoor temperature set point 20 °C
Φi1 = 0 * np.ones(n)         # solar radiation absorbed by the glass
Φo2 = Φo1 = Φi2 = Φi1        # auxiliary heat sources and solar radiation


data = {'To': To, 'Ti1': Ti1, 'To': To, 'Ti2': Ti2, 'To': To, 'Φo1': Φo1, 'Φi1': Φi1, 'Φi2': Φi2, 'Φo2': Φo2}
input_data_set = pd.DataFrame(data, index=time)

# inputs in time from input_data_set
u = dm4bem.inputs_in_time(us, input_data_set)

# Input data set
input_data_set = pd.read_csv('./MODEL/input_data_set.csv', index_col=0, parse_dates=True)
# Resample hourly data to time step dt
input_data_set = input_data_set.resample(
    str(dt) + 's').interpolate(method='linear')

# Get input from input_data_set
u = dm4bem.inputs_in_time(us, input_data_set)

# initial conditions
θ0 = 20.0                   # initial temperatures
θ_exp = pd.DataFrame(index=u.index)
θ_exp[As.columns] = θ0      # Fill θ with initial valeus θ0

# time integration
I = np.eye(As.shape[0])     # identity matrix

for k in range(u.shape[0] - 1):
    θ_exp.iloc[k + 1] = (I + dt * As) @ θ_exp.iloc[k] + dt * Bs @ u.iloc[k]
    
# outputs
y = (Cs @ θ_exp.T + Ds @  u.T).T

data = pd.DataFrame({'To': input_data_set['To'],
                     'θi1': y['ow0_θ5'], 'θi2': y['ow1_θ5'],})

fig, axs = plt.subplots(1, 1)
data[['To', 'θi1', 'θi2']].plot(ax=axs,
                        xticks=[],
                        ylabel='Temperature, $θ$ / °C')
axs.legend(['$θ_{outdoor}$', '$θ_{room1}$', '$θ_{room2}$'],
              loc='upper right')


axs.set_title(f'With controller || Time step: $dt$ = {dt:.0f} s;'
                 f'$dt_{{max}}$ = {Δtmax:.0f} s')
plt.show()