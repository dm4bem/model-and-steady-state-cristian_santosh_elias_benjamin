import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dm4bem

controller = False
neglect_air_glass_capacity = True # Always true in this model
imposed_time_step = False
Δt = 1800    # s, imposed time step

# MODEL
# =====
# Thermal circuit
TC = dm4bem.file2TC('./MODEL/TC.csv', name='', auto_number=False)

# by default TC['G']['q6'] = 0 and TC['G']['q14'] = 0, i.e. Kp -> 0, no controller (free-floating)

if controller:
    TC['G']['q6'] = 1e3        # Kp -> ∞, almost perfect controller
    TC['G']['q14'] = 1e3 

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
Φi1 = 0 * np.ones(n)         # solar radiation 
Φo2 = Φo1 = Φi2 = Φi1        # solar radiation


data = {'To': To, 'Ti1': Ti1, 'To': To, 'Ti2': Ti2, 'To': To, 'Φo1': Φo1, 'Φi1': Φi1, 'Φi2': Φi2, 'Φo2': Φo2}
input_data_set = pd.DataFrame(data, index=time)

# inputs in time from input_data_set
u = dm4bem.inputs_in_time(us, input_data_set)

# Time integration with Euler method
# Initial conditions
θ_exp = pd.DataFrame(index=u.index)     # empty df with index for explicit Euler
θ_imp = pd.DataFrame(index=u.index)     # empty df with index for implicit Euler

θ0 = 0.0                    # initial temperatures
θ_exp[As.columns] = θ0      # fill θ for Euler explicit with initial values θ0
θ_imp[As.columns] = θ0      # fill θ for Euler implicit with initial values θ0

I = np.eye(As.shape[0])     # identity matrix
for k in range(u.shape[0] - 1):
    θ_exp.iloc[k + 1] = (I + dt * As) @ θ_exp.iloc[k] + dt * Bs @ u.iloc[k]
    θ_imp.iloc[k + 1] = np.linalg.inv(I - dt * As) @ (θ_imp.iloc[k] + dt * Bs @ u.iloc[k])

# output for rooms

y_exp = (Cs @ θ_exp.T + Ds @  u.T).T
y_imp = (Cs @ θ_imp.T + Ds @  u.T).T

# plot results
y = pd.concat([y_exp, y_imp], axis=1)
# Flatten the two-level column labels into a single level
y.columns = y.columns = ['y_explicit_Room1', 'y_explicit_Room2', 'y_implicit_Room1', 'y_implicit_Room2']

ax = y.plot()
ax.set_xlabel('Time')
ax.set_ylabel('Indoor temperature, $\\theta_i$ / °C')
ax.set_title(f'Time step: $dt$ = {dt:.0f} s; $dt_{{max}}$ = {Δtmax:.0f} s')
plt.show()

print('Steady-state indoor temperature obtained with:')
print(f'- steady-state response to step input:{y_exp["θ5"].tail(1).values[0]:.4f} °C')

#TRY ALSO WITH THE CONTROLLER ON 
