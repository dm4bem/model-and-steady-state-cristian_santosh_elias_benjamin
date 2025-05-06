import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

import dm4bem


L = 4                 # m length of the walls room
height = 3            # m height of the walls room

Sg = L*height         # m² surface area of the glass wall
Sc2 = Si2 = 4 * Sg    # m² surface area of concrete & insulation of the 4 walls of room 2
Sc1 = Si1 = 5 * Sg    # m² surface area of concrete & insulation of the 5 walls of room 1



# room properties
air = {'Density': 1.2,                      # kg/m³
       'Specific heat': 1000}               # J/(kg·K)
pd.DataFrame(air, index=['Air'])

# wall properties
concrete_middlewall = {'Conductivity': 1.400, # W/(m·K)
            'Density': 2300.0,                # kg/m³
            'Specific heat': 880,             # J/(kg⋅K)
            'Width': 0.2,                     # m
            'Surface': Sg}                    # m²

concrete_room1 = {'Conductivity': 1.400,          # W/(m·K)
            'Density': 2300.0,              # kg/m³
            'Specific heat': 880,           # J/(kg⋅K)
            'Width': 0.2,                   # m
            'Surface': Sc1}                 # m²

insulation_room1 = {'Conductivity': 0.027,  # W/(m·K)
              'Density': 55.0,              # kg/m³
              'Specific heat': 1210,        # J/(kg⋅K)
              'Width': 0.08,                # m
              'Surface': Si1}               # m²

concrete_room2 = {'Conductivity': 1.400,    # W/(m·K)
            'Density': 2300.0,              # kg/m³
            'Specific heat': 880,           # J/(kg⋅K)
            'Width': 0.2,                   # m
            'Surface': Sc2}                 # m²

insulation_room2 = {'Conductivity': 0.027,  # W/(m·K)
              'Density': 55.0,              # kg/m³
              'Specific heat': 1210,        # J/(kg⋅K)
              'Width': 0.08,                # m
              'Surface': Si2}               # m²

glass = {'Conductivity': 1.4,               # W/(m·K)
         'Density': 2500,                   # kg/m³
         'Specific heat': 1210,             # J/(kg⋅K)
         'Width': 0.04,                     # m
         'Surface': Sg}                     # m²

wall = pd.DataFrame.from_dict({'Layer_out_1': concrete_room1,
                               'Layer_in_1': insulation_room1,
                               'Layer_out_2': concrete_room2,
                               'Layer_in_2': insulation_room2,
                               'Layer_middle': concrete_middlewall,
                               'Glass': glass},
                              orient='index')

# radiative properties
ε_wLW = 0.85    # long wave emmisivity: wall surface (concrete and insulation)
ε_gLW = 0.90    # long wave emmisivity: glass pyrex
α_wSW = 0.25    # short wave absortivity: white smooth surface
α_gSW = 0.38    # short wave absortivity: reflective blue glass
τ_gSW = 0.30    # short wave transmitance: reflective blue glass
σ = 5.67e-8     # W/(m²⋅K⁴) Stefan-Bolzmann constant

h = pd.DataFrame([{'in': 10, 'out': 25}], index=['h'])  # W/(m²⋅K)

# conduction
G_cd = wall['Conductivity'] / wall['Width'] * wall['Surface']
pd.DataFrame(G_cd, columns=['Conductance'])

# convection
Gconv_1 = h * wall['Surface'].iloc[0]       # wall
Gconv_2= h * wall['Surface'].iloc[2]        # glass
Gconv_middle = h * wall['Surface'].iloc[4]  # middle wall
Gconv_glass = h * wall['Surface'].iloc[5]   # glass


# view factor concretewall-glass
Fcwg = glass['Surface'] / concrete_room2['Surface']
# view factor insulationwall-glass
Fiwg = glass['Surface'] / insulation_room2['Surface']


# long wave radiation insulation - glass
Tm = 20 + 273   # K, mean temp for radiative exchange

GLW1 = 4 * σ * Tm**3 * ε_wLW / (1 - ε_wLW) * wall['Surface']['Layer_in_2']
GLW12 = 4 * σ * Tm**3 * Fiwg * wall['Surface']['Layer_in_2']
GLW2 = 4 * σ * Tm**3 * ε_gLW / (1 - ε_gLW) * wall['Surface']['Glass']

#The equivalent conductance, in W/K, for the radiative long-wave heat exchange between the wall and the glass window is
GLW = 1 / (1 / GLW1 + 1 / GLW12 + 1 / GLW2)

# long wave radiation concrete - glass
GLW1_c = 4 * σ * Tm**3 * ε_wLW / (1 - ε_wLW) * wall['Surface']['Layer_middle']
GLW12_c = 4 * σ * Tm**3 * Fcwg * wall['Surface']['Layer_middle']
GLW2_c = 4 * σ * Tm**3 * ε_gLW / (1 - ε_gLW) * wall['Surface']['Glass']

#The equivalent conductance, in W/K, for the radiative long-wave heat exchange between the wall and the glass window is
GLW_c = 1 / (1 / GLW1_c + 1 / GLW12_c + 1 / GLW2_c)

R_glass = 0.506 # Ohm*m^2
G_glass =  wall['Surface']['Glass']/ R_glass # S



# 17 temperature nodes
θ = ['θ0', 'θ1', 'θ2', 'θ3', 'θ4', 'θ5', 'θ6', 'θ7', 'θ8', 'θ9', 'θ10', 'θ11', 'θ12', 'θ13', 'θ14', 'θ15']

# 22 flow-rate branches
q = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18', 'q19', 'q20', 'q21']

# temperature nodes
nθ = 16      # number of temperature nodes
θ = [f'θ{i}' for i in range(nθ)]

# flow-rate branches
nq = 22     # number of flow branches
q = [f'q{i}' for i in range(nq)]


# P-controler gain
# Kp = 1e4            # almost perfect controller Kp -> ∞
# Kp = 1e-3           # no controller Kp -> 0
Kp = 0


C = wall['Density'] * wall['Specific heat'] * wall['Surface'] * wall['Width']
pd.DataFrame(C, columns=['Capacity'])

# Create a zero matrix with 22 rows and 16 columns (updated to match the highest column index used: 16)
A = np.zeros([22, 16])

A[0,0] = 1
A[1,0],A[1,1] = -1,1
A[2,1],A[2,2] = -1,1
A[3,2],A[3,3] = -1,1
A[4,3],A[4,4] = -1,1
A[5,4],A[5,5] = -1,1
A[6,5] = 1
A[7,5],A[7,6] = 1,-1
A[8,6],A[8,7] = 1,-1
A[9,7],A[9,8] = 1,-1
A[10,8],A[10,9] = 1,-1
A[11,8],A[11,10] = 1,-1
A[12,9] = 1
A[13,9],A[13,10] = -1,1
A[14,10] = 1
A[15,9],A[15,11] = -1,1
A[16,15] = 1
A[17,14],A[17,15] = 1,-1
A[18,13],A[18,14] = 1,-1
A[19,12],A[19,13] = 1,-1
A[20,11],A[20,12] = 1,-1
A[21,10],A[21,11] = 1,-1

print(A)
# Create DataFrame
df = pd.DataFrame(A, index=q, columns=θ)


G = np.array(np.hstack(
    [Gconv_1['out'],
     2*G_cd['Layer_out_1'],
     2*G_cd['Layer_out_1'],
     2*G_cd['Layer_in_1'],
     2*G_cd['Layer_in_1'],
     Gconv_1['in'],
     Kp,
     Gconv_middle['in'],
     2*G_cd['Layer_middle'],
     2*G_cd['Layer_middle'],
     GLW,
     Gconv_middle['in'],
     G_glass,
     Gconv_glass['out'],
     Kp,
     GLW_c,
     Gconv_2['out'],
     2*G_cd['Layer_out_2'],
     2*G_cd['Layer_out_2'],
     2*G_cd['Layer_in_2'],
     2*G_cd['Layer_in_2'],
     Gconv_2['in'],    
     ]))

print("10 g")
print(GLW)
print("15 g")
print(GLW_c)

# np.set_printoptions(precision=3, threshold=16, suppress=True)
# pd.set_option("display.precision", 1)
pd.DataFrame(G, index=q)

#we neglect air and glass capacitance
neglect_air_glass = True

if neglect_air_glass:
    C = np.array([0, C['Layer_out_1'], 0, C['Layer_in_1'], 0, 0, 0, C['Layer_middle'],0, 0, 0, 0, C['Layer_in_2'], 0, C['Layer_out_2'], 0])

# pd.set_option("display.precision", 3)
pd.DataFrame(C, index=θ)

b = pd.Series(['To', 0, 0, 0, 0, 0, 'Ti1', 0, 0, 0, 0, 0, 'To', 0, 'Ti2', 0, 'To', 0, 0, 0, 0, 0],
              index=q)

f = pd.Series(['Φo1', 0, 0, 0, 0, 0, 0, 0,'Φi1', 0, 0, 'Φi2', 0, 0, 0, 'Φo2'],
              index=θ)

y = np.zeros(16)            # nodes
y[[5, 10]] = 1              # nodes (temperatures) of interest 

pd.DataFrame(y, index=θ)

# thermal circuit
A = pd.DataFrame(A, index=q, columns=θ)
G = pd.Series(G, index=q)
C = pd.Series(C, index=θ)
b = pd.Series(b, index=q)
f = pd.Series(f, index=θ)
y = pd.Series(y, index=θ)

TC = {"A": A,
      "G": G,
      "C": C,
      "b": b,
      "f": f,
      "y": y}
# print(TC)

# print('A')
# print(TC['A'])
# print('G')
# print(TC['G'])
# print('C')
# print(TC['C'])
# print('b')
# print(TC['b'])
# print('f')
# print(TC['f'])


# Steady state with no flow-rate sources
bss = np.zeros(22)        # temperature sources b for steady state
bss[[0, 12, 16]] = 10     # outdoor temperature To : 10 °C
bss[[6, 14]] = 20         # indoor temperature Ti1, Ti2 : 20 °C

fss = np.zeros(16)        # flow-rate sources f for steady state

A = TC['A']
G = TC['G']
diag_G = pd.DataFrame(np.diag(G), index=G.index, columns=G.index)

θss = np.linalg.inv(A.T @ diag_G @ A) @ (A.T @ diag_G @ bss + fss)
# print(f'θss = {np.around(θss, 2)} °C')

# State-space
[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)

bT = np.array([10, 20, 10, 20, 10])     # [To, Ti1, To, Ti2, To]
fQ = np.array([0, 0, 0, 0])             # [Φo1, Φi1, Φi2, Φo2]
uss = np.hstack([bT, fQ])               # input vector for state space
# print(f'uss = {uss}')


inv_As = pd.DataFrame(np.linalg.inv(As),
                      columns=As.index, index=As.index)
yss = (-Cs @ inv_As @ Bs + Ds) @ uss                        # output vector for state space are the temperatures θ5 and θ10
yss = [float(yss.values[0]), float(yss.values[1])]

print(f'yss_θ5 = {np.around(yss[0], 2)} °C')
print(f'yss_θ10 = {np.around(yss[1], 2)} °C')

# Compare: the error between the steady-state values obtained from the system of DAE and the output of the state-space representation yss
print(f'Errors between DAE and state-space: {abs(θss[5] - yss[0]):.2e} °C and {abs(θss[10] - yss[1]):.2e} °C')

# Writing TC to a CSV file

dm4bem.TC2file(TC, './MODEL/TC.csv')
