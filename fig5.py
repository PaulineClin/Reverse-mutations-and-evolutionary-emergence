'''
Contour plot: 
Probability of emergence as a function of parameters. 

This script run an ODE model with and without mutations to simulate the probability of emergence. 
it return a 3x3 plot with a subplot for each tested parameters. 
 
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.integrate import solve_ivp
from matplotlib.colors import ListedColormap, BoundaryNorm

# --- Fixed parameters ---
d = 1 # baseline death rate
T = 1000 # ODE time horizon
V0 = 100  # initial inoculum size
fR = 0.9 # fraction of resistant hosts

# --- Semi-fixed parameters ---
n = 5 # number of resistance types
b = 5 # transmission rate

# --- Reference probability ---
# to calculate the reference probability of emergence in the absence of mutation. 
P0 = 1 - (fR + d / b) ** V0

# --- Parameters to vary ---
# Defines the parameter grid for sensitivity analysis
LC = 100
C = np.linspace(0, 0.4, LC)[::-1]  # # escape mutation cost
LMu = 100
Mu = np.logspace(-2, -1, LMu) # mutation rates

# Pre-allocates an array to store the difference in emergence probability between forward-only and forward+reverse mutations.
Delta = np.zeros((LC, LMu))

# --- Function for ODE system ---
def qSys(t, q, n, W, d, z, i, Li, Lz):
    qprime = np.zeros(n + 1)
    for ii in range(Li):
        I = i[ii]
        D = np.sum(W[ii, :]) + d
        E = d
        for zz in range(n - I, Lz - I):
            Z = z[zz]
            q_index = ii + Z
            if 0 <= q_index < len(q):
                E += W[ii, zz] * q[ii] * q[q_index]
        pi = E / D
        qprime[ii] = D * (pi - q[ii])
    return qprime

# --- Compute Delta ---
for k in range(LC):
    c = C[k]
    for j in range(LMu):
        mu = Mu[j]
        nu = 0  # Forward only

        i = np.arange(n + 1)
        bi = b * (1 - c) ** i

        z = np.arange(-n, n + 1)
        Li = len(i)
        Lz = len(z)
        f = np.zeros((Li, Lz))
        for ii in range(Li):
            I = i[ii]
            for zz in range(Lz):
                Z = z[zz]
                x = np.arange(0, n - I + 1)
                f[ii, zz] = np.sum(binom.pmf(x, n - I, mu) * binom.pmf(x - Z, I, nu))

        Fi = fR * (i / n) + (1 - fR)
        W = np.zeros((Li, Lz))
        for ii in range(Li):
            I = i[ii]
            for zz in range(n - I, Lz - I):
                Z = z[zz]
                W[ii, zz] = bi[ii] * f[ii, zz] * Fi[ii + Z]

        qInit = np.full(n + 1, 0.1)
        sol = solve_ivp(lambda t, q: qSys(t, q, n, W, d, z, i, Li, Lz), [0, T], qInit, method='LSODA')
        q0 = sol.y[0, -1]
        Q0 = (1 - fR) * q0 + fR

        Delta[k, j] = -(1 - Q0 ** V0)  # Forward only

    for j in range(LMu):
        mu = Mu[j]
        nu = mu  # Forward + reverse

        i = np.arange(n + 1)
        bi = b * (1 - c) ** i

        z = np.arange(-n, n + 1)
        Li = len(i)
        Lz = len(z)
        f = np.zeros((Li, Lz))
        for ii in range(Li):
            I = i[ii]
            for zz in range(Lz):
                Z = z[zz]
                x = np.arange(0, n - I + 1)
                f[ii, zz] = np.sum(binom.pmf(x, n - I, mu) * binom.pmf(x - Z, I, nu))

        Fi = fR * (i / n) + (1 - fR)
        W = np.zeros((Li, Lz))
        for ii in range(Li):
            I = i[ii]
            for zz in range(n - I, Lz - I):
                Z = z[zz]
                W[ii, zz] = bi[ii] * f[ii, zz] * Fi[ii + Z]

        qInit = np.full(n + 1, 0.1)
        sol = solve_ivp(lambda t, q: qSys(t, q, n, W, d, z, i, Li, Lz), [0, T], qInit, method='LSODA')
        q0 = sol.y[0, -1]
        Q0 = (1 - fR) * q0 + fR

        Delta[k, j] += 1 - Q0 ** V0  # Forward + reverse

# --- Plot with three colors ---
vmax = 2e-2
vmin = -vmax

# Define three bins
levels = [-np.inf, vmin, vmax, np.inf]
cmap = ListedColormap(['blue', 'lightgrey', 'red'])
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)

plt.figure(figsize=(8, 6))
cf = plt.contourf(Mu, C, Delta, levels=levels, cmap=cmap, norm=norm)
plt.xscale('log')

# Keep only 10^-2 and 10^-1 on x-axis
plt.xticks([1e-2, 1e-1], [r'$10^{-2}$', r'$10^{-1}$'], fontsize=14)

plt.xlabel('Mutation rate ($\mu$)', fontsize=14)
plt.ylabel('Escape mutation cost ($c$)', fontsize=14)
plt.grid(False)
plt.show()