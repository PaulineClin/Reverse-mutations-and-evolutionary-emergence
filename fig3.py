'''
Sensitivity analysis: 
Probability of emergence as a function of parameters. 

This script run an ODE model with and without mutations to simulate the probability of emergence. 
it return a 3x3 plot with a subplot for each tested parameters. 
 
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.integrate import solve_ivp


# Plot font size

plt.rcParams.update({
    "font.size": 14,       # default text size
    "axes.titlesize": 16,  # axes title
    "axes.labelsize": 16,  # x and y labels
    "xtick.labelsize": 14, # x tick labels
    "ytick.labelsize": 14, # y tick labels
    "legend.fontsize": 14  # legend
})

# --- Reference/default parameters ---
nRef, bRef, dRef, cRef, muRef, fRRef, V0Ref = 5, 1.2, 1, 0, 0.05, 0.4, 100
T = 1000

# --- ODE system ---
def qSys(t, q, n, W, d, z, i):
    """
    Compute dq/dt for the ODE model.
    
    Parameters:
    - t: time (not used explicitly since system is autonomous)
    - q: vector of probabilities for each resistance type
    - n: number of resistance types
    - W: transmission-weighted mutation matrix
    - d: baseline death/removal rate
    - z: array of mutation shifts
    - i: indices of resistance types (0..n)
    
    Returns:
    - qprime: derivative vector dq/dt
    """
    Li = len(i)
    Lz = len(z)
    qprime = np.zeros(n+1)
    for ii in range(Li):
        I = i[ii]
        D = np.sum(W[ii, :]) + d # total outgoing rate
        E = d # baseline extinction contribution
        
        # sum over possible mutation shifts
        for zz in range(n-I, Lz-I):
            Z = z[zz]
            E += W[ii, zz] * q[ii] * q[ii+Z]
        pi = E/D
        qprime[ii] = D*(pi - q[ii])
    return qprime

# --- Probability of emergence ---
def prob_emergence(n, b, d, c, mu, nu, fR, V0):
    """
    Compute probability of pathogen emergence given parameters.
    
    Parameters:
    - n, b, d, c, mu, nu, fR, V0: model parameters
    Returns:
    - P_emergence: probability that pathogen successfully emerges
    """
    i = np.arange(n+1)
    bi = b*(1-c)**i # fitness penalty due to resistance cost
    z = np.arange(-n, n+1) # possible mutation shifts
    Li = len(i)
    Lz = len(z)
    
    # --- Mutation distribution matrix ---
    f = np.zeros((Li, Lz))
    for ii in range(Li):
        I = i[ii]
        for zz in range(Lz):
            Z = z[zz]
            x = np.arange(0, n-I+1)
            # probability of moving from I -> I+Z via forward/backward mutations
            f[ii, zz] = np.sum(binom.pmf(x, n-I, mu) * binom.pmf(x-Z, I, nu))
    
    # --- Host fitness ---
    Fi = fR*(i/n) + (1-fR)
    
    # --- Transmission-weighted mutation matrix ---
    W = np.zeros((Li, Lz))
    for ii in range(Li):
        I = i[ii]
        for zz in range(n-I, Lz-I):
            Z = z[zz]
            W[ii, zz] = bi[ii]*f[ii, zz]*Fi[ii+Z]
            
     # --- Solve ODE ---
    qInit = np.full(n+1, 0.1)
    sol = solve_ivp(lambda t, q: qSys(t, q, n, W, d, z, i),
                    [0, T], qInit, method='LSODA')
    
    q0 = sol.y[0, -1]  # probability for type I=0 at final time
    Q0 = (1-fR)*q0 + fR # weighted by resistant host fraction
    return 1 - Q0**V0 # emergence probability


# --- Sensitivity analysis ---
param_sets = {
    'fR': np.linspace(0, 1, 100),
    'mu': np.logspace(-3, -1, 100),
    'c': np.linspace(0, 0.06, 100),
    'b': np.linspace(1, 1.4, 100),
    'n': np.arange(1, 11),
    'V0': np.logspace(0, 3, 100)
}

ref_values = {
    'fR': fRRef,
    'mu': muRef,
    'c': cRef,
    'b': bRef,
    'n': nRef,
    'V0': V0Ref
}

# --- Create figure ---
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
axes = axes.flatten()

for idx, (param, values) in enumerate(param_sets.items()):
    P_fwd = []
    P_rev = []

    for val in values:
        # Reset parameters
        n, b, d, c, mu, fR, V0 = nRef, bRef, dRef, cRef, muRef, fRRef, V0Ref
        locals()[param] = val

        P_fwd.append(prob_emergence(n, b, d, c, mu, 0, fR, V0))
        P_rev.append(prob_emergence(n, b, d, c, mu, mu, fR, V0))

    ax = axes[idx]
    if param == 'n':
        ax.plot(values, P_fwd, 'ko', label='Forward only')
        ax.plot(values, P_rev, 'bo', label='Forward + reverse')
    else:
        if param in ['mu', 'V0']:
            ax.semilogx(values, P_fwd, 'k-', label='Forward only')
            ax.semilogx(values, P_rev, 'b-', label='Forward + reverse')
        else:
            ax.plot(values, P_fwd, 'k-', label='Forward only')
            ax.plot(values, P_rev, 'b-', label='Forward + reverse')

    # Plot reference values
    ref_val = ref_values[param]
    idx_ref = (np.abs(values - ref_val)).argmin()
    ax.plot(ref_val, P_fwd[idx_ref], 'k*', markersize=10)
    ax.plot(ref_val, P_rev[idx_ref], 'b*', markersize=10)
    
    # Labels
    labels = {
        'fR': 'Fraction of resistant host ($f_R$)',
        'mu': 'Mutation rate ($\\mu$)',
        'c': 'Escape mutation cost ($c$)',
        'b': 'Transmission rate ($b$)',
        'n': 'Number of resistance types ($n$)',
        'V0': 'Inoculum size ($V_0$)'
    }

    ax.set_xlabel(param)
    ax.set_ylabel('Probability of emergence')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    # Special case: subplot (2,2) → index 3
    if idx == 3:
        ax.set_xlim(left=1)
        
    ax.grid(True)
    ax.legend()
    


plt.tight_layout()
plt.show()
