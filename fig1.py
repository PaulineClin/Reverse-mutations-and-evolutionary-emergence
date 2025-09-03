'''
Probability of emergence as a function of the fraction of resistant hosts in the mixture. 

This script run a CTMC model and an ODE model with and without mutations to simulate the probability of emergence. 
The evolutionary of emergence is represented in grey on the plot. 
 
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

# reproducible CTMC runs (optional): Fixes the random seed so Monte Carlo CTMC results are reproducible.
np.random.seed(0)

# -------------------------
# Model parameters
# -------------------------
n = 1 # number of resistant types
b = 1.5 # transmission rate
d = 1 # death rate
c = 0.05 # cost of escape mutation
V0 = 10**1 # inoculu size

T = 1000  # ODE time horizon

# CTMC parameters 
max_events = 10**3 # maximum events per CTMC trajectory to prevent infinite loops.
sim_number = 10**3 # number of Monte Carlo CTMC simulations.

# fraction of resistant grid
LfR = 50 # 50 points for fR ranging from 0 to 1.
fR = np.linspace(0.0, 1.0, LfR) # Each fR is a fraction of resistant hosts.

# Precomputed arrays: fixed arrays that depend on n,b,c
i = np.arange(0, n + 1)          # number of resistant types present (0..n) --> [0, 1, ..., n]
bi_template = b * (1 - c) ** i   # transmission rate modified by mutation cost for each i.
z = np.arange(-n, n + 1)         # index offsets for mutation transitions. --> [-n, ..., 0, ..., n]
Lz = len(z)                      # lengths of z
Li = len(i)                      # lengths of i

# -------------------------
# ODE system to calculate the probability of emergence
# -------------------------
def qSys(t, q, n, W, d, z, i, Li, Lz):
    qprime = np.zeros(n + 1)
    for ii in range(Li):
        I = i[ii]
        D = np.sum(W[ii, :]) + d
        E = d
        # zz range in python: range(n-I, Lz-I)  (end exclusive)
        for zz in range((n - I), (Lz - I)):
            Z = z[zz]
            E += W[ii, zz] * q[ii] * q[ii + Z] # W[ii, zz] represents contributions of mutations.
        pi = E / D
        qprime[ii] = D * (pi - q[ii]) # q[ii] is the probability that there are exactly I resistant hosts.
    return qprime

# -------------------------
# Run both ODE and CTMC models for a given (mu = foreward mutation rate, nu = reverse mutation rate)
# returns: P_det (ODE), P_ctmc (CTMC) as arrays length LfR
# -------------------------
def run_models(mu, nu):
    bi = bi_template.copy()   # depends on b,c,n 
    # build f (mutation distribution) for this (mu,nu)
    f = np.zeros((Li, Lz))
    for ii in range(Li):
        I = i[ii]
        for zz in range(Lz):
            Z = z[zz]
            # x ranges 0..(n-I)
            x = np.arange(0, n - I + 1)
            # binom.pmf with mu, nu is the probability mass function of a binomial.
            # it gives the probability of transitioning from I to I + Z due to mutations.
            f[ii, zz] = np.sum(binom.pmf(x, n - I, mu) * binom.pmf(x - Z, I, nu))

    # Store results
    P_det = np.zeros(LfR)   # deterministic (ODE) result
    P_ctmc = np.zeros(LfR)  # CTMC estimate

    # Maps the 6 possible CTMC events to which state increments/decrements.
    # CTMC transition index arrays
    ChangePlus = [1, 2, 0, 2, 1, 0]
    ChangeMinus = [0, 0, 1, 0, 0, 2]

    # Loop over fraction resistant
    for ffR in range(LfR):
        FR = fR[ffR]
        Fi = FR * (i / n) + (1 - FR)    # Fi is the fitness of each type given the fraction of resistant hosts.

        # build W matrix for this FR
        W = np.zeros((Li, Lz))
        for ii in range(Li):
            I = i[ii]
            for zz in range((n - I), (Lz - I)):
                Z = z[zz]
                W[ii, zz] = bi[ii] * f[ii, zz] * Fi[ii + Z]

        # -------------------------
        # Deterministic ODE using a LSODA solver
        # -------------------------
        qInit = np.full(n + 1, 0.1)
        sol = solve_ivp(lambda t, q: qSys(t, q, n, W, d, z, i, Li, Lz),
                        [0, T], qInit, method='LSODA')
        # last state:
        q_last = sol.y[:, -1]
        q0 = q_last[0]   # corresponds to I=0
        Q0 = (1 - FR) * q0 + FR
        # Compute the final probability of emergence.
        P_det[ffR] = 1.0 - Q0**V0

        # -------------------------
        # CTMC Monte Carlo to simulate stochastic CTMC trajectories.
        # -------------------------
        # transition probs for n=1 (W shape is Li x Lz); these indices assume n=1:
        # W[0,1] = prob 0->0 transmission without mutation
        # W[0,2] = prob 0->1 transmission with forward mutation
        # W[1,1] = prob 1->0 transmission with reverse mutation (if nu>0) 
        # Extract them now:
        P00 = W[0, 1] if (W.shape[1] > 1) else 0.0
        P01 = W[0, 2] if (W.shape[1] > 2) else 0.0
        P0d = d
        P10 = W[1, 1] if (W.shape[0] > 1 and W.shape[1] > 1) else 0.0
        P1M1 = W[1, 0] if (W.shape[0] > 1 and W.shape[1] > 0) else 0.0
        P1d = d

        Extinction = 0
        for _ in range(sim_number):
            X = [1, 0]  # I0=1, I1=0 to keep track of numbers in each state.
            j = 0
            # simulate CTMC until extinction or max_events
            while (X[0] + X[1]) > 0 and j < max_events:
                Prob = [P00 * X[0], P01 * X[0], P0d * X[0],
                        P10 * X[1], P1M1 * X[1], P1d * X[1]]
                Sumprob = sum(Prob)
                if Sumprob <= 0:
                    break
                # two uniforms
                u1, u2 = np.random.rand(2)
                # time increment (not used further here but mimic original)
                # t_next = -np.log(u1) / Sumprob
                thresh = u2 * Sumprob
                Countersum = 0.0
                for k, pk in enumerate(Prob):
                    Countersum += pk
                    if thresh < Countersum:
                        # apply event k
                        if ChangePlus[k] > 0:
                            idx = ChangePlus[k] - 1  # convert to 0-based
                            X[idx] += 1
                        if ChangeMinus[k] > 0:
                            idx = ChangeMinus[k] - 1
                            X[idx] -= 1
                        break
                j += 1
            if (X[0] + X[1]) == 0:
                Extinction += 1

        # Compute probability of emergence from CTMC.
        q0_ctmc = Extinction / sim_number
        Q0_ctmc = (1 - FR) * q0_ctmc + FR
        P_ctmc[ffR] = 1.0 - Q0_ctmc**V0

    return P_det, P_ctmc

# -------------------------
# Run twice: (mu=nu=0) and (mu=nu>0)
# -------------------------
mu0 = 0.0
nu0 = 0.0
mu1 = 0.01
nu1 = 0.01

print("Running for mu=nu=0  (this may take a little time)...")
P_det_0, P_ctmc_0 = run_models(mu0, nu0)

print("Running for mu=nu=0.01  (this may take a little time)...")
P_det_1, P_ctmc_1 = run_models(mu1, nu1)

# -------------------------
# Plot results: both runs on same figure
# -------------------------
plt.figure(figsize=(8, 6))

# Shade between ODE curves
plt.fill_between(fR, P_det_0, P_det_1, color="gray", alpha=0.2) #, label="Difference (ODE)"

# Plot ODE curves
plt.plot(fR, P_det_0, 'k--', linewidth=1.5, label='ODE (μ=ν=0)')
plt.plot(fR, P_det_1, 'k-', linewidth=1.5, label=f'ODE (μ=ν={mu1})')

# Plot CTMC curves
plt.plot(fR, P_ctmc_0, 'b--', linewidth=1.5, label='CTMC (μ=ν=0)')
plt.plot(fR, P_ctmc_1, 'b-', linewidth=1.5, label=f'CTMC (μ=ν={mu1})')

# Labels
plt.xlabel('Fraction of resistant (fR)')
plt.ylabel('Probability of emergence')

# Force axes to start from 0
plt.xlim(left=0)
plt.ylim(bottom=0)

# Remove grid
plt.grid(False)

plt.legend()
plt.tight_layout()
plt.show()
