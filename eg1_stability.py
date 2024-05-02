import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from lindbladsolver import *

# define the model
def model_decay_two_level(lambda0, nu):
    H = Qobj(np.zeros((2,2)))
    L1 = np.sqrt(lambda0*(nu+1))*sigmam()
    L2 = np.sqrt(lambda0*nu)*sigmap()
    return LindbladEq(2,H,[L1,L2])

lam0 = 5
lbeq = model_decay_two_level(lam0, 0.5)
lbeq.get_lindblad_superop()

dt = 0.42;
N = 7 # number of steps
T = dt*N # = 2.94 approximately 3
plot_step = 1

logscale = True

# initial conditions
rx = 1/np.sqrt(6)
ry = 1/np.sqrt(3)
rz = 1/np.sqrt(2)
rho0 = 0.5*(qeye(2) + rx*sigmax() + ry*sigmay() + rz*sigmaz())

listofobservable = [sigmax(), sigmay(), sigmaz()]
Ns = 100

_,_,outcome_firstorder = detsolver(PTP_firstorder,
    lbeq, rho0.copy(), dt, N, False, listofobservable)
_,_,outcome_secondorder_MP = detsolver(PTP_secondorder_MP,
    lbeq, rho0.copy(), dt, N, False, listofobservable)
_,_,outcome_secondorder_RK = detsolver(RK2,
    lbeq, rho0.copy(), dt, N, False, listofobservable)
_,_,outcome_exact = detsolver(RK2,
    lbeq, rho0.copy(), dt/Ns, N*Ns, False, listofobservable)


timelist = np.arange(0, N+1)*dt
timelist_exact = np.arange(0, N*Ns+1)*(dt/Ns)

fig = plt.figure(figsize=(6,2.5))
plt.subplot(1,2,1)
# first part Sigma_X
plt.plot(timelist_exact[0:(Ns*N+1):(Ns*plot_step)],
    np.abs(outcome_exact[0,0:(Ns*N+1):(Ns*plot_step)].real), '--o', markersize=10, color=color_list[0])
plt.plot(timelist[0:(N+1):plot_step],
    np.abs(outcome_firstorder[0,0:(N+1):plot_step].real),'--o', markersize=7, color=color_list[1])
plt.plot(timelist[0:(N+1):plot_step],
    np.abs(outcome_secondorder_RK[0,0:(N+1):plot_step].real),'--s', markersize=7, color=color_list[2])
plt.plot(timelist[0:(N+1):plot_step],
    np.abs(outcome_secondorder_MP[0,0:(N+1):plot_step].real),'--s', markersize=7, color=color_list[3])
if logscale:
    plt.yscale("log")
plt.xlabel('time', fontsize=14)
plt.ylabel(r"$|\langle\sigma_X\rangle_{\rho_t}|$", fontsize=14)


plt.subplot(1,2,2)
# second part Sigma_Y
plt.plot(timelist_exact[0:(Ns*N+1):(Ns*plot_step)],
    np.abs(outcome_exact[1,0:(Ns*N+1):(Ns*plot_step)].real), '--o', markersize=10, color=color_list[0], label="exact")
plt.plot(timelist[0:(N+1):plot_step],
    np.abs(outcome_firstorder[1,0:(N+1):plot_step].real),'--o', markersize=7, color=color_list[1], label="SP1")
plt.plot(timelist[0:(N+1):plot_step],
    np.abs(outcome_secondorder_RK[1,0:(N+1):plot_step].real),'--s', markersize=7, color=color_list[2], label="RK2")
plt.plot(timelist[0:(N+1):plot_step],
    np.abs(outcome_secondorder_MP[1,0:(N+1):plot_step].real),'--s', markersize=7, color=color_list[3], label="SP2(MP)")
if logscale:
    plt.yscale("log")
plt.xlabel("time", fontsize=14)
plt.ylabel(r"$|\langle\sigma_Y\rangle_{\rho_t}|$", fontsize=14)

plt.figlegend(['exact', 'SP1', 'RK2', 'SP2(MP)'],
    bbox_to_anchor=(0.18, 1.1, 0.75, .10),
    fontsize=12,
    ncols=2, mode="expand", borderaxespad=0.)
plt.tight_layout()
filename = "assets/decay2_stability.pdf"
plt.savefig(filename, bbox_inches='tight')
