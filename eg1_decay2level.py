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

lam0_list = [1.0, 2.0, 3.0]

fig, ax = plt.subplots(1, len(lam0_list), figsize=(len(lam0_list)*3, 3))

num_init = 5 # choose multiple randomly generated initial density matrix

T = 1
Nlist= np.array([10, 20, 40, 100])

for lam_idx in range(len(lam0_list)):
    lam0 = lam0_list[lam_idx]
    print("lambda0 is {:.2f}".format(lam0))

    lbeq = model_decay_two_level(lam0, 0.5)
    lbeq.get_lindblad_superop()

    # variables to store errors
    err_1st_RK = np.zeros(len(Nlist))
    err_1st_SP = np.zeros(len(Nlist))
    err_2nd_RK = np.zeros(len(Nlist))
    err_2nd_MP = np.zeros(len(Nlist))
    err_3nd_RK = np.zeros(len(Nlist))
    err_3nd_SP = np.zeros(len(Nlist))
    err_4_RK = np.zeros(len(Nlist))
    err_4_SP = np.zeros(len(Nlist))

    # set random generator
    np.random.seed(1)

    # find terminal error at time T.
    for i in range(num_init):
        rho0 = rand_dm(2)

        for j in range(len(Nlist)):
            err_1st_RK[j] += get_terminal_error(RK1, lbeq, rho0, T, Nlist[j])
            err_1st_SP[j] += get_terminal_error(PTP_firstorder, lbeq, rho0, T, Nlist[j])
            err_2nd_RK[j] += get_terminal_error(RK2, lbeq, rho0, T, Nlist[j])
            err_2nd_MP[j] += get_terminal_error(PTP_secondorder_MP, lbeq, rho0, T, Nlist[j])
            err_3nd_RK[j] += get_terminal_error(RK3, lbeq, rho0, T, Nlist[j])
            err_3nd_SP[j] += get_terminal_error(PTP_thirdorder, lbeq, rho0, T, Nlist[j])
            err_4_RK[j] += get_terminal_error(RK4, lbeq, rho0, T, Nlist[j])
            err_4_SP[j] += get_terminal_error(PTP_fourthorder, lbeq, rho0, T, Nlist[j])

    err_1st_RK/=num_init
    err_1st_SP/=num_init
    err_2nd_RK/=num_init
    err_2nd_MP/=num_init
    err_3nd_RK/=num_init
    err_3nd_SP/=num_init
    err_4_RK/=num_init
    err_4_SP/=num_init

    ax[lam_idx].plot(Nlist, err_1st_RK, '--o', markersize=7, label='RK1', alpha=opacity, color=color_list[0])
    ax[lam_idx].plot(Nlist, err_1st_SP, '--o', markersize=7, label='SP1', alpha=opacity, color=color_list[1])
    ax[lam_idx].plot(Nlist, err_2nd_RK, '--s', markersize=7, label='RK2', alpha=opacity, color=color_list[2])
    ax[lam_idx].plot(Nlist, err_2nd_MP, '--s', markersize=7, label='SP2(MP)', alpha=opacity, color=color_list[3])
    ax[lam_idx].plot(Nlist, err_3nd_RK, '--P', markersize=7, label='RK3', alpha=opacity, color=color_list[4])
    ax[lam_idx].plot(Nlist, err_3nd_SP, '--P', markersize=7, label='SP3', alpha=opacity, color=color_list[5])
    ax[lam_idx].plot(Nlist, err_4_RK, '--X', markersize=7, label='RK4', alpha=opacity, color=color_list[6])
    ax[lam_idx].plot(Nlist, err_4_SP, '--X', markersize=7, label='SP4', alpha=opacity, color=color_list[7])

    ax[lam_idx].set_xscale('log')
    ax[lam_idx].set_yscale('log')

    mystr="{:1.0f}".format(lam0)
    ax[lam_idx].set_title(r"$\lambda_0="+mystr+"$", fontsize=14)
    ax[lam_idx].set_xlabel(r'$N$', fontsize=14)

    if lam_idx == 0:
        ax[lam_idx].set_ylabel(r'Avg. error', fontsize=14)

    # if lam_idx == 2:
    #     ax[lam_idx].legend()

plt.figlegend(['RK1', 'SP1', 'RK2', 'SP2(MP)', 'RK3', 'SP3', 'RK4', 'SP4'],
    bbox_to_anchor=(0.15, 1.1, 0.75, .10),
    fontsize=12,
    ncols=4, mode="expand", borderaxespad=0.)

plt.tight_layout()
filename="assets/decay2_convergence_"+str(num_init)+".pdf"
plt.savefig(filename, bbox_inches='tight')
