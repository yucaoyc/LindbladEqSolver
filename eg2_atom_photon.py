import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from lindbladsolver import *

def model_atom_photon(omega=1.0, Omega=1.0, g=1.0,
        alpha=1.0, beta=1.0, gamma=1.0, nu=0.5, eta=0.5,
        num_photon=10):
    """
        create a Lindblad equation for atom-photon coupling systems.
    """
    a = destroy(num_photon)
    adag = a.dag()
    numopa = adag*a

    H = tensor(qeye(2), omega*numopa)
    H += tensor(Omega*sigmaz(), qeye(num_photon))
    H += (-g)*(tensor(sigmam(), adag) + tensor(sigmap(), a))

    L1 = tensor(qeye(2), np.sqrt(alpha*(nu+1)) * a)
    L2 = tensor(qeye(2), np.sqrt(alpha*nu) * adag)
    L3 = tensor(np.sqrt(beta*(1-eta)) * sigmam(), qeye(num_photon))
    L4 = tensor(np.sqrt(beta*eta) * sigmap(), qeye(num_photon))
    L5 = tensor(np.sqrt(gamma)*sigmaz(), qeye(num_photon))

    return LindbladEq([2,num_photon], H, [L1, L2, L3, L4, L5])


num_init = 5 # choose multiple randomly generated initial density matrix
T = 1.0
Nlist= int(T)*np.array([20,50,100,200])

for num_photon in [2, 5, 10]:

    fig, ax = plt.subplots(1,3,figsize=(9, 3))
    # parameters to choose
    for idx in range(3):
        if idx == 0:
            alpha = beta = gamma = 1.0e-2;
        elif idx == 1:
            alpha = beta = gamma = 0.1;
        else:
            alpha = beta = gamma = 1.0;

        # set up Lindblad eq.
        lbeq = model_atom_photon(alpha=alpha, beta=beta, gamma=gamma, num_photon=num_photon)
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
            rho0 = rand_dm(np.product(lbeq.dim), dims=[lbeq.dim,lbeq.dim])
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

        ax[idx].plot(Nlist, err_1st_RK, '--o', markersize=7, label='RK1', alpha=opacity, color=color_list[0])
        ax[idx].plot(Nlist, err_1st_SP, '--o', markersize=7, label='SP1', alpha=opacity, color=color_list[1])
        ax[idx].plot(Nlist, err_2nd_RK, '--s', markersize=7, label='RK2', alpha=opacity, color=color_list[2])
        ax[idx].plot(Nlist, err_2nd_MP, '--s', markersize=7, label='SP2(MP)', alpha=opacity, color=color_list[3])
        ax[idx].plot(Nlist, err_3nd_RK, '--P', markersize=7, label='RK3', alpha=opacity, color=color_list[4])
        ax[idx].plot(Nlist, err_3nd_SP, '--P', markersize=7, label='SP3', alpha=opacity, color=color_list[5])
        ax[idx].plot(Nlist, err_4_RK, '--X', markersize=7, label='RK4', alpha=opacity, color=color_list[6])
        ax[idx].plot(Nlist, err_4_SP, '--X', markersize=7, label='SP4', alpha=opacity, color=color_list[7])

        ax[idx].set_xscale('log')
        ax[idx].set_yscale('log')
        ax[idx].set_title(r"$\alpha=\beta=\gamma="+str(alpha)+"$", fontsize=14)
        ax[idx].set_xlabel(r'$N$', fontsize=14)

        if idx == 0:
            ax[idx].set_ylabel(r'Avg. error', fontsize=14)

    plt.figlegend(['RK1', 'SP1', 'RK2', 'SP2(MP)', 'RK3', 'SP3', 'RK4', 'SP4'],
        bbox_to_anchor=(0.15, 1.1, 0.75, .10),
        fontsize=12,
        ncols=4, mode="expand", borderaxespad=0.)
    plt.tight_layout()
    filename="assets/convergence_photon_"+str(num_photon)+"_"+str(num_init)+".pdf"
    plt.savefig(filename, bbox_inches='tight')
