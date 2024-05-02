import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from lindbladsolver import *
import time

def model_Ising(Delta=1.0, J=1.0, gamma=0.1, n_atom=2):
    """
        create a Lindblad equation for dissipative Ising model.
    """
    sz = sigmaz()
    sx = sigmax()

    H = 0*tensor([qeye(2) for j in range(n_atom)])
    for j in range(1, n_atom+1):
        left = [qeye(2) for k in range(j-1)]
        right = [qeye(2) for k in range(n_atom-j)]
        H += Delta * tensor(left + [sz] + right)
    for j in range(1, n_atom):
        left = [qeye(2) for k in range(j-1)]
        right = [qeye(2) for k in range(n_atom-j-1)]
        H += (-J) * tensor(left + [sx, sx] + right)

    L_list = []
    for j in range(1, n_atom):
        left = [qeye(2) for k in range(j-1)]
        right = [qeye(2) for k in range(n_atom-j)]
        L_list.append(np.sqrt(gamma) * tensor(left + [sigmam()] + right))

    return LindbladEq([2 for j in range(n_atom)], H, L_list)


num_init = 5 # choose multiple randomly generated initial density matrix

# parameters to choose
T = 1.0
Nlist= int(T)*np.array([20,50,100,200])

for n_atom in [2, 4, 6]:

    t_start = time.time()
    print("num atom is {:d}".format(n_atom))

    fig, ax = plt.subplots(1,3,figsize=(9, 3))
    for idx in range(3):
        # different models
        if idx == 0:
            gamma = 1.0e-2;
        elif idx == 1:
            gamma = 0.1;
        else:
            gamma = 1.0;

        # set up Lindblad eq.
        lbeq = model_Ising(gamma=gamma, n_atom=n_atom)
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
            print(i)
            rho0 = rand_dm(np.product(lbeq.dim), dims=[lbeq.dim,lbeq.dim])
            exactrho = ((lbeq.lb_superop*T).expm())(rho0)

            for j in range(len(Nlist)):
                err_1st_RK[j] += get_terminal_error(RK1, lbeq, rho0, T, Nlist[j], exactrho=exactrho)
                err_1st_SP[j] += get_terminal_error(PTP_firstorder, lbeq, rho0, T, Nlist[j], exactrho=exactrho)
                err_2nd_RK[j] += get_terminal_error(RK2, lbeq, rho0, T, Nlist[j], exactrho=exactrho)
                err_2nd_MP[j] += get_terminal_error(PTP_secondorder_MP, lbeq, rho0, T, Nlist[j], exactrho=exactrho)
                err_3nd_RK[j] += get_terminal_error(RK3, lbeq, rho0, T, Nlist[j], exactrho=exactrho)
                err_3nd_SP[j] += get_terminal_error(PTP_thirdorder, lbeq, rho0, T, Nlist[j], exactrho=exactrho)
                err_4_RK[j] += get_terminal_error(RK4, lbeq, rho0, T, Nlist[j], exactrho=exactrho)
                err_4_SP[j] += get_terminal_error(PTP_fourthorder, lbeq, rho0, T, Nlist[j], exactrho=exactrho)

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
        ax[idx].set_title(r"$\gamma="+str(gamma)+"$", fontsize=14)
        ax[idx].set_xlabel(r'$N$', fontsize=14)

        if idx == 0:
            ax[idx].set_ylabel(r'Avg. error', fontsize=14)

    plt.figlegend(['RK1', 'SP1', 'RK2', 'SP2(MP)', 'RK3', 'SP3', 'RK4', 'SP4'],
        bbox_to_anchor=(0.15, 1.1, 0.75, .10),
        fontsize=12,
        ncols=4, mode="expand", borderaxespad=0.)

    plt.tight_layout()
    filename="assets/convergence_ising_"+str(n_atom)+"_"+str(num_init)+".pdf"
    plt.savefig(filename, bbox_inches='tight')
    elapsed = time.time() - t_start
    print("Elapsed time: {:.0f} seconds".format(elapsed))
