from qutip import *
import numpy as np

import os
if not os.path.exists('assets'):
    os.makedirs('assets')

color_list = ['#D8A6A6', '#C1272D',
              '#AECEE8', '#1A80BB',
              '#9FC8C8', '#1F6F6F',
              '#FFBB6F', '#EA801C']
opacity=1.0

class LindbladEq:
    def __init__(self, dim, Hami, Loplist):
        """
            initialize the system.
        """
        self.dim = dim
        self.numop = len(Loplist)
        self.Hami = Hami
        self.Loplist = Loplist
        # effective Hamiltonian
        self.effHami = Hami.copy()
        for k in range(self.numop):
            L = Loplist[k]
            self.effHami += 1/complex(0,2)*(L.dag())*L
        # for general cases, we don't compute the lindblad super operator
        self.lb_superop = None

    def get_lindblad_superop(self):
        """
            get the exact Lindblad equation superoperator.
        """
        self.lb_superop = spre(complex(0,-1)*self.Hami)
        self.lb_superop += spost(complex(0,1)*self.Hami)
        for k in range(self.numop):
            L = self.Loplist[k]
            self.lb_superop += to_super(L)
            self.lb_superop += -0.5*spre(L.dag()*L)
            self.lb_superop += -0.5*spost(L.dag()*L)

    def apply_lindblad_superop(self, rho):
        """
            compute \mathcal{L}(rho)
        """
        mat = complex(0,-1)*self.Hami*rho
        mat += complex(0,1)*rho*self.Hami
        for k in range(self.numop):
            L = self.Loplist[k]
            mat += L*rho*(L.dag())
            LdagL = (L.dag())*L
            mat -= 0.5*LdagL*rho
            mat -= 0.5*rho*LdagL
        return mat

    def apply_lindblad_Lsuperop(self, rho):
        """
            compute \mathcal{L}_L(rho)
        """
        mat = 0.0*rho # create a zero object
        for k in range(self.numop):
            L = self.Loplist[k]
            mat += L*rho*(L.dag())
        return mat

def get_op_J(lbeq, dt, m):
    """
        compute the taylor expansion of (-i Heff * dt) up to level m.
    """
    op = qeye(lbeq.dim)
    tmp = qeye(lbeq.dim)
    for j in range(1, m+1):
        tmp *= complex(0,-1)*lbeq.effHami*dt
        op += tmp/np.math.factorial(j)
    return op

# a list of Lindblad solvers.

def RK1(lbeq, rho, dt):
    """
        The first-order Runge-Kutta.
    """
    Lrho = lbeq.apply_lindblad_superop(rho)
    return rho + dt*Lrho

def RK2(lbeq, rho, dt):
    """
        The second-order Runge-Kutta.
    """
    Lrho = lbeq.apply_lindblad_superop(rho)
    LLrho = lbeq.apply_lindblad_superop(Lrho)
    return rho + dt*Lrho + dt**2/2*LLrho

def RK3(lbeq, rho, dt):
    """
        The third-order Runge-Kutta.
    """
    Lrho = lbeq.apply_lindblad_superop(rho)
    LLrho = lbeq.apply_lindblad_superop(Lrho)
    LLLrho = lbeq.apply_lindblad_superop(LLrho)
    return rho + dt*Lrho + dt**2/2*LLrho + dt**3/6*LLLrho

def RK4(lbeq, rho, dt):
    """
        The fourth-order Runge-Kutta.
    """
    Lrho = lbeq.apply_lindblad_superop(rho)
    LLrho = lbeq.apply_lindblad_superop(Lrho)
    LLLrho = lbeq.apply_lindblad_superop(LLrho)
    LLLLrho = lbeq.apply_lindblad_superop(LLLrho)
    return rho + dt*Lrho + dt**2/2*LLrho + dt**3/6*LLLrho + dt**4/24 * LLLLrho

def PTP_firstorder(lbeq, rho, dt):
    """
        The first-order structure-preserving scheme
    """
    Kmat = get_op_J(lbeq, dt, 1)
    nextrho = Kmat*rho*(Kmat.dag())
    nextrho += dt*lbeq.apply_lindblad_Lsuperop(rho)
    c = nextrho.tr()
    return nextrho/c

def PTP_secondorder_TR(lbeq, rho, dt):
    """
        The second-order structure-preserving scheme with Trapezoidal rule
    """
    Kmat_1 = get_op_J(lbeq, dt, 2)
    nextrho = Kmat_1*rho*(Kmat_1.dag())

    Kmat_2 = get_op_J(lbeq, dt, 1)
    nextrho += dt/2*Kmat_2*(lbeq.apply_lindblad_Lsuperop(rho))*(Kmat_2.dag())
    nextrho += dt/2*lbeq.apply_lindblad_Lsuperop(Kmat_2*rho*(Kmat_2.dag()))

    nextrho += (dt**2)/2*lbeq.apply_lindblad_Lsuperop(lbeq.apply_lindblad_Lsuperop(rho))

    c = nextrho.tr()
    return nextrho/c

def PTP_secondorder_MP(lbeq, rho, dt):
    """
        The second-order structure-preserving scheme with midpoint rule
    """
    Kmat_1 = get_op_J(lbeq, dt, 2)
    nextrho = Kmat_1*rho*(Kmat_1.dag())

    Kmat_2 = get_op_J(lbeq, dt/2, 1)
    temprho = Kmat_2*rho*(Kmat_2.dag())
    temprho = lbeq.apply_lindblad_Lsuperop(temprho)
    nextrho += dt*Kmat_2*temprho*(Kmat_2.dag())

    nextrho += (dt**2)/2*lbeq.apply_lindblad_Lsuperop(lbeq.apply_lindblad_Lsuperop(rho))
    c = nextrho.tr()
    return nextrho/c

def PTP_thirdorder(lbeq, rho, dt):
    """
        The third-order structure-preserving scheme.
    """
    Lrho = lbeq.apply_lindblad_Lsuperop(rho)

    # Level 0
    Kmat_0 = get_op_J(lbeq, dt, 3)
    nextrho = Kmat_0*rho*(Kmat_0.dag())

    # Leve 1
    Kmat_1 = get_op_J(lbeq, dt/3, 2)
    Kmat_2 = get_op_J(lbeq, dt/3*2, 2)
    Kmat_3 = get_op_J(lbeq, dt, 2)

    nextrho += 3*dt/4 * Kmat_1*lbeq.apply_lindblad_Lsuperop(Kmat_2*rho*(Kmat_2.dag()))*Kmat_1.dag()
    nextrho += dt/4 * Kmat_3*Lrho*Kmat_3.dag()

    # Level 2
    Kmat_4 = get_op_J(lbeq, dt/3, 1)
    tmp = lbeq.apply_lindblad_Lsuperop(Kmat_4*rho*Kmat_4.dag())
    tmp = lbeq.apply_lindblad_Lsuperop(Kmat_4*tmp*Kmat_4.dag())
    nextrho += dt**2/2*Kmat_4*tmp*Kmat_4.dag()

    # Level 3
    nextrho += (dt**3)/6*lbeq.apply_lindblad_Lsuperop(lbeq.apply_lindblad_Lsuperop(Lrho))

    # correct trace
    c = nextrho.tr()
    return nextrho/c

def PTP_fourthorder(lbeq, rho, dt):
    """
        The fourth-order structure-preserving scheme.
    """
    Lrho = lbeq.apply_lindblad_Lsuperop(rho)

    # Level 0
    Kmat_0 = get_op_J(lbeq, dt, 4)
    nextrho = Kmat_0*rho*(Kmat_0.dag())

    # Leve 1
    Kmat_1 = get_op_J(lbeq, (3 - np.sqrt(3))/6 * dt, 3)
    Kmat_2 = get_op_J(lbeq, (3 + np.sqrt(3))/6 * dt, 3)
    nextrho += dt/2 * Kmat_2*(lbeq.apply_lindblad_Lsuperop(Kmat_1*rho*Kmat_1.dag()))*Kmat_2.dag()
    nextrho += dt/2 * Kmat_1*(lbeq.apply_lindblad_Lsuperop(Kmat_2*rho*Kmat_2.dag()))*Kmat_1.dag()

    # Level 2
    Kmat_3 = get_op_J(lbeq, dt/4, 2)
    Kmat_4 = get_op_J(lbeq, dt/2, 2)
    Kmat_5 = get_op_J(lbeq, 3*dt/4, 2)
    Kmat_6 = get_op_J(lbeq, dt, 2)
    nextrho += dt**2/9 * Kmat_5*lbeq.apply_lindblad_Lsuperop(Kmat_3*Lrho*Kmat_3.dag())*Kmat_5.dag()
    nextrho += dt**2/3 * Kmat_3*lbeq.apply_lindblad_Lsuperop(Kmat_3*lbeq.apply_lindblad_Lsuperop(Kmat_4*rho*Kmat_4.dag())*Kmat_3.dag())*Kmat_3.dag()
    nextrho += dt**2/18 * lbeq.apply_lindblad_Lsuperop(Kmat_6*Lrho*Kmat_6.dag())

    # Level 3
    Kmat_7 = get_op_J(lbeq, dt/4, 1)
    tmp = lbeq.apply_lindblad_Lsuperop(Kmat_7*rho*Kmat_7.dag())
    tmp = lbeq.apply_lindblad_Lsuperop(Kmat_7*tmp*Kmat_7.dag())
    tmp = lbeq.apply_lindblad_Lsuperop(Kmat_7*tmp*Kmat_7.dag())
    nextrho += dt**3/6 * Kmat_7 * tmp * Kmat_7.dag()

    # Level 4
    nextrho += (dt**4)/24*lbeq.apply_lindblad_Lsuperop(lbeq.apply_lindblad_Lsuperop(lbeq.apply_lindblad_Lsuperop(Lrho)))

    # correct trace
    c = nextrho.tr()
    return nextrho/c


def detsolver(solvername, lbeq, rho0, dt, N, to_storage=False, listofobservable=[]):
    """
        solvername is the name of previous a few functions
        lbeq, rho0, dt, N are parameters
        to_storage determines whether to keep track of all rho_t
        listofobservable contains a list of observables (or Hermitian operators).
    """
    # whether to store the whole trajectory
    if to_storage:
        storage = [None for j in range(N+1)]
        storage[0] = rho0.copy()
    else:
        storage = []

    # whether to make observation
    num_observable = len(listofobservable)
    if num_observable > 0:
        outcome = np.zeros((num_observable, N+1),dtype=complex)
        for k in range(num_observable):
            outcome[k,0] = (rho0*listofobservable[k]).tr()
    else:
        outcome = []

    rho = rho0.copy()
    for j in range(1,N+1):
        rho = solvername(lbeq, rho, dt)
        if to_storage:
            storage[j] = rho.copy()
        if num_observable > 0:
            for k in range(num_observable):
                outcome[k,j] = (rho*listofobservable[k]).tr()

    return (rho, storage, outcome)


def get_terminal_error(solvername, lbeq, rho0, T, N, exactrho=None):
    """
        Return the error at the terminal time T.
        solvername is the name of previous a few functions
        lbeq, rho0, T, N are parameters.
    """

    rho_final, _, _ = detsolver(solvername, lbeq, rho0, T/N, N)
    if exactrho != None:
        err =  (exactrho - rho_final).norm()
        return err
    else:
        if lbeq.lb_superop == None:
            lbeq.get_lindblad_superop()
        exactrho = ((lbeq.lb_superop*T).expm())(rho0)
        err =  (exactrho - rho_final).norm()
        return err
