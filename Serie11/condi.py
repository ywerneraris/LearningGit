from numpy import *
from matplotlib.pyplot import *
from numpy.linalg import cond, norm

# bilde matrix A = A1 + epsilon*Aeps
A1 = array([[1, 1],
            [1, 1],
            [0, 0]])
Aeps = array([[+1, 0],
              [-1, 0],
              [+1, +1]])

def B_alpha(A, alpha):
    # TODO implement the matrix B_alpha
    return np.empty((A.shape[0]*2, A.shape[1]*2))

def compute_condition_numbers(eps):
    # TODO compute the condition numbers of all three matrices.
    return 1.0, 1.0, 1.0

n_epsilon = 20  # anzahl epsilons im intervall 1.e-5 < epsilon <= 1
epsilons = logspace(0, -5, num=n_epsilon)

# erstelle speicher fuer die verschiedenen Kondition s Werte
cond_A   = zeros(n_epsilon)
cond_ATA = zeros(n_epsilon)
cond_B1  = zeros(n_epsilon)
cond_Balpha = zeros(n_epsilon)


# berechne Kondition(en) fuer alle epsilons
for i, eps in enumerate(epsilons):
    cond_A[i], cond_B1[i], cond_Balpha[i] = compute_condition_numbers(eps)

# zeichne die Konditions Zahlen als Funktion von epsilon
loglog(epsilons, cond_A,   label=r'cond$_2({\bf A})$')
loglog(epsilons, cond_ATA, label=r'cond$_2({\bf A}^{T}{\bf A})$')
loglog(epsilons, cond_B1,  label=r'cond$_2({\bf B}_1)$')
loglog(epsilons, cond_Balpha, label=r'cond$_2({\bf B}_{\alpha})$')
grid(True)
legend()
xlabel(r'$\epsilon$')
savefig('condiLSP.pdf')
show()
