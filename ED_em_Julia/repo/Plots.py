import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.special import factorial


#n = np.arange(1, 10)

#dim_tot = 4**n

#dim_sym = (4**n)/n

#plt.semilogy(n, dim_tot, label = r'$\dim(\mathcal{H}_{total})$')
#plt.semilogy(n, dim_sym, label = r'$\dim(\mathcal{H}_{sim})$')
#plt.xlabel(r'$N_{sitios}$', fontsize = 20)
#plt.ylabel(r'$\dim(\mathcal{H})$', fontsize = 20)

#plt.tick_params(labelsize = 15)

#plt.grid(visible = True)

#plt.legend()

#plt.savefig('Dims.pdf', format = 'pdf', bbox_inches = 'tight')
#plt.show()



df = pd.read_csv(r'Termodyn_N8.csv', sep = ',')

T = np.array(df['Temperature'])

labels_cv = ['CV1', 'CV2', 'CV3', 'CV4', 'CV5']
labels_E = ['Em1', 'Em2', 'Em3', 'Em4', 'Em5']
labels_S = ['S1', 'S2', 'S3', 'S4', 'S5']
U = [0, 1, 5, 10, 20]

for i in range(len(U)):
    cv_arr = np.array(df[labels_cv[i]])/8
    Em_arr = np.array(df[labels_E[i]])/8
    S_arr = np.array(df[labels_S[i]])/8
    #plt.plot(T, cv_arr, label = f'U = {U[i]}')
    #plt.plot(T, S_arr, label = f'U = {U[i]}')
    #plt.plot(T, Em_arr, label = f'U = {U[i]}')

#plt.xlabel('T', fontsize = 20)
#plt.ylabel(r'$E_m/N$', fontsize = 20)
#plt.ylabel(r'$C_V/N$', fontsize  =20)
#plt.ylabel('S/N', fontsize = 20)

#plt.tick_params(labelsize = 15)
#plt.grid(visible = True)

#plt.legend()

#plt.savefig('Energia_media_N8.pdf', format = 'pdf', bbox_inches = 'tight')
#plt.show()
eta = 0.01

def delta(x):
    return (1/np.pi)*eta/(x**2 + eta**2)

labels = ['En1', 'En2', 'En3', 'En4', 'En5']

df1 = pd.read_csv(r'Niveis_N8.csv')

Enl = np.empty((5, 4900))

for i in range(len(U)):
    Enl_arr = np.array(df1[labels[i]])
    Enl[i] = Enl_arr

Nom = 2000

omega = np.zeros((5, Nom))

DOS = np.zeros((5, Nom))

for i in range(len(U)):
    omega[i] = np.linspace(np.min(Enl[i]), np.max(Enl[i]), Nom)

    for j in range(4900):
        DOS[i] += delta(omega[i]-Enl[i, j])

DOS = DOS/8

plt.plot(omega[3]+8*U[3]/4, DOS[3], label = f'U = {3}')
plt.xlabel(r'$\omega$', fontsize = 20)
plt.ylabel(r'$DOS(\omega)$', fontsize = 20)


plt.tick_params(labelsize = 15)
plt.grid(visible = True)

plt.legend()

#plt.savefig('DOSU10_N8.pdf', format = 'pdf', bbox_inches = 'tight')
plt.show()
