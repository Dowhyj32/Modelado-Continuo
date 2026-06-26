import numpy as np
from scipy.linalg import solve # para matrices sin aprovechar esparcidad
from scipy.sparse import diags_array # para matrices esparzas (ralas)
from scipy.sparse.linalg import spsolve # métodos para matrices esparzas
import matplotlib.pyplot as plt
import time as time # para medir tiempos de ejecución

""" Problema a resolver:
    ecuación unidimensional de Poisson con condiciones de borde tipo Dirichlet
    /
    |u_xx(x) = f(x) para x en (0, 1)
    |u(0) = alfa
    |u(1) = beta
    \
"""
def f(x):
    return np.sin(2.0*np.pi*x)*2.0*np.pi*2.0*np.pi

# Solución exacta ue''=f, ue(0)=a=alfa, ue(1)=b=beta
def ue(x,a,b):
    return -np.sin(2.0*np.pi*x)+(b-a)*x+a

alfa = 1.5
beta = -0.5

################################################################################
# Para probar
################################################################################
k = 13 # 13  (cambiar a más chico si no anda)
# Probar otros valores mayores y prestar atención al mensaje de error.
# Comentando/eliminando la matriz A densa y sus derivados ¿Hasta qué k llega en su máquina?
################################################################################
nx = 2**k # nx cantidad de puntos intermedios entre x=0 y x=1

# Puntos en el intervalo
h = 1.0/(nx+1) # Paso
x = np.array([0.0 + h*i for i in range(nx+2)]) # x[0]=0.0, x[nx+1]=1.0

""" Esquema diferencias centradas.
Aproximamos la ecuación diferencial por:
    u_xx(x) ~ ( u(x+h) - 2*u(x)  + u(x-h) )/(h*h) = f(x)
que en los puntos de nuestro intervalo se calcula
    u_xx(x[i]) ~ ( u(x[i+1]) - 2*u(x[i]) + u(x[i-1]) )/(h*h) ~ ( u[i+1] - 2*u[i] + u[i-1] )/(h*h) = f(x[i])
donde u[i] será nuestra solución aproximada para u(x[i]).
 
Como tenemos u[0] = u(x[0]) = alfa fijo, u[0] no es incógnita.
La ecuación ( u[2] - 2*u[1] + u[0] )/(h*h) = f(x[1]) hay que reescribirla como
 ( u[2] - 2*u[1] )/(h*h) = f(x[1]) - u[0]/(h*h) = f(x[1]) - alfa/(h*h)
Lo mismo en el otro extremo

Las ecuaciones para i entre 1 y nx inclusive, forman un sistema lineal Au = b con incógnitas u[i].
"""

# Armado de la matriz A de diferenciación para este esquema.

############################### caso denso ################################
# matriz con -2.0 en la diagonal principal: np.diag(-2.0*np.ones(nx)) 
# matriz con 1.0 en la primera diagonal arriba de la principal: np.diag(1.0*np.ones(nx-1),+1)
# matriz con 1.0 en la primera diagonal abajo de la principal: np.diag(1.0*np.ones(nx-1),-1)
A = 1.0/(h*h) *( np.diag(-2.0*np.ones(nx)) + np.diag(1.0*np.ones(nx-1),+1) + np.diag(1.0*np.ones(nx-1),-1) )
###########################################################################

############################## caso esparzo ###############################
# Versión para matrices esparzas (ralas)
# diags_array usa offsets= para desplazar las diagonales respecto de la principal.
# tocsc() pasa a formato "Compressed Sparse Column" para usar spsolve.
Asp = 1.0/(h*h) * diags_array([-2.0*np.ones(nx),1.0*np.ones(nx-1),1.0*np.ones(nx-1)],offsets=[0,1,-1]).tocsc()
###########################################################################

# La ecuación A u_aux = b solo se resolverá para x[i] con i de 1 a nx inclusive, así que los índices para estas u_aux y b estarán corridos respecto al planteo del esquema:
# u_aux[0] = u[1], u_aux[1] = u[2], b[1] = f(x[2]), etc.

b = np.zeros(nx)
for i in range(nx):
    b[i] = f(x[i+1])
b[0] += -alfa/(h*h)
b[nx-1] += -beta/(h*h)

# Resuelve el problema lineal A u_aux = b: solve() para matrices en general, spsolve() para esparzas.

t0 = time.time()
############################### caso denso ################################
u_aux = solve(A,b)
###########################################################################
t1 = time.time()

t0sp = time.time()
############################## caso esparzo ###############################
usp_aux = spsolve(Asp,b)
###########################################################################
t1sp = time.time()

# Error de truncado
et = A@u_aux-b
etsp = Asp@usp_aux-b

u = np.concatenate(([alfa], u_aux, [beta])) # agraga u[0] y u[nx+1]
usp = np.concatenate(([alfa], usp_aux, [beta])) # agraga u[0] y u[nx+1] solución con métodos de matrices esparzas
uex = ue(x, alfa, beta) # Cálculo de la solución exacta

# Error en norma infinito
error = np.max(np.abs(uex-u))
errorsp = np.max(np.abs(uex-usp))

# Error de truncado en norma infinito
e_d_t = np.max(np.abs(et))
e_d_tsp = np.max(np.abs(etsp))

tiempo = t1-t0
tiemposp = t1sp-t0sp
print("Con matriz densa")
print("error =", error, "  error de truncado = ", e_d_t, "  tiempo =", tiempo)
print("Con matriz esparza")
print("error =", errorsp, "  error de truncado = ", e_d_tsp, "  tiempo =", tiemposp)

fig, ax = plt.subplots()
plt.plot(x, u, 'rx', label='sol. numérica densa')
plt.plot(x, usp, 'y.', label='sol. numérica esparza')
plt.plot(x, uex, 'b', label='sol. exacta')
ax.legend()
plt.savefig("ejemplo_edp_sol.png", dpi=300)
plt.close()

fig, ax = plt.subplots()
ax.set_title("Error puntual")
plt.plot(x, u-uex, 'r.', label='sol. densa')
plt.plot(x, usp-uex, 'y', label='sol. esparza')
ax.legend()
plt.savefig("ejemplo_edp_error.png", dpi=300)
plt.close()
