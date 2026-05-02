import numpy as np

# Ecuaciones diferenciales del sistema en formato para solve_ivp
# El estado es [mu, y] donde mu = 1/r  e  y = dmu/dtheta
def derivadas(theta, estado, alpha, delta):
    mu = estado[0]
    y  = estado[1]

    dmu_dtheta = y
    dy_dtheta  = -mu + 1.0/alpha + delta * mu**2

    return [dmu_dtheta, dy_dtheta]


# Convierte de coordenadas polares (theta, mu=1/r) a cartesianas (x, y)
def a_cartesiano(thetas, mu_arr):
    r = 1.0 / np.array(mu_arr)
    x = r * np.cos(np.array(thetas))
    y = r * np.sin(np.array(thetas))
    return x, y