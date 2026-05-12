import numpy as np
import time

def DFT(x):
    inicio = time.perf_counter()
    N = len(x)
    
    # Vector frecuencias discretas
    X = np.zeros(N, dtype=complex)

    for n in range(N):          # índice de frecuencia
        suma = 0

        for k in range(N):      # índice temporal / muestra
            suma += x[k] * np.exp(-1j * 2 * np.pi * n * k / N)

        X[n] = suma

    fin = time.perf_counter()
    tiempo_total = fin - inicio
    
    return X, tiempo_total