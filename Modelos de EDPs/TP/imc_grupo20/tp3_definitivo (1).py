import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
import time

# ---------------------------------------------------------------------
# Funciones f(x) de cada ejercicio
# ---------------------------------------------------------------------
def f_seno(x):
    # f(x) = sen(128*pi*x)   (ejercicios 1 y 2)
    return np.sin(128.0 * np.pi * x)


def f_exp(x):
    # f(x) = e^{-x^2}        (ejercicio 3)
    return np.exp(-x * x)


# ---------------------------------------------------------------------
# Solucion exacta (se obtiene integrando dos veces u'' = f y
# aplicando las condiciones de borde). Sirve para medir el error.
# ---------------------------------------------------------------------
def solucion_exacta_seno(x):
    # u'' = sen(128 pi x)
    # u   = -sen(128 pi x)/(128 pi)^2 + C1 x + C2
    # u(0)=2  -> C2 = 2
    # u'(1)=1 -> C1 = 1 + 1/(128 pi)
    k = 128.0 * np.pi
    c1 = 1.0 + 1.0 / k
    c2 = 2.0
    return -np.sin(k * x) / (k * k) + c1 * x + c2

# ---------------------------------------------------------------------
# Construcción del sistema lineal A u = b con matriz DENSA.
# ---------------------------------------------------------------------
def construir_sistema_denso(n, f, alpha, beta):
    N = 2 ** n + 2                 # cantidad de puntos
    h = 1.0 / (2 ** n + 1)         # paso de la malla

    # Vector de posiciones x_j = h * j
    x = np.zeros(N)
    for j in range(N):
        x[j] = h * j

    A = np.zeros((N, N))
    b = np.zeros(N)

    # Fila 0: condición de Dirichlet u(0) = alpha
    A[0, 0] = 1.0
    b[0] = alpha

    # Filas interiores: diferencias centradas para la derivada segunda
    for i in range(1, N - 1):
        A[i, i - 1] = 1.0 / h ** 2
        A[i, i] = -2.0 / h ** 2
        A[i, i + 1] = 1.0 / h ** 2
        b[i] = f(x[i])

    # Última fila: condición de Neumann u'(1) = beta con método backward orden 2
    A[N - 1, N - 3] = 1.0 / (2.0 * h)
    A[N - 1, N - 2] = -4.0 / (2.0 * h)
    A[N - 1, N - 1] = 3.0 / (2.0 * h)
    b[N - 1] = beta

    return A, b, x, h


# ---------------------------------------------------------------------
# Misma construcción pero ahora con matriz ESPARSA (formato lil para armarla
# y luego csr para resolver). Solo se guardan los elementos no
# nulos: ~3 por fila, o sea que en memoria es O(N) en lugar de O(N^2).
# ---------------------------------------------------------------------
def construir_sistema_esparso(n, f, alpha, beta):
    N = 2 ** n + 2
    h = 1.0 / (2 ** n + 1)

    x = np.zeros(N)
    for j in range(N):
        x[j] = h * j

    A = sparse.lil_matrix((N, N))
    b = np.zeros(N)

    # Dirichlet
    A[0, 0] = 1.0
    b[0] = alpha

    # Interiores
    for i in range(1, N - 1):
        A[i, i - 1] = 1.0 / h ** 2
        A[i, i] = -2.0 / h ** 2
        A[i, i + 1] = 1.0 / h ** 2
        b[i] = f(x[i])

    # Neumann (backward orden 2)
    A[N - 1, N - 3] = 1.0 / (2.0 * h)
    A[N - 1, N - 2] = -4.0 / (2.0 * h)
    A[N - 1, N - 1] = 3.0 / (2.0 * h)
    b[N - 1] = beta

    return A.tocsr(), b, x, h


# =====================================================================
# EJERCICIO 1
# Plantear el sistema A u = b para f(x)=sen(128 pi x), alpha=2, beta=1.
# Mostrar las fórmulas usadas y algunas matrices A y b con n chico.
# =====================================================================
def ejercicio_1():
    print("=" * 60)
    print("EJERCICIO 1 - Planteo del sistema A u = b")
    print("=" * 60)
    print("f(x) = sen(128*pi*x),  alpha = 2,  beta = 1")
    print()
    print("Aproximaciones usadas:")
    print("  Derivada 2da (diferencias centradas, orden 2):")
    print("     u''(x_j) ~= (u_{j-1} - 2 u_j + u_{j+1}) / h^2")
    print("  Neumann en x=1 (backward orden 2):")
    print("     u'(x_{N-1}) ~= (3 u_{N-1} - 4 u_{N-2} + u_{N-3}) / (2h) = beta")
    print()

    alpha = 2.0
    beta = 1.0

    # Imprimimos matrices chicas para que entren bien en el trabajo escrito.
    for n in [1, 2]:
        A, b, x, h = construir_sistema_denso(n, f_seno, alpha, beta)
        print("-" * 60)
        print("n =", n, " ->  N =", 2 ** n + 2, "nodos,  h =", round(h, 5))
        print("Matriz A:")
        # Se imprime con pocos decimales para que se lea bien.
        np.set_printoptions(precision=2, suppress=True, linewidth=120)
        print(A)
        print("Vector b:")
        print(b)
        print()


# =====================================================================
# EJERCICIO 2
# Resolvemos para n=3..14, calculamos el error en norma infinito respecto a la
# solución exacta, graficamos log(error) vs log(h) y con eso estimamos el orden del error.
# =====================================================================
def ejercicio_2():
    print("=" * 60)
    print("EJERCICIO 2 - Error y orden del método")
    print("=" * 60)

    alpha = 2.0
    beta = 1.0

    lista_n = list(range(3, 15))   # n = 3, 4, ..., 14
    lista_h = []
    lista_error = []

    # Guardamos algunas soluciones para graficarlas después.
    soluciones_guardadas = {}
    n_para_graficar = [3, 6, 10]

    for n in lista_n:
        A, b, x, h = construir_sistema_denso(n, f_seno, alpha, beta)
        u_num = np.linalg.solve(A, b)
        u_exa = solucion_exacta_seno(x)

        # Error en norma infinito:
        error = 0.0
        for i in range(len(x)):
            diferencia = abs(u_num[i] - u_exa[i])
            if diferencia > error:
                error = diferencia

        lista_h.append(h)
        lista_error.append(error)
        print("n =", n, " h =", format(h, ".2e"), " error_inf =", format(error, ".3e"))

        if n in n_para_graficar:
            soluciones_guardadas[n] = (x, u_num, u_exa)

    # -----------------------------------------------------------------
    # Gráfico 1: soluciones numérica vs exacta para algunos n
    # -----------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for n in n_para_graficar:
        x, u_num, u_exa = soluciones_guardadas[n]
        plt.plot(x, u_num, label="numérica n=" + str(n))
    # La exacta es la misma curva siempre; la dibujamos una sola vez
    # usando la malla de mayor n.
    x_fino = soluciones_guardadas[max(n_para_graficar)][0]
    plt.plot(x_fino, solucion_exacta_seno(x_fino), "k--", label="exacta")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Solución de u'' = sen(128 pi x) para distintos n")
    plt.legend()
    plt.grid(True)
    plt.savefig("tp3_ej2_soluciones.png", dpi=120)
    plt.close()
    print("Guardado: tp3_ej2_soluciones.png")

    # -----------------------------------------------------------------
    # Gráfico 2: log(error) vs log(h) y recta ajustada (orden del método)
    # -----------------------------------------------------------------
    log_h = np.log10(np.array(lista_h))
    log_error = np.log10(np.array(lista_error))

    # Estimamos la pendiente usando usando n a partir de 10 (donde ya estamos en la zona de convergencia)
    n_asintotico = 10
    log_h_asin = []
    log_error_asin = []
    for k in range(len(lista_n)):
        if lista_n[k] >= n_asintotico:
            log_h_asin.append(log_h[k])
            log_error_asin.append(log_error[k])
    log_h_asin = np.array(log_h_asin)
    log_error_asin = np.array(log_error_asin)

    # Ajuste lineal: la pendiente de la recta ajustada es el orden del método.
    coeficientes = np.polyfit(log_h_asin, log_error_asin, 1)
    pendiente = coeficientes[0]

    # Recta de referencia con pendiente exactamente 2.
    ordenada_pendiente_2 = np.mean(log_error_asin - 2.0 * log_h_asin)
    recta_pendiente_2 = ordenada_pendiente_2 + 2.0 * log_h

    print("Pendiente del ajuste log-log en la zona asintótica (n >=",
          n_asintotico, ") =", round(pendiente, 3))

    plt.figure(figsize=(10, 6))
    plt.plot(log_h, log_error, "o", color="tab:blue")
    # Al lado de cada punto se escribe el valor de n correspondiente.
    for k in range(len(lista_n)):
        plt.annotate("n=" + str(lista_n[k]),
                     (log_h[k], log_error[k]),
                     textcoords="offset points", xytext=(6, 6), fontsize=9)
    plt.plot(log_h, recta_pendiente_2, "k-",
             label="referencia pendiente=2")
    plt.xlabel("log10(h)")
    plt.ylabel("log10(error_inf)")
    plt.title("Error vs paso de malla (escala log-log)")
    plt.legend()
    plt.grid(True)
    plt.savefig("tp3_ej2_error_loglog.png", dpi=120)
    plt.close()
    print("Guardado: tp3_ej2_error_loglog.png")
    print()

    return pendiente


# =====================================================================
# EJERCICIO 3
# Se repite el ítem 1 con f(x)=e^{-x^2} usando matrices esparsas. Se busca el mayor n
# que la máquina local pueda resolver y se grafica esa solución.
# =====================================================================
def ejercicio_3():
    print("=" * 60)
    print("EJERCICIO 3 - Matrices esparsas y mayor n posible")
    print("=" * 60)
    print("f(x) = e^{-x^2},  alpha = 2,  beta = 1")
    print()

    alpha = 2.0
    beta = 1.0

    # Probamos distintos n:

    n_a_probar = list(range(10, 23))
    n_maximo = None
    datos_maximo = None

    for n in n_a_probar:
        N = 2 ** n + 2
        try:
            t0 = time.time()
            A, b, x, h = construir_sistema_esparso(n, f_exp, alpha, beta)
            u_num = spsolve(A, b)
            t1 = time.time()
            print("n =", n, " N =", N, " resuelto en", round(t1 - t0, 2), "s")
            n_maximo = n
            datos_maximo = (x, u_num)
        except (MemoryError, Exception) as e:
            print("n =", n, " N =", N, " FALLO:", repr(e))
            break

    print()
    print("Mayor n resuelto con matrices esparsas:", n_maximo)
    print()

    # -----------------------------------------------------------------
    # Gráfico de la solución para el n más alto alcanzado
    # -----------------------------------------------------------------
    if datos_maximo is not None:
        x, u_num = datos_maximo
        plt.figure(figsize=(10, 6))
        plt.plot(x, u_num, "b-", label="numérica (esparsa) n=" + str(n_maximo))
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.title("Solución de u'' = e^{-x^2} con n=" + str(n_maximo) +
                  " (matrices esparsas)")
        plt.legend()
        plt.grid(True)
        plt.savefig("tp3_ej3_solucion.png", dpi=120)
        plt.close()
        print("Guardado: tp3_ej3_solucion.png")
    print()

    return n_maximo



def main():
    ejercicio_1()
    ejercicio_2()
    ejercicio_3()


if __name__ == "__main__":
    main()