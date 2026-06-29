# Trabajo Práctico N.º 3 — Ecuación de Poisson 1D

Introducción al Modelado Continuo — Primer cuatrimestre 2026

**Integrantes:** ____________________________   **Nro. de libreta:** __________

**Integrantes:** ____________________________   **Nro. de libreta:** __________

**Integrantes:** ____________________________   **Nro. de libreta:** __________

---

## 1. Problema

Se busca resolver numéricamente la ecuación de Poisson en una dimensión, con condición de Dirichlet en $x=0$ y de Neumann en $x=1$:

$$
\begin{cases}
u''(x) = f(x) & x \in (0,1) \\
u(0) = \alpha & \text{(Dirichlet)} \\
u'(1) = \beta & \text{(Neumann)}
\end{cases}
$$

Se trabaja sobre una malla uniforme $x_j = h\,j$ con $j = 0, 1, \dots, 2^n+1$ y paso $h = \dfrac{1}{2^n+1}$. Es decir, hay $N = 2^n + 2$ nodos, con $x_0 = 0$ y $x_{2^n+1} = 1$.

La implementación se hizo en Python (numpy y scipy). El código está en `tp3.py` y los gráficos en los archivos PNG que se incluyen en este informe.

---

## 2. Modelo numérico

El método elegido es el de **diferencias finitas**: se reemplazan las derivadas de la ecuación por cocientes incrementales evaluados en los nodos de la malla, lo que transforma la ecuación diferencial en un sistema lineal $A\,u = b$.

### 2.1 Aproximación de las derivadas

**Derivada segunda (nodos interiores) — diferencias centradas:**

$$
u''(x_j) \approx \frac{u_{j-1} - 2u_j + u_{j+1}}{h^2}
$$

Para ver de dónde sale la fórmula, se desarrollan por Taylor los valores a izquierda y derecha del nodo $x_j$:

$$
u(x_j+h) = u(x_j) + h u'(x_j) + \frac{h^2}{2}u''(x_j)
          + \frac{h^3}{6}u'''(x_j) + \frac{h^4}{24}u^{(4)}(x_j) + O(h^5)
$$

$$
u(x_j-h) = u(x_j) - h u'(x_j) + \frac{h^2}{2}u''(x_j)
          - \frac{h^3}{6}u'''(x_j) + \frac{h^4}{24}u^{(4)}(x_j) + O(h^5)
$$

Al sumar ambos desarrollos se cancelan los términos impares:

$$
u_{j+1} - 2u_j + u_{j-1} = h^2 u''(x_j) + O(h^4)
$$

Luego, dividiendo por $h^2$:

$$
u''(x_j) = \frac{u_{j-1} - 2u_j + u_{j+1}}{h^2} + O(h^2)
$$

**Condición de Neumann en $x=1$ — diferencias backward de orden 2 (3 nodos):**

$$
u'(x_{N-1}) \approx \frac{3u_{N-1} - 4u_{N-2} + u_{N-3}}{2h} = \beta
$$

Como en el extremo derecho no hay nodos a la derecha, se usan los tres últimos nodos y se desarrollan hacia atrás alrededor de $x_{N-1}=1$:

$$
u_{N-2} = u_{N-1} - h u'(x_{N-1}) + \frac{h^2}{2}u''(x_{N-1}) + O(h^3)
$$

$$
u_{N-3} = u_{N-1} - 2h u'(x_{N-1}) + 2h^2 u''(x_{N-1}) + O(h^3)
$$

La combinación $3u_{N-1} - 4u_{N-2} + u_{N-3}$ cancela los términos de orden cero y segundo, y deja:

$$
3u_{N-1} - 4u_{N-2} + u_{N-3}
= 2h\,u'(x_{N-1}) + O(h^3)
$$

Por lo tanto:

$$
u'(x_{N-1}) =
\frac{3u_{N-1} - 4u_{N-2} + u_{N-3}}{2h} + O(h^2)
$$

### 2.2 Armado del sistema lineal $A\,u = b$

Tomando como incógnitas el valor de $u$ en todos los nodos, cada fila de $A$ es una ecuación: la fila 0 impone la condición de Dirichlet, las filas interiores usan las diferencias centradas y la última fila usa la diferencia backward de la condición de Neumann. En forma explícita:

- **Fila 0** (Dirichlet): $u_0 = \alpha$, es decir $A_{0,0}=1$ y $b_0=\alpha$.
- **Filas interiores** $i = 1,\dots,N-2$ (centradas):

$$
\frac{1}{h^2}u_{i-1} - \frac{2}{h^2}u_i + \frac{1}{h^2}u_{i+1} = f(x_i)
$$

- **Fila $N-1$** (Neumann backward):

$$
\frac{1}{2h}u_{N-3} - \frac{4}{2h}u_{N-2} + \frac{3}{2h}u_{N-1} = \beta
$$

### 2.3 Características del método

- **Orden 2.** Las dos aproximaciones usadas son $O(h^2)$, por lo que el método global también es de segundo orden: al reducir el paso $h$ a la mitad, el error debería dividirse aproximadamente por $4$.
- **Matriz rala y en banda.** $A$ tiene a lo sumo $3$ elementos no nulos por fila (es tridiagonal en el interior, más la fila de Dirichlet y la de Neumann). Esta estructura permite resolverla de manera eficiente.
- **Dos estrategias de resolución.** Se usa una matriz **densa** (con `np.linalg.solve`), que ocupa memoria del orden de $N^2$, y una matriz **esparsa** (con `scipy.sparse` y `spsolve`), que guarda solo los elementos no nulos y ocupa memoria del orden de $N$.

---

<div style="page-break-before: always;"></div>

## 3. Resultados

### 3.1 Planteo del sistema para $f(x)=\operatorname{sen}(128\pi x)$ (ítem 1)

Con $f(x) = \operatorname{sen}(128\pi x)$, $\alpha = 2$ y $\beta = 1$, se construyó la matriz $A$ y el vector $b$. Para $n$ chico el sistema entra en papel y se ve claramente la estructura descrita en la sección 2.

$n = 1 \rightarrow N = 4$ nodos, $h = 1/3$:

```
A =
[[  1.    0.    0.    0. ]
 [  9.  -18.    9.    0. ]
 [  0.    9.  -18.    9. ]
 [  0.    1.5  -6.    4.5]]

b = [ 2.00   0.87  -0.87   1.00 ]
```

$n = 2 \rightarrow N = 6$ nodos, $h = 1/5$:

```
A =
[[  1.    0.    0.    0.    0.    0. ]
 [ 25.  -50.   25.    0.    0.    0. ]
 [  0.   25.  -50.   25.    0.    0. ]
 [  0.    0.   25.  -50.   25.    0. ]
 [  0.    0.    0.   25.  -50.   25. ]
 [  0.    0.    0.    2.5 -10.    7.5]]

b = [ 2.00  -0.95  -0.59   0.59   0.95   1.00 ]
```

Se observa la primera fila igual a la identidad (Dirichlet), las filas centrales tridiagonales con los coeficientes $(1,-2,1)/h^2$ y la última fila con los tres términos $(1,-4,3)/(2h)$ de la condición de Neumann.

### 3.2 Error y orden de convergencia (ítem 2)

La solución exacta (integrando $u''=\operatorname{sen}(128\pi x)$ dos veces y aplicando las condiciones de borde, con $k=128\pi$) es:

$$
u(x) = -\frac{\operatorname{sen}(128\pi x)}{(128\pi)^2}
       + \left(1 + \frac{1}{128\pi}\right)x + 2
$$

Se resolvió el sistema para $n = 3, \dots, 14$ con matrices densas y se midió el error en norma infinito $\| u_{\text{num}} - u_{\text{exacta}} \|_\infty$ (el mayor error en valor absoluto sobre todos los nodos):

| n  | h        | error_inf |
|----|----------|-----------|
| 3  | 1.11e-01 | 1.86e-01  |
| 4  | 5.88e-02 | 6.40e-02  |
| 5  | 3.03e-02 | 8.67e-02  |
| 6  | 1.54e-02 | 1.62e-01  |
| 7  | 7.75e-03 | 2.35e-03  |
| 8  | 3.89e-03 | 1.42e-03  |
| 9  | 1.95e-03 | 5.60e-04  |
| 10 | 9.76e-04 | 1.55e-04  |
| 11 | 4.88e-04 | 3.96e-05  |
| 12 | 2.44e-04 | 9.96e-06  |
| 13 | 1.22e-04 | 2.50e-06  |
| 14 | 6.10e-05 | 6.09e-07  |

La siguiente figura compara la solución numérica con la exacta para algunos $n$. Para mallas con $n$ chico ($n=3$, $n=6$) la solución todavía no queda bien representada y se separa de la exacta; con $n=10$ coinciden.

![Solución numérica comparada con la exacta para distintos valores de n](tp3_ej2_soluciones.png)

Para estimar el orden del método se grafica el error en función del paso $h$ en escala log-log. Para $h$ grande el error se mantiene alto (la malla todavía no representa bien la función), y recién para $h$ chico el error decae de manera regular. Por eso el orden se estima ajustando una recta a esa zona ($n \ge 10$). En la figura se muestra una recta de referencia con pendiente $2$ extendida sobre todo el rango del gráfico:

![Error en norma infinito en escala log-log con referencia de pendiente 2](tp3_ej2_error_loglog.png)

$$
\text{pendiente} \approx 1.997 \approx 2
$$

La pendiente $\approx 2$ confirma que el método es de **orden 2**, como se esperaba a partir de las aproximaciones usadas.

### 3.3 Matrices esparsas y mayor $n$ posible (ítem 3)

Se repite el planteo del ítem 1, pero usando $f(x)=e^{-x^2}$, $\alpha=2$ y $\beta=1$, y armando la matriz como **esparsa**. La estructura de $A$ es la misma que antes, porque no cambiaron ni la malla ni las aproximaciones de las derivadas; lo que cambia es el vector $b$.

Las ecuaciones del sistema quedan:

- **Fila 0** (Dirichlet):

$$
u_0 = 2
$$

- **Filas interiores** $i=1,\dots,N-2$:

$$
\frac{1}{h^2}u_{i-1} - \frac{2}{h^2}u_i + \frac{1}{h^2}u_{i+1}
= e^{-x_i^2}
$$

- **Fila $N-1$** (Neumann backward):

$$
\frac{1}{2h}u_{N-3} - \frac{4}{2h}u_{N-2}
+ \frac{3}{2h}u_{N-1} = 1
$$

En forma vectorial, el lado derecho es:

$$
b =
\begin{bmatrix}
2 \\
e^{-x_1^2} \\
e^{-x_2^2} \\
\vdots \\
e^{-x_{N-2}^2} \\
1
\end{bmatrix}
$$

Por ejemplo, para $n=2$ se tiene $h=1/5$ y $x=(0,0.2,0.4,0.6,0.8,1)$. La matriz $A$ es la misma estructura tridiagonal mostrada en el ítem 1 para ese $n$, pero:

```
b = [ 2.00   0.96   0.85   0.70   0.53   1.00 ]
```

Se emplearon los métodos `scipy.sparse` y `spsolve`. La matriz tiene solo 3 elementos no nulos por fila (con excepción de la primera, donde hay 1), por lo que conviene guardarla como esparsa: se evita el costo $O(N^2)$ de una matriz densa y se aprovecha la estructura tridiagonal de la matriz.

Para estimar el mayor $n$ utilizable se hizo una prueba empírica en la máquina local, que reporta **32 procesadores lógicos**. El criterio usado fue tomar como límite práctico el mayor $n$ que se resolvió con `spsolve` en un tiempo razonable sin que el proceso falle ni se vuelva impráctico para una ejecución local.

Tiempos medidos:

| n  | N         | tiempo spsolve |
|----|-----------|----------------|
| 18 | 262.146   | 1.11 s         |
| 19 | 524.290   | 2.25 s         |
| 20 | 1.048.578 | 4.62 s         |
| 21 | 2.097.154 | 9.28 s         |
| 22 | 4.194.306 | 19.22 s        |

Al aumentar $n$ en una unidad, la cantidad de incógnitas se duplica. En la tabla se ve que el tiempo también crece aproximadamente por un factor cercano a $2$. Con este criterio práctico, el mayor caso resuelto fue $n=22$, con $N=4.194.306$ incógnitas.

En general no existe un $n$ máximo único que sea válido para cualquier máquina local: depende de la RAM disponible, del procesador, del método de resolución utilizado y de cuánto tiempo se considere razonable esperar. Lo que sí es general es que, como $N=2^n+2$, al aumentar $n$ el tamaño del sistema crece exponencialmente. Por lo tanto, para toda máquina concreta habrá algún $n$ suficientemente grande que ya no será resoluble de forma práctica.

La figura debajo muestra la solución obtenida para $n=22$ (malla muy fina) y la función $f(x)=e^{-x^2}$:

![Solución con matriz esparsa para n igual a 22](tp3_ej3_solucion.png)

---

## 4. Conclusiones

- El esquema de diferencias finitas (centradas para la derivada segunda y backward de orden 2 para la condición de Neumann) resuelve correctamente el problema mixto Dirichlet/Neumann.
- El método es de **orden 2**: la pendiente del gráfico log-log resultó $\approx 2$, lo que coincide con que ambas aproximaciones son $O(h^2)$.
- La matriz del problema es esparsa (casi tridiagonal). Aprovechar esa estructura baja la memoria de $O(N^2)$ a $O(N)$ y permite resolver con mallas mucho más finas que con matrices densas en la misma máquina.
