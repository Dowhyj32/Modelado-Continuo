import numpy as np

def detectar_digito(frecuencias, magnitud):
    tonos_bajos = np.array([697, 770, 852, 941])
    tonos_altos = np.array([1209, 1336, 1477])

    tabla = {
        (697,1209): '1',
        (697,1336): '2',
        (697,1477): '3',

        (770,1209): '4',
        (770,1336): '5',
        (770,1477): '6',

        (852,1209): '7',
        (852,1336): '8',
        (852,1477): '9',

        (941,1209): '*',
        (941,1336): '0',
        (941,1477): '#'
    }
    
    energias_bajas = []
    
    for f in tonos_bajos:
        i = np.argmin(np.abs(frecuencias - f))
        energias_bajas.append(magnitud[i])
        
    energias_altas = []
    
    for f in tonos_altos:
        i = np.argmin(np.abs(frecuencias-f))
        energias_altas.append(magnitud[i])
        
    idx_baja = np.argmax(energias_bajas)
    idx_alta = np.argmax(energias_altas)

    freq_baja = tonos_bajos[idx_baja]
    freq_alta = tonos_altos[idx_alta]

    digito = tabla.get((int(freq_baja), int(freq_alta)), '?')

    return digito, freq_baja, freq_alta
