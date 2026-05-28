# Detección automática de los picos DTMF en cada segmento.
# Para cada segmento se toma el pico de mayor magnitud en la banda baja
# [600, 1000] Hz y en la banda alta [1100, 1700] Hz; cada pico se asocia
# a la frecuencia DTMF estándar más cercana y se determina el dígito.

import numpy as np
from scipy.fft import fft, fftfreq

# Frecuencias DTMF estándar
frec_bajas_dtmf = [697, 770, 852, 941]
frec_altas_dtmf = [1209, 1336, 1477, 1633]
frec_dtmf = frec_bajas_dtmf + frec_altas_dtmf

dtmf_tabla = {
    (697, 1209): '1', (697, 1336): '2', (697, 1477): '3',
    (770, 1209): '4', (770, 1336): '5', (770, 1477): '6',
    (852, 1209): '7', (852, 1336): '8', (852, 1477): '9',
    (941, 1209): '*', (941, 1336): '0', (941, 1477): '#',
}

def detectar_picos(segmento, fs):
    N = len(segmento)
    esp = np.abs(fft(segmento))[:N//2]
    fr = fftfreq(N, d=1/fs)[:N//2]
    mask_baja = (fr >= 600) & (fr <= 1000)
    mask_alta = (fr >= 1100) & (fr <= 1700)
    f_baja = fr[np.argmax(esp * mask_baja)]
    f_alta = fr[np.argmax(esp * mask_alta)]
    ref_baja = min(frec_bajas_dtmf, key=lambda x: abs(x - f_baja))
    ref_alta = min(frec_altas_dtmf, key=lambda x: abs(x - f_alta))
    digito = dtmf_tabla[(ref_baja, ref_alta)]
    return f_baja, f_alta, ref_baja, ref_alta, digito

def procesar(lista, cortes_, fs, nombre):
    print(f"=== Señal {nombre} ===")
    digitos = []
    for idx, segmento in enumerate(lista):
        ini, fin = cortes_[2*idx], cortes_[2*idx+1]
        fb, fa, rb, ra, d = detectar_picos(segmento, fs)
        digitos.append(d)
        print(f"Seg {idx+1} [{ini:.2f}-{fin:.2f} s]: "
              f"f_baja = {fb:.1f} Hz → {rb} Hz  ·  "
              f"f_alta = {fa:.1f} Hz → {ra} Hz  →  dígito {d}")
    print(f"Número {nombre}: {''.join(digitos)}\n")
    return digitos