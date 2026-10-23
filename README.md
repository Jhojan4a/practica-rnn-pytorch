# Practica 06: Redes Recurrentes (RNN y LSTM) en PyTorch

## 1. Inspeccion de Dimensiones (Tensores 3D)
Con `n_puntos = 600` y `seq_len = 50`, `X.shape` resulta ser `[550, 50, 1]`:
- **Eje 0 (Batch Size = 550):** Numero de subsecuencias generadas por la ventana deslizante.
- **Eje 1 (Sequence Length = 50):** Longitud de pasos temporales analizados.
- **Eje 2 (Features = 1):** Unica caracteristica observada (amplitud escalar de la senal).

## 2. Modificacion de Hiperparametros (hidden_size)
Incrementar `hidden_size` de 16 a 64 dispara los parametros entrenables debido a la matriz recursiva interna de pesos ($W_{hh}$), cuyo crecimiento es cuadratico ($hidden\_size \times hidden\_size$).

## 3. Comportamiento en Secuencias Largas: RNN vs LSTM
- **Amnesia matematica en RNN Simple:** A 50 pasos temporales, el gradiente por BPTT sufre desvanecimiento exponencial, reduciendo la capacidad de retener el inicio de la serie.
- **Superioridad de LSTM:** Sus compuertas logicas y celda de estado mantienen constante la senal del gradiente, permitiendo un ajuste mas rapido y menor error (MSE Loss).
