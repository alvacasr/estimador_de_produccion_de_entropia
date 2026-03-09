import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

# coeficientes de fricción
k, e = 1, 1

def muestra(n_libertad, T1, T2, n_trayectorias):
    if n_libertad == 2:
        cov = [
            [(7 * T1 + T2) / (12.0 * k), (T1 + T2) / (6.0 * k)],
            [(T1 + T2) / (6.0 * k), (T1 + 7 * T2) / (12.0 * k)],
        ]
        N = MultivariateNormal(torch.zeros(2), torch.tensor(cov))
        posiciones = N.sample((n_trayectorias,))

    return posiciones

def simulacion(n_trayectorias, longitud_trayectoria, n_libertad, T1, T2, dt, procesamiento='cpu', seed=0):
    T = torch.linspace(T1, T2, n_libertad).to(procesamiento)
    dt = torch.tensor(dt).to(procesamiento)
    dt2 = torch.sqrt(dt).to(procesamiento)
    
    trayectoria = torch.zeros(n_trayectorias, longitud_trayectoria, n_libertad).to(procesamiento)
    desviacion = torch.zeros(n_libertad, n_libertad).to(procesamiento)
    posicion = muestra(n_libertad, T1, T2, n_trayectorias).to(procesamiento)
    
    for i in range(n_libertad):
        if i > 0:
            desviacion[i][i - 1] = k / e
        if i < n_libertad - 1:
            desviacion[i][i + 1] = k / e
        desviacion[i][i] = -2 * k / e

    fz = torch.zeros(n_libertad).to(procesamiento)
    for i in range(n_libertad):
        fz[i] = torch.sqrt(2 * e * T[i])

    torch.manual_seed(seed)
            
    for it in range(longitud_trayectoria):
        fuerza = torch.randn(n_trayectorias, n_libertad, device=procesamiento)
        fuerza *= fz

        fuerza_desviacion = torch.einsum('ij,aj->ai', desviacion, posicion)

        posicion += (fuerza_desviacion * dt + fuerza * dt2) / e

        trayectoria[:, it, :] = posicion

    return trayectoria

def p_ee(n_libertad, x, T1, T2):
    if n_libertad == 2:
        x1, x2 = x[:, :, 0], x[:, :, 1]
        return torch.exp(
            -0.5
            * (
                4
                * k
                * (
                    T2 * (7 * x1 ** 2 - 4 * x1 * x2 + x2 ** 2)
                    + T1 * (x1 ** 2 - 4 * x1 * x2 + 7 * x2 ** 2)
                )
            )
            / (T1 ** 2 + 14 * T1 * T2 + T2 ** 2)
        )

    return -1

def dif_entropia_shannon(trayectoria, T1, T2):
    n_libertad = trayectoria.shape[-1]

    entropia_anterior = -torch.log(p_ee(n_libertad, trayectoria[:, :-1, :], T1, T2))
    entropia_siguiente = -torch.log(p_ee(n_libertad, trayectoria[:, 1:, :], T1, T2))

    return entropia_siguiente - entropia_anterior

def dif_entropia_medio(trayectoria, T1, T2):
    n_libertad = trayectoria.shape[-1]

    desviacion = torch.zeros(n_libertad, n_libertad).to(trayectoria.device)
    T = torch.linspace(T1, T2, n_libertad).to(trayectoria.device)

    for i in range(n_libertad):
        if i > 0:
            desviacion[i][i - 1] = k / e
        if i < n_libertad - 1:
            desviacion[i][i + 1] = k / e
        desviacion[i][i] = -2 * k / e

    x_anterior = trayectoria[:, :-1, :]
    x_siguiente = trayectoria[:, 1:, :]
    dx = x_siguiente - x_anterior

    Fx = ((x_siguiente + x_anterior)/2) @ desviacion

    dQ = Fx * dx
    entropia = torch.sum(dQ / T, dim=2)

    return entropia