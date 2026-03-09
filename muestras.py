import random
from itertools import product

import torch

class muestra:
    def __init__(self, M, L, tamaño_entrada, procesamiento="cpu", entrenar=True):
        self.tamaño = M * (L - 1)
        self.M = M
        self.L = L
        self.tamaño_entrada = tamaño_entrada
        self.procesamiento = procesamiento
        self.entrenamiento = entrenar
        self.indice = 0

    def train(self):
        self.entrenamiento = True

    def eval(self):
        self.entrenamiento = False

    def __iter__(self):
        self.indice = 0
        return self

    def __next__(self):
        if self.entrenamiento:
            indice_entropia = torch.randint(self.M, (self.tamaño_entrada,), device=self.procesamiento)
            ind_trayectoria = torch.randint(0, (self.L - 1), (self.tamaño_entrada,), device=self.procesamiento)
            batch = (indice_entropia, ind_trayectoria)
            next_batch = (indice_entropia, ind_trayectoria + 1)
            return batch, next_batch
        else:
            indice_anterior = self.indice * self.tamaño_entrada
            indice_siguiente = (self.indice + 1) * self.tamaño_entrada
            if indice_anterior >= self.tamaño:
                raise StopIteration
            elif indice_siguiente >= self.tamaño:
                indice_siguiente = self.tamaño
            indice_entropia = torch.arange(indice_anterior, indice_siguiente, device=self.procesamiento) // (self.L - 1)
            ind_trayectoria = torch.arange(indice_anterior, indice_siguiente, device=self.procesamiento) % (self.L - 1)
            self.indice += 1
            batch = (indice_entropia, ind_trayectoria)
            next_batch = (indice_entropia, ind_trayectoria + 1)
            return batch, next_batch