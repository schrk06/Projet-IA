import random as rd
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations




"""goodint = False
while not goodint:
    N  = int(input("Enter the number of cities: "))
    try:
        if(N > 0):
            goodint = True
        else:
            print("That's not a positive int, try again !")    
    except ValueError:
        print("That's not an int!, try again !")"""
    
N=5
M = 10 #It represents the population's dimension
x = np.random.uniform(0, 1, N) #coordonné des N villes sur l'axe X
y = np.random.uniform(0, 1, N) #coordonné des N villes sur l'axe Y

def generate_population():
    """Génère aléatoirement une population """
    all_perms = list(permutations(range(0, N )))  # Generate all the permutaions
    pop_mat = rd.sample(all_perms, min(M, len(all_perms)))  # Select M (or less if len(all_perms)<M) random permutaions
    return [list(p) for p in pop_mat]

def fitness(chemin):
    """ Calcule la distance totale parcourue pour un chemin donné. """
    xy = np.column_stack((x[chemin], y[chemin]))  # Coordonnées des villes dans l'ordre du chemin
    distance = np.sum(np.sqrt(np.sum((xy - np.roll(xy, -1, axis=0))**2, axis=1)))  
    return distance

