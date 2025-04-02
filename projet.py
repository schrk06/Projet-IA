from math import factorial
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations




goodint = False
while not goodint:
    N  = int(input("Enter the number of cities: "))
    try:
        if(N > 0):
            goodint = True
        else:
            print("That's not a valid int, try again !")    
    except ValueError:
        print("That's not an int!, try again !")
    

M = 10 #It represents the population's dimension
x = np.random.uniform(0, 1, N) #coordonné des N villes sur l'axe X
y = np.random.uniform(0, 1, N) #coordonné des N villes sur l'axe Y

"""def generate_population():
    #Génère aléatoirement une population
    all_perms = list(permutations(range(0, N )))  # Generate all the permutaions
    pop_mat = rd.sample(all_perms, min(M, len(all_perms)))  # Select M (or less if len(all_perms)<M) random permutaions
    return [list(p) for p in pop_mat] """  #trop côuteuse 

def generate_population():
    """Génère M chemins aléatoires uniques sans calculer toutes les permutations."""
    max_permutations = min(M, factorial(N))  # Empêche d'aller au-delà de N!
    pop_set = set()
    
    while len(pop_set) < max_permutations:
        chemin = tuple(rd.sample(range(N), N))
        pop_set.add(chemin)
    return [list(chemin) for chemin in pop_set]

def fitness(chemin):
    """ Calcule la distance totale parcourue pour un chemin donné. """
    xy = np.column_stack((x[chemin], y[chemin]))  # Coordonnées des villes dans l'ordre du chemin
    distance = np.sum(np.sqrt(np.sum((xy - np.roll(xy, -1, axis=0))**2, axis=1)))  
    return distance

def showmap(villes):
    plt.figure(figsize=(6, 6))
    for chemin in villes:
        chemin_complet = chemin + [chemin[0]]  # Revenir au point de départ
        plt.plot(x[chemin_complet], y[chemin_complet], marker="o", linestyle="-", label="Path")
    
    plt.scatter(x, y, color='red', marker='o', s=100, label="Cities")  # Afficher les villes en rouge
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Représentation des chemins des villes")
    plt.grid(True)
    plt.show() 
showmap(generate_population())
    