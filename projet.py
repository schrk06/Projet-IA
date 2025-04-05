from math import factorial
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations



goodint = False
while not goodint:
    N  = int(input("Enter the number of cities: "))
    try:
        if(N > 3):
            goodint = True
        else:
            print("That's not a valid int, try again !")    
    except ValueError:
        print("That's not an int!, try again !")
    

M = 15 #It represents the population's dimension
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

def dist_max(chemin):
    """ Calcule la distance totale parcourue pour un chemin donné. """
    xy = np.column_stack((x[chemin], y[chemin]))  # Coordonnées des villes dans l'ordre du chemin
    distance = np.sum(np.sqrt(np.sum((xy - np.roll(xy, -1, axis=0))**2, axis=1)))  
    return distance



def fitness(population):
    total_dist = []
    for i in range (0, len(population)):
            total_dist.append(dist_max(population[i]))
            
    max_cost = max(total_dist)
    population_fitness = max_cost - total_dist
    population_fitness_tot = sum(population_fitness)
    fit = population_fitness / population_fitness_tot
    return fit



def selection(population, fit_prob, elitism_count =1):
    elite_indices = np.argsort(fit_prob)[-elitism_count:][::-1]
    elites = [population[i] for i in elite_indices]
    selected_population = elites.copy()
    fit_cum = np.cumsum(fit_prob)
    for _ in range(len(population)-elitism_count):
        r = rd.uniform(0,1)
        for i, c in enumerate(fit_cum):
            if (r<=c):
                selected_population.append(population[i])
                break
    return selected_population


def crossover(p1, p2):
    cut = rd.randint(1,N-1)
    cross1 = p1[0:cut]
    cross1 += [ville for ville in p2 if ville not in cross1]
    cross2 = p2[0:cut]
    cross2+= [ville for ville in p1 if ville not in cross2]
    return cross1, cross2

def mutation(cross):
    index1=0
    index2=0
    while (index1==index2):
        index1=rd.randint(0,N-1)
        index2=rd.randint(0,N-1)
    temp= cross[index1]
    cross[index1]=cross[index2]
    cross[index2]=temp
    return cross 

""" def genetic_algorithm(mutation_prob=0.1) : 
    
    population = generate_population()
    for generation in range(M):
        fit_prob= population.fitness()
        selected_population = population(population, fit_prob) 
        new_population =[]
        while len(new_population) < N :
            parent1, parent2 = rd.sample(selected_population, 2)
            offspring1, offspring2 = crossover(parent1, parent2)
            p = rd.uniform(0, 1)
            if p < mutation_prb :
                offspring1 = mutation(offspring1)
            if p < mutation_prb :
                offspring2 = mutation(offspring2)
            new_population.append(offspring1)
            new_population.append(offspring2)
        population = new_population   
    best_index = np.argmax(fit_prob)
    return population[best_index], fit_prob[best_index] """  #wsh rayk
    



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

popo =generate_population()
pipi = fitness(popo)
print(pipi)
print(selection(popo,pipi, 4))
