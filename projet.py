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


def generate_population(max_attempts=5):
    """Génère M chemins aléatoires uniques sans calculer toutes les permutations."""
    max_permutations = min(M, factorial(N))  # Empêche d'aller au-delà de N!
    attempts = 0

    while attempts < max_attempts:
        pop_set = set()
        while len(pop_set) < max_permutations:
            chemin = tuple(rd.sample(range(N), N))
            pop_set.add(chemin)

        population = [list(chemin) for chemin in pop_set]
        fitness_scores = fitness(population)
        sum_fit = sum(fitness_scores)

        if sum_fit > 0:
            return population
        else:
            attempts += 1
            print(f"Attempt {attempts}: all paths had the same fitness. Retrying...")

    # Si on arrive ici, toutes les tentatives ont échoué
    raise Exception(f"All generated paths have the same global distance. Last attempted population: {population}")

    

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

def genetic_algorithm(population, mutation_prob=0.1): 
    for generation in range(M):
        fit_prob = fitness(population)
        selected_population = selection(population, fit_prob)  # FIXED: call 'selection', not 'population(...)'

        new_population = []

        # Generate new population (maintain size)
        while len(new_population) < len(population):
            # Safety check to ensure enough individuals to sample from
            if len(selected_population) < 2:
                selected_population = generate_population()

            parent1, parent2 = rd.sample(selected_population, 2)
            offspring1, offspring2 = crossover(parent1, parent2)

            # Mutate with probability
            if rd.random() < mutation_prob:
                offspring1 = mutation(offspring1)
            if rd.random() < mutation_prob:
                offspring2 = mutation(offspring2)

            new_population.append(offspring1)
            if len(new_population) < len(population):
                new_population.append(offspring2)

        population = new_population

    fit_prob = fitness(population)  # Recalculate fitness for final population
    best_index = np.argmax(fit_prob)
    return population[best_index], fit_prob[best_index], dist_max(population[best_index])
    



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

"""popo =generate_population()
pipi = fitness(popo)
print(pipi)
print(selection(popo,pipi, 4))"""

pop = generate_population()
best = genetic_algorithm(pop)
print(pop)
print(best[0])
print(best[1])
print(best[2])
