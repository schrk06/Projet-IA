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
    

M = 50 #It represents the population's dimension
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



"""def fitness(population):   #si les distances sont proches les unes des autres, la fitness avoisinera 0, ce qui donnera une mauvaise selection
    total_dist = []
    for i in range (0, len(population)):
            total_dist.append(dist_max(population[i]))
            
    max_cost = max(total_dist)
    population_fitness = max_cost - total_dist
    population_fitness_tot = sum(population_fitness)      
    fit = population_fitness / population_fitness_tot
    return fit"""
def fitness(population):
    total_dist = np.array([dist_max(path) for path in population])
    population_fitness = 1 / total_dist
    population_fitness /= population_fitness.sum()
    return population_fitness



def selection(population, fit_prob, elitism_count=10):

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


def draw_all_edges(x, y):
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            plt.plot([x[i], x[j]], [y[i], y[j]], color='blue', linewidth=0.5)


def genetic_algorithm(population, mutation_prob=0.15, gen_num=200, elitism_count=10):
    best_distances = []
    best_path = None
    best_distance = float('inf')
    best_gen = 0

    plt.ion()
    plt.figure(figsize=(8, 6))

    for generation in range(gen_num):
        fit_prob = fitness(population)
        population = selection(population, fit_prob, elitism_count)
        elites = population[:elitism_count]
        next_generation = elites.copy()

        for i in range(elitism_count, len(population) - 1, 2):
            c1, c2 = crossover(population[i], population[i + 1])
            next_generation.extend([c1, c2])

        if len(population) % 2 != 0:
            next_generation.append(crossover(population[-1], population[elitism_count])[0])

        for i in range(elitism_count, len(next_generation)):
            if rd.random() < mutation_prob:
                next_generation[i] = mutation(next_generation[i])

        population = next_generation
        distances = [dist_max(p) for p in population]
        min_idx = np.argmin(distances)
        current_best = distances[min_idx]

        if current_best < best_distance:
            best_distance = current_best
            best_path = population[min_idx]
            best_gen = generation + 1

        if best_distances:
            best_distances.append(min(best_distances[-1], current_best))
        else:
            best_distances.append(current_best)

        # Affichage interactif
        plt.clf()
        draw_all_edges(x, y)
        best_current_path = population[min_idx] + [population[min_idx][0]]
        plt.plot(x[best_current_path], y[best_current_path], 'o-r', label=f"Génération {generation + 1} | Dist: {current_best:.4f}")
        plt.title("Évolution du meilleur chemin")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.pause(0.03)

    plt.ioff()
    print("\nMeilleur chemin trouvé :", best_path)
    print("Distance totale :", best_distance)
    print("Génération :", best_gen)

    # Visualisation finale
    plt.figure(figsize=(8, 6))
    draw_all_edges(x, y)
    final_path = best_path + [best_path[0]]
    plt.plot(x[final_path], y[final_path], 'o-r', label=f"Meilleur chemin | Dist: {best_distance:.4f} | Génération: {best_gen}")
    plt.title("Chemin optimal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Évolution de la distance
    plt.figure(figsize=(8, 4))
    plt.plot(best_distances, label="Distance du meilleur chemin")
    plt.title("Évolution du meilleur global")
    plt.xlabel("Générations")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_path


population = generate_population()
chemin_ideal = genetic_algorithm(population, mutation_prob=0.1 ,gen_num=300)
