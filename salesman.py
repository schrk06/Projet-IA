import random
import numpy as np

# Nombre de villes
NUM_CITIES = 10

# Génération d'une matrice de distances symétrique
MATRIX = np.random.randint(10, 100, size=(NUM_CITIES, NUM_CITIES))
np.fill_diagonal(MATRIX, 0)
MATRIX = (MATRIX + MATRIX.T) // 2  # Rendre la matrice symétrique

# Calculer la fitness (distance totale du trajet)
def fitness(route):
    return sum(MATRIX[route[i], route[i + 1]] for i in range(len(route) - 1)) + MATRIX[route[-1], route[0]]

# Générer une population initiale
def generate_population(size):
    return [random.sample(range(NUM_CITIES), NUM_CITIES) for _ in range(size)]

# Sélection par tournoi
def select_parents(population, num_parents):
    return [min(random.sample(population, 5), key=fitness) for _ in range(num_parents)]

# Croisement (crossover)
def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [-1] * len(parent1)
    child[start:end] = parent1[start:end]
    remaining = [city for city in parent2 if city not in child]
    index = 0
    for i in range(len(child)):
        if child[i] == -1:
            child[i] = remaining[index]
            index += 1
    return child

# Mutation (échange aléatoire de deux villes)
def mutate(route, mutation_rate=0.1):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

# Algorithme génétique principal
def genetic_algorithm(pop_size=100, generations=500):
    population = generate_population(pop_size)
    for _ in range(generations):
        parents = select_parents(population, pop_size // 2)
        children = [mutate(crossover(random.choice(parents), random.choice(parents))) for _ in range(pop_size // 2)]
        population = parents + children
    best_route = min(population, key=fitness)
    return best_route, fitness(best_route)

# Exécuter l'algorithme et afficher le meilleur chemin trouvé
best_route, best_distance = genetic_algorithm()
print(f"Meilleur chemin trouve: {best_route}")
print(f"Distance totale: {best_distance:.2f}")
