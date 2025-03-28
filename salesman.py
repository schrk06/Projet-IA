import random
import numpy as np

# Générer des villes avec des coordonnées aléatoires
NUM_CITIES = 10
CITIES = {i: (random.randint(0, 100), random.randint(0, 100)) for i in range(NUM_CITIES)}

# Calculer la distance entre deux villes
def distance(city1, city2):
    x1, y1 = CITIES[city1]
    x2, y2 = CITIES[city2]
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Calculer la fitness (distance totale du trajet)
def fitness(route):
    return sum(distance(route[i], route[i + 1]) for i in range(len(route) - 1)) + distance(route[-1], route[0])

# Générer une population initiale
def generate_population(size):
    return [random.sample(CITIES.keys(), len(CITIES)) for _ in range(size)]

# Sélection par tournoi
def select_parents(population, num_parents):
    selected = []
    for _ in range(num_parents):
        candidates = random.sample(population, 5)
        selected.append(min(candidates, key=fitness))
    return selected

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
        children = [crossover(random.choice(parents), random.choice(parents)) for _ in range(pop_size // 2)]
        children = [mutate(child) for child in children]
        population = parents + children
    best_route = min(population, key=fitness)
    return best_route, fitness(best_route)

# Exécuter l'algorithme et afficher le meilleur chemin trouvé
best_route, best_distance = genetic_algorithm()
print(f"Meilleur chemin trouvé: {best_route}")
print(f"Distance totale: {best_distance:.2f}")
