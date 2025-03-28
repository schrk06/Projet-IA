import random
import numpy as np
import matplotlib.pyplot as plt

# Nombre de villes
NUM_CITIES =random.randint(5, 50)

# Génération d'une matrice de distances symétrique
MATRIX = np.random.randint(10, 100, size=(NUM_CITIES, NUM_CITIES))
np.fill_diagonal(MATRIX, 0)
MATRIX = (MATRIX + MATRIX.T) // 2  # Rendre la matrice symétrique

# Génération de coordonnées fictives pour la visualisation
np.random.seed(42)
city_coords = np.random.rand(NUM_CITIES, 2) * 100

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

# Fonction pour afficher le chemin trouvé
def plot_route(route):
    plt.figure(figsize=(8, 6))
    for i in range(len(route) - 1):
        x1, y1 = city_coords[route[i]]
        x2, y2 = city_coords[route[i + 1]]
        plt.plot([x1, x2], [y1, y2], 'bo-')
        distance = MATRIX[route[i], route[i + 1]]
        plt.text((x1 + x2) / 2, (y1 + y2) / 2, str(distance), fontsize=10, color='blue')
    
    # Ajouter la dernière connexion pour fermer le cycle
    x1, y1 = city_coords[route[-1]]
    x2, y2 = city_coords[route[0]]
    plt.plot([x1, x2], [y1, y2], 'bo-')
    distance = MATRIX[route[-1], route[0]]
    plt.text((x1 + x2) / 2, (y1 + y2) / 2, str(distance), fontsize=10, color='blue')
    
    plt.scatter(city_coords[:, 0], city_coords[:, 1], c='red', marker='o')
    for i, (x, y) in enumerate(city_coords):
        plt.text(x, y, str(i), fontsize=12, ha='right')
    plt.title("Meilleur chemin trouvé avec distances")
    plt.show()

# Exécuter l'algorithme et afficher le meilleur chemin trouvé
best_route, best_distance = genetic_algorithm()
print(f"Meilleur chemin trouvé: {best_route}")
print(f"Distance totale: {best_distance:.2f}")
plot_route(best_route)
