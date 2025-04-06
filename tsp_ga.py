from math import factorial
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

M = 50  # Taille de la population

def dist_max(chemin, x, y):
    x_vals = x[chemin]
    y_vals = y[chemin]
    x_vals = np.append(x_vals, x_vals[0])
    y_vals = np.append(y_vals, y_vals[0])
    dx = np.diff(x_vals)
    dy = np.diff(y_vals)
    distance = np.sum(np.sqrt(dx**2 + dy**2))
    return distance

def fitness(population, x, y):
    total_dist = np.array([dist_max(path, x, y) for path in population])
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
            if r <= c:
                selected_population.append(population[i])
                break
    return selected_population

def crossover(p1, p2, N):
    cut = rd.randint(1, N-1)
    cross1 = p1[0:cut] + [ville for ville in p2 if ville not in p1[0:cut]]
    cross2 = p2[0:cut] + [ville for ville in p1 if ville not in p2[0:cut]]
    return cross1, cross2

def mutation(cross, N):
    i1, i2 = rd.sample(range(N), 2)
    cross[i1], cross[i2] = cross[i2], cross[i1]
    return cross

def draw_all_edges(x, y):
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            plt.plot([x[i], x[j]], [y[i], y[j]], color='blue', linewidth=0.5)

def generate_population(N):
    pop_set = set()
    while len(pop_set) < min(M, factorial(N)):
        chemin = tuple(rd.sample(range(N), N))
        pop_set.add(chemin)
    return [list(c) for c in pop_set]

def genetic_algorithm(x, y, N, mutation_prob=0.15, gen_num=200, elitism_count=10):
    x = np.array(x)
    y = np.array(y)
    population = generate_population(N)

    best_distances = []
    best_path = None
    best_distance = float('inf')
    best_gen = 0

    plt.ion()
    fig = plt.figure(2, figsize=(8, 6))
    for generation in range(gen_num):
        fit_prob = fitness(population, x, y)
        population = selection(population, fit_prob, elitism_count)
        elites = population[:elitism_count]
        next_generation = elites.copy()

        for i in range(elitism_count, len(population) - 1, 2):
            c1, c2 = crossover(population[i], population[i + 1], N)
            next_generation.extend([c1, c2])
        if len(population) % 2 != 0:
            next_generation.append(crossover(population[-1], population[elitism_count], N)[0])

        for i in range(elitism_count, len(next_generation)):
            if rd.random() < mutation_prob:
                next_generation[i] = mutation(next_generation[i], N)

        population = next_generation
        distances = [dist_max(p, x, y) for p in population]
        min_idx = np.argmin(distances)
        current_best = distances[min_idx]

        if current_best < best_distance:
            best_distance = current_best
            best_path = population[min_idx]
            best_gen = generation + 1

        best_distances.append(min(best_distances[-1], current_best) if best_distances else current_best)

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

    plt.figure(figsize=(8, 4))
    plt.plot(best_distances, label="Distance du meilleur chemin")
    plt.title("Évolution du meilleur global")
    plt.xlabel("Générations")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_path

def run():
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.2)
    plt.title("Choisissez un mode pour placer les villes")
    plt.axis([0, 1, 0, 1])

    textbox_ax = plt.axes([0.35, 0.05, 0.1, 0.05])
    text_box = TextBox(textbox_ax, 'N = ', initial="5")

    def start_random(event):
        try:
            N = int(text_box.text)
            if N < 4:
                print("N doit être ≥ 4")
                plt.title("N doit être ≥ 4")
                return
        except ValueError:
            print("Veuillez entrer un entier.")
            return
        
        # x = np.random.choice(np.linspace(0, 1, 10000), size=N, replace=False) (dans le cas où l'on veut éviter les doublons, mais ça ne sert à rien)
        x = np.random.rand(N)
        y = np.random.rand(N)
        plt.close()
        genetic_algorithm(x, y, N)

    def start_manual(event):
        try:
            N = int(text_box.text)
            if N < 4:
                print("N doit être ≥ 4")
                plt.title("N doit être ≥ 4")
                return
        except ValueError:
            print("Veuillez entrer un entier.")
            return
        plt.title(f"Cliquez {N} fois pour placer les villes")
        coords = plt.ginput(N, timeout=0)
        if len(coords) != N:
            print("Nombre de points incorrect.")
            return
        x, y = zip(*coords)
        x, y = np.array(x), np.array(y)
        plt.close()
        genetic_algorithm(x, y, N)

    ax_random = plt.axes([0.1, 0.05, 0.2, 0.075])
    btn_random = Button(ax_random, 'Aléatoire')
    btn_random.on_clicked(start_random)

    ax_manual = plt.axes([0.55, 0.05, 0.2, 0.075])
    btn_manual = Button(ax_manual, 'Manuel')
    btn_manual.on_clicked(start_manual)

    plt.show()
