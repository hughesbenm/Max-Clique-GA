import copy
import time
import networkx as nx 
import matplotlib.pyplot as plt 
import random
import numpy as np
import math

def graph_from_adj(adj):
  G = nx.Graph()
  for i in range(len(adj)):
    G.add_node(i)
  
  for i in range(len(adj)): 
    for j in range(len(adj[0])):
      if adj[i][j] == 1: 
          G.add_edge(i,j) 
  return G

def gen_graph(nodes, clique_size, edge_prob):
  G = nx.Graph()

  adj = np.zeros((nodes, nodes))
    
  sample = random.sample(range(0, nodes - 1), clique_size)

  for i in sample:
    for j in sample:
      if i != j:
        adj[i][j] = 1

  for i in range(nodes):
    for j in sample:
      if i != j:
        if random.random() < edge_prob:
          adj[i][j] = 1

  G = graph_from_adj(adj)

  return G

def get_fitness(chromosome, G, k):
  S = G.subgraph(chromosome_to_node_list(chromosome))
  nodes = S.number_of_nodes()
  edges = S.size()
  if nodes == 0:
    return 0
  if (nodes * (nodes - 1) / 2) != 0:
    completeness = edges / (nodes * (nodes - 1) / 2)
  else:
    completeness = 0
  enough_nodes = nodes / k
  if enough_nodes > 1:
    enough_nodes = 1
  fitness = (completeness + enough_nodes) / 2
  return fitness

def gen_chromosome(nodes):
  chromosome = np.zeros((nodes, 1))
  for i in range(nodes):
    if random.random() < 0.5:
      chromosome[i] = 1
  return chromosome

def gen_population(pop_size, nodes):
  pop = np.empty((pop_size, nodes, 1))

  for i in range(pop_size):
    pop[i] = gen_chromosome(nodes)
  return pop

def chromosome_to_node_list(chromosome):
  list = []
  for i in range(len(chromosome)):
    if chromosome[i] == 1:
      list.append(i)
  return list

def rank_select(pop, G, k, elites):
  fitnesses = [get_fitness(i, G, k) for i in pop]
  ranked = [sorted(fitnesses).index(i) + 1 for i in fitnesses]
  sum_ranks = sum(ranked)
  probabilities = [ranked[i] / sum_ranks for i in range(len(pop))]
  parents = random.choices(pop, probabilities, k = len(pop) - elites)
  return parents

def roulette_select(pop, G, k, elites):
  fitnesses = [get_fitness(i, G, k) for i in pop]
  sum_fitnesses = sum(fitnesses)
  probabilities = [fitnesses[i] / sum_fitnesses for i in range(len(pop))]
  parents = random.choices(pop, probabilities, k = len(pop) - elites)
  return parents

def tournament_select(pop, G, k, elites, alpha):
  parents = []
  for i in range(len(pop - elites)):
    first = random.randint(0, len(pop) - 1)
    second = first
    while second == first:
      second = random.randint(0, len(pop) - 1)
    if random.random() > alpha:
      if get_fitness(pop[first], G, k) > get_fitness(pop[second], G, k):
        parents.append(pop[first])
      else:
        parents.append(pop[second])
    else:
      if get_fitness(pop[first], G, k) > get_fitness(pop[second], G, k):
        parents.append(pop[second])
      else:
        parents.append(pop[first])
  return parents

def uniform_cross(parents):
  children = copy.deepcopy(parents)
  for i in range(int(len(parents)/2)):
    for j in range(len(parents[0])):
      rand = random.random()
      if rand < 0.5:
        children[2 * i][j] = parents[2 * i][j]
        children[2 * i + 1][j] = parents[2 * i + 1][j]
      else:
        children[2 * i + 1][j] = parents[2 * i][j]
        children[2 * i][j] = parents[2 * i + 1][j]
  return children

def point_cross(parents):
  children = copy.deepcopy(parents)

  for i in range(int(len(parents)/2)):
    for j in range(len(parents[0])):
      pivot = random.randint(0, len(parents[0] - 1))
      if j < pivot:
        children[2 * i][j] = parents[2 * i][j]
        children[2 * i + 1][j] = parents[2 * i + 1][j]
      else:
        children[2 * i][j] = parents[2 * i + 1][j]
        children[2 * i + 1][j] = parents[2 * i][j]
  return children

def random_single(chromosome, mutation_rate):
  new_chromosome = copy.deepcopy(chromosome)
  if random.random() < mutation_rate:
    rand_index = random.randint(0, len(new_chromosome) - 1)
    new_chromosome[rand_index] = (new_chromosome[rand_index] + 3) % 2
  return new_chromosome

def random_multi(chromosome, mutation_rate):
  new_chromosome = copy.deepcopy(chromosome)
  for i in range(random.randint(1, 5)):
    if random.random() < mutation_rate:
      rand_index = random.randint(0, len(new_chromosome) - 1)
      new_chromosome[rand_index] = (new_chromosome[rand_index] + 3) % 2
  return new_chromosome

def random_present(chromosome):
  present_nodes = chromosome_to_node_list(chromosome)
  if len(present_nodes) < 1:
    return 0
  return present_nodes[random.randint(0, len(present_nodes) - 1)]

def random_missing(chromosome):
  missing_nodes = []
  for i in range(len(chromosome)):
    if chromosome[i] == 0:
      missing_nodes.append(i)
  if len(missing_nodes) < 1:
    return 0
  return missing_nodes[random.randint(0, len(missing_nodes) - 1)]

def find_elites(pop, num_elites, G, k):
  fitnesses = [get_fitness(i, G, k) for i in pop]
  ranked = [sorted(fitnesses).index(i) + 1 for i in fitnesses]
  ranked_sorted = sorted(ranked, reverse=True)
  elites = []
  for i in range(num_elites):
    elites.append(pop[ranked.index(ranked_sorted[i])])
  return elites

def check_for_clique(chromosome, G, k):
  list = chromosome_to_node_list(chromosome)
  S = G.subgraph(list)

  nodes = S.number_of_nodes()
  if S.size() == ((nodes * (nodes - 1)) / 2) and nodes >= k:
    return nodes
  return -1

def GA(G, nodes, k, edge_prob, pop_size, num_elites, mutation_rate, generations, tournament_alpha, select, cross, mutate):
  print("Starting Generational Genetic Algorithm")
  start_time = time.perf_counter()
  end_time = 0
  G = gen_graph(nodes, k, edge_prob)

  pop = gen_population(pop_size, nodes)

  current_gen = 0
  current_clique = 3
  while current_gen < generations:
    if select == TOURNEY:
      parents = tournament_select(pop, G, current_clique, num_elites, tournament_alpha)
    elif select == RANK:
      parents = rank_select(pop, G, current_clique, num_elites)
    elif select == ROULETTE:
      parents = roulette_select(pop, G, current_clique, num_elites)
      
    if cross == POINT:
      children = point_cross(parents)
    elif cross == UNIFORM:
      children = uniform_cross(parents)

    if mutate == MULTI:
      mutated_children = [random_multi(i, mutation_rate) for i in children]
    elif mutate == SINGLE:
      mutated_children = [random_single(i, mutation_rate) for i in children]

    elites = find_elites(pop, num_elites, G, current_clique)

    next_pop = copy.deepcopy(pop)
    for i in range(pop_size - num_elites):
      next_pop[i] = mutated_children[i]
    for i in range(num_elites):
      next_pop[pop_size - i - 1] = elites[i]

    pop = next_pop
    
    current_gen += 1

    found_clique = check_for_clique(elites[0], G, current_clique)
    if found_clique >= current_clique:
        print("Found clique of ", found_clique, " after ", current_gen, " generations")
        current_clique = found_clique + 1
        current_gen = 0
        end_time = time.perf_counter()
        if found_clique == k:
          break
    
  print("Highest Clique Guranteed: ", k)
  print("Highest Clique Found: ", current_clique - 1)
  print("Took ", end_time - start_time, " seconds")
  time_diff  = end_time - start_time
  return current_clique - 1, time_diff

def SA(G, nodes, k, edge_prob, generations, init_temp, iterations, alpha, beta, mutate):
  print("Starting Simulated Annealing Algorithm")
  start_time = time.perf_counter()
  end_time = 0

  chromosome = gen_chromosome(nodes)
  temperature = init_temp
  current_clique = 3
  e = 0
  current_gen = 0
  while current_gen < generations:
    curr_iteration = 1
    while curr_iteration < iterations:
      old = get_fitness(chromosome, G, current_clique)
      if mutate == SINGLE:
        new_chromosome = random_single(chromosome, 1)
      elif mutate == MULTI:
        new_chromosome = random_multi(chromosome, 1)
      new = get_fitness(new_chromosome, G, current_clique)
      quotient = (new - old) / temperature
      if quotient > 200 or quotient < -200:
        e = 0
      else:
        e = math.exp((new - old) / temperature)

      if new > old:
        chromosome = copy.deepcopy(new_chromosome)
      elif random.random() < e:
        chromosome = copy.deepcopy(new_chromosome)

      found_clique = check_for_clique(chromosome, G, current_clique)

      if found_clique >= current_clique:
        print("Found clique of ", found_clique, " after ", current_gen, " generations")
        current_clique = found_clique + 1
        current_gen = 0
        end_time = time.perf_counter()
        temperature = init_temp
        break
      
      curr_iteration *= beta

    current_gen += 1
    temperature *= alpha

  print("Largest Clique Found: ", current_clique - 1)
  print("Largest Clique Guranteed: ", k)
  print("Took ", end_time - start_time, " seconds")
  time_diff  = end_time - start_time
  return current_clique - 1, time_diff

def HC(G, nodes, k, edge_prob, pop_size, num_elites, mutation_rate, generations, tournament_alpha, mutate):
  print("Starting Hill CLimbing Algorithm")
  start_time = time.perf_counter()
  end_time = 0
  

  chromosome = gen_chromosome(nodes)
  current_clique = 3
  current_gen = 0
  found_clique = -1
  while current_gen < generations:
    current_gen = 0

    while current_gen < generations:
      old = get_fitness(chromosome, G, current_clique)
      if mutate == SINGLE:
        new_chromosome = random_single(chromosome, 1)
      elif mutate == MULTI:
        new_chromosome = random_multi(chromosome, 1)
      new = get_fitness(new_chromosome, G, current_clique)

      if new > old:
        chromosome = copy.deepcopy(new_chromosome)

      found_clique = check_for_clique(chromosome, G, current_clique)

      if found_clique >= current_clique:
        print("Found clique of ", found_clique, " after ", current_gen, " generations")
        current_clique = found_clique + 1
        end_time = time.perf_counter()
        break

      current_gen += 1
  
  print("Largest Clique Found: ", current_clique - 1)
  print("Largest Clique Guranteed: ", k)
  print("Took ", end_time - start_time, " seconds")
  time_diff  = end_time - start_time
  return current_clique - 1, time_diff



random.seed()

TOURNEY = 0
RANK = 1
ROULETTE = 2

POINT = 0
UNIFORM = 1

SINGLE = 0
MULTI = 1

NODES = 200
K = 40
EDGE_PROB = 0.75

POP_SIZE = 50
NUM_ELITES = 2

MUTATION_RATE = 0.15
GA_GENERATIONS = 500
SA_GENERATIONS = 100
HC_GENERATIONS = 10000
TOURNAMENT_ALPHA = 0.05

INITIAL_TEMPERATURE = 5
ANNEALING_ALPHA = 0.85
ANNEALING_BETA = 1.05
ITERATIONS = 100

G = gen_graph(NODES, K, EDGE_PROB)

print("\nTotal Nodes: ", NODES)
print("Guranteed Clique Size: ", K)
print("Edge Probabilty: ", EDGE_PROB)
print("\nNumber of Elites: ", NUM_ELITES)
print("Mutation Rate: ", MUTATION_RATE)
print("Population Size: ", POP_SIZE)
print("GA Generations: ", GA_GENERATIONS)
print("SA Generations: ", SA_GENERATIONS)
print("HC Generations: ", HC_GENERATIONS)
print("Alpha: ", ANNEALING_ALPHA)
print("Beta: ", ANNEALING_BETA)

gene_results = GA(G, NODES, K, EDGE_PROB, POP_SIZE, NUM_ELITES, MUTATION_RATE, GA_GENERATIONS, TOURNAMENT_ALPHA, RANK, UNIFORM, SINGLE)

print("GA Found Clique of Size ", gene_results[0], " and took ", gene_results[1], " seconds")

simu_results = SA(G, NODES, K, EDGE_PROB, SA_GENERATIONS, INITIAL_TEMPERATURE, ITERATIONS, ANNEALING_ALPHA, ANNEALING_BETA, SINGLE)

print("SA Found Clique of Size ", simu_results[0], " and took ", simu_results[1], " seconds")

hill_results = HC(G, NODES, K, EDGE_PROB, POP_SIZE, NUM_ELITES, MUTATION_RATE, HC_GENERATIONS, TOURNAMENT_ALPHA, SINGLE)

print("HC Found Clique of Size ", hill_results[0], " and took ", hill_results[1], " seconds")
