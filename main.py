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
  # S = G.subgraph(chromosome_to_node_list(chromosome))
  # nodes = S.number_of_nodes()
  # edges = S.size()
  # max_edges = nodes * (nodes - 1) / 2

  # edge_fitness = 0
  # if max_edges != 0:
  #   edge_fitness = edges / max_edges
  
  # node_fitness = nodes / k
  # if node_fitness > 1:
  #   node_fitness = 2 - node_fitness
  
  # return edge_fitness + 2 * node_fitness

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

  # S = G.subgraph(chromosome_to_node_list(chromosome))
  # nodes = S.number_of_nodes()
  # edges = S.size()
  # if nodes == 0:
  #   return 0
  # fitness = edges / (nodes * (nodes - 1) / 2)
  # return fitness

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

# Takes in list of chromosomes, returns list of randomly selected parents
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
    # nx.draw(S, with_labels=True)
    # plt.show()
    return nodes
  return -1

def GA(G, nodes, k, edge_prob, pop_size, num_elites, mutation_rate, generations, tournament_alpha):
  print("Starting Generational Genetic Algorithm")
  start_time = time.perf_counter()
  end_time = 0
  G = gen_graph(nodes, k, edge_prob)

  pop = gen_population(pop_size, nodes)

  current_gen = 0
  current_clique = 3
  while current_gen < generations:
    # parents = rank_select(pop, G, clique_size, num_elites)
    # parents = roulette_select(pop, G, clique_size, num_elites)
    parents = tournament_select(pop, G, current_clique, num_elites, tournament_alpha)

    # children = uniform_cross(parents)
    children = point_cross(parents)

    mutated_children = [random_multi(i, mutation_rate) for i in children]
    # mutated_children = [fit_single(i, mutation_rate, G, k) for i in children]

    elites = find_elites(pop, num_elites, G, current_clique)

    next_pop = copy.deepcopy(pop)
    for i in range(pop_size - num_elites):
      next_pop[i] = mutated_children[i]
    for i in range(num_elites):
      next_pop[pop_size - i - 1] = elites[i]

    pop = next_pop
    
    current_gen += 1

    # nx.draw(G.subgraph(chromosome_to_node_list(elites[0])), with_labels=True)
    # plt.show()
    # nx.draw(G.subgraph(chromosome_to_node_list(fit_single(elites[0], 1, G, k))), with_labels=True)
    # plt.show()

    found_clique = check_for_clique(elites[0], G, current_clique)
    if found_clique >= current_clique:
        print("Found clique of ", found_clique, " after ", current_gen, " generations")
        current_clique = found_clique + 1
        current_gen = 0
        end_time = time.perf_counter()
    
  print("Highest Clique Guranteed: ", k)
  print("Highest Clique Found: ", current_clique - 1)
  print("Took ", end_time - start_time, " seconds")
  return current_clique - 1

def SA(G, nodes, k, edge_prob, generations, init_temp, iterations, alpha, beta):
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
        break
      
      curr_iteration *= beta

    current_gen += 1
    # print("Current temp: ", temperature, "Current fitness: ", get_fitness(chromosome, G, k), "Current generation: ", current_gen)
    temperature *= alpha

  print("Largest Clique Found: ", current_clique - 1)
  print("Largest Clique Guranteed: ", k)
  print("Took ", end_time - start_time, " seconds")
  return current_clique - 1

def HC(G, nodes, k, edge_prob, pop_size, num_elites, mutation_rate, generations, tournament_alpha):
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
      new_chromosome = random_multi(chromosome, 1)
      new = get_fitness(new_chromosome, G, current_clique)

      if new > old:
        chromosome = copy.deepcopy(new_chromosome)

      found_clique = check_for_clique(chromosome, G, current_clique)

      if found_clique >= current_clique:
        print("Found clique of ", found_clique, " after ", current_gen, " generations")
        # print("Took ", current_gen, " generations")
        current_clique = found_clique + 1
        end_time = time.perf_counter()
        break
      
      # if current_gen % 50 == 0:
        # print("Current fitness: ", get_fitness(chromosome, G, current_clique), "Current generation: ", current_gen)
      current_gen += 1
  
  print("Largest Clique Found: ", current_clique - 1)
  print("Largest Clique Guranteed: ", k)
  print("Took ", end_time - start_time, " seconds")
  return current_clique - 1



random.seed()


NODES = 100
K = 17
EDGE_PROB = 0.25
POP_SIZE = 50
NUM_ELITES = 2
MUTATION_RATE = 0.15
GA_GENERATIONS = 200
HC_GENERATIONS = 10000
TOURNAMENT_ALPHA = 0.05

SA_GENERATIONS = 100
INITIAL_TEMPERATURE = 5
ANNEALING_ALPHA = 0.85
ANNEALING_BETA = 1.05
ITERATIONS = 100

G = gen_graph(NODES, K, EDGE_PROB)

gene_clique = GA(G, NODES, K, EDGE_PROB, POP_SIZE, NUM_ELITES, MUTATION_RATE, GA_GENERATIONS, TOURNAMENT_ALPHA)
simu_clique = SA(G, NODES, K, EDGE_PROB, SA_GENERATIONS, INITIAL_TEMPERATURE, ITERATIONS, ANNEALING_ALPHA, ANNEALING_BETA)
hill_clique = HC(G, NODES, K, EDGE_PROB, POP_SIZE, NUM_ELITES, MUTATION_RATE, HC_GENERATIONS, TOURNAMENT_ALPHA)
