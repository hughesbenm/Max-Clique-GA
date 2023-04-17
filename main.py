import copy
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
  fitness = edges / (nodes * (nodes - 1) / 2)
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

def fit_single(chromosome, mutation_rate, G, k):
  if random.random() < mutation_rate:
    present_nodes = chromosome_to_node_list(chromosome)
    S = G.subgraph(present_nodes)
    num_nodes = S.number_of_nodes()
    rand_miss = random_missing(chromosome)
    rand_pres = random_present(chromosome)
    if num_nodes <= k:
      chromosome[rand_miss] = 1
    elif num_nodes >= k:
      chromosome[rand_pres] = 0
  return chromosome

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

  if S.number_of_nodes() != k:
    return False
  if S.size() != ((k * (k - 1)) / 2):
    return False
  return True

def GA(nodes, k, edge_prob, pop_size, num_elites, mutation_rate, generations, tournament_alpha):
  G = gen_graph(nodes, k, edge_prob)

  pop = gen_population(pop_size, nodes)

  total_gens = 0
  
  for clique_size in range(3, nodes):
    print("looking for clique size ", clique_size)
    current_gen = 0

    while current_gen < generations:
      # parents = rank_select(pop, G, clique_size, num_elites)
      # parents = roulette_select(pop, G, clique_size, num_elites)
      parents = tournament_select(pop, G, clique_size, num_elites, tournament_alpha)

      # children = uniform_cross(parents)
      children = point_cross(parents)

      mutated_children = [random_single(i, mutation_rate) for i in children]
      # mutated_children = [fit_single(i, mutation_rate, G, k) for i in children]

      elites = find_elites(pop, num_elites, G, clique_size)

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

      if current_gen % 1 == 0:
        print("gen ", current_gen, " best fitness: ", get_fitness(elites[0], G, clique_size))
        # print(*elites[0])
        # print(*elites[1])
        nx.draw(G.subgraph(chromosome_to_node_list(elites[0])), with_labels=True)
        plt.show()

      has_clique = check_for_clique(elites[0], G, clique_size)
      if has_clique:
        total_gens += current_gen
        print("Has Clique of ", clique_size)
        print("Took ", current_gen, " generations")
        break
    if current_gen >= generations:
      print("Highest Clique Guranteed: ", k)
      print("Highest Clique Found: ", clique_size - 1)
      print("It took ", total_gens, " generations to get that clique")
      break

def SA(nodes, k, edge_prob, pop_size, num_elites, mutation_rate, generations, tournament_alpha, init_temp, alpha, beta):
  G = gen_graph(nodes, k, edge_prob)

  chromosome = gen_chromosome(nodes)
  temperature = init_temp

  iterations = 0

  while iterations < 100:
    old = get_fitness(chromosome, G, k)
    new_chromosome = random_single(chromosome, 1)
    new = get_fitness(new_chromosome, G, k)
    # print("diff: ", new - old)
    # print("quotient: ", (old - new) / temperature)
    print("e: ", math.exp((new - old) / temperature))
    if new > old:
      chromosome = copy.deepcopy(new_chromosome)
      print("better")
    elif random.random() < math.exp((new - old) / temperature):
      chromosome = copy.deepcopy(new_chromosome)
      print("random")
    else:
      print("same")
    temperature *= alpha
    print("fit: ", get_fitness(chromosome, G, k))
    print("temp: ", temperature)
    # print(get_fitness(new_chromosome, G, k))
    iterations += 1

  return False



random.seed()


NODES = 100
K = 17
EDGE_PROB = 0.1
POP_SIZE = 50
NUM_ELITES = 2
MUTATION_RATE = 0.15
GENERATIONS = 1000
TOURNAMENT_ALPHA = 0.05
INITIAL_TEMPERATURE = 0.01
ANNEALING_ALPHA = 0.98
ANNEALING_BETA = 1.01

# GA(NODES, K, EDGE_PROB, POP_SIZE, NUM_ELITES, MUTATION_RATE, GENERATIONS, TOURNAMENT_ALPHA)
SA(NODES, K, EDGE_PROB, POP_SIZE, NUM_ELITES, MUTATION_RATE, GENERATIONS, TOURNAMENT_ALPHA, INITIAL_TEMPERATURE, ANNEALING_ALPHA, ANNEALING_BETA)

# G = gen_graph(NODES, K, EDGE_PROB)

# chromo = gen_chromosome(NODES)

# nx.draw(G.subgraph(chromosome_to_node_list(chromo)), with_labels=True)
# plt.show() 

# fit_single(chromo, 1, G, K)

# nx.draw(G.subgraph(chromosome_to_node_list(chromo)), with_labels=True)
# plt.show()