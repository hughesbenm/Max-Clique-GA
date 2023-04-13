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

# WRONG, NEED TO CONSIDER SUBGRAPH NOT FULL GRAPH
# def get_fitness(chromosome, G, k, experimental = True):
#   list = chromosome_to_node_list(chromosome)

#   S = G.subgraph(list)

#   fitness = S.size()
  
#   # Experimental Fitness
#   # Subtraction If Over
#   if (experimental):
#     max_fitness = (k * (k - 1)) / 2
#     if fitness > max_fitness:
#       fitness = 2 * max_fitness - fitness

#   return fitness

# def get_fitness(chromosome, G, k):
#   list = chromosome_to_node_list(chromosome)

#   S = G.subgraph(list)

#   fitness = 0

#   for edge in S.edges:
#     fitness += ((S.degree[edge[0]] - math.sqrt(k)) * (S.degree[edge[1]] - math.sqrt(k)))
#   return fitness

def get_fitness(chromosome, G, k):
  S = G.subgraph(chromosome_to_node_list(chromosome))
  nodes = S.number_of_nodes()
  edges = S.size()
  max_edges = nodes * (nodes - 1) / 2

  edge_fitness = 0
  if max_edges != 0:
    edge_fitness = edges / max_edges
  
  node_fitness = nodes / k
  if node_fitness > 1:
    node_fitness = 2 - node_fitness
  
  return edge_fitness + 2 * node_fitness

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

def random_single(chromosome, mutation_rate):
  if random.random() < mutation_rate:
    rand_index = random.randint(0, len(chromosome) - 1)
    chromosome[rand_index] = (chromosome[rand_index] + 3) % 2
  return chromosome

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

def SGA(nodes, k, edge_prob, pop_size, num_elites, mutation_rate, generations):
  G = gen_graph(nodes, k, edge_prob)

  pop = gen_population(pop_size, nodes)
  
  for clique_size in range(2, k + 2):
    print("looking for clique size ", clique_size)
    current_gen = 0

    while current_gen < generations:
      parents = rank_select(pop, G, clique_size, num_elites)

      children = uniform_cross(parents)

      # mutated_children = [random_single(i, mutation_rate) for i in children]
      mutated_children = [fit_single(i, mutation_rate, G, k) for i in children]

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

      if current_gen % 50 == 0:
        print("gen ", current_gen, " best fitness: ", get_fitness(elites[0], G, clique_size))
        # print(*elites[0])
        # print(*elites[1])
        # nx.draw(G.subgraph(chromosome_to_node_list(elites[0])), with_labels=True)
        # plt.show()

      has_clique = check_for_clique(elites[0], G, clique_size)
      if has_clique:
        print("Has Clique of ", clique_size)
        print("Took ", current_gen, " generations")
        break




random.seed()


NODES = random.randint(30, 150)
K = 5
EDGE_PROB = 0.4
POP_SIZE = 100
NUM_ELITES = 4
MUTATION_RATE = 0.15
GENERATIONS = 1000
RAND_K = random.randint(5, int(NODES / 4))

SGA(NODES, RAND_K, EDGE_PROB, POP_SIZE, NUM_ELITES, MUTATION_RATE, GENERATIONS)

print(RAND_K)

# G = gen_graph(NODES, K, EDGE_PROB)

# chromo = gen_chromosome(NODES)

# nx.draw(G.subgraph(chromosome_to_node_list(chromo)), with_labels=True)
# plt.show() 

# fit_single(chromo, 1, G, K)

# nx.draw(G.subgraph(chromosome_to_node_list(chromo)), with_labels=True)
# plt.show()