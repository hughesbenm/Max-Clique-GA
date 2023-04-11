import copy
import networkx as nx 
import matplotlib.pyplot as plt 
import random
import numpy as np

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
def get_fitness(chromosome, nodes, G, k, experimental = False):
  list = chromosome_to_node_list(chromosome, nodes)

  S = G.subgraph(list)

  fitness = 0
  for i in list:
    fitness += S.degree[i]
  
  # Experimental Fitness
  # Subtraction If Over
  if (experimental):
    max_fitness = (k * (k - 1)) / 2
    if fitness > max_fitness:
      fitness = 2 * max_fitness - fitness

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

def chromosome_to_node_list(chromosome, nodes):
  list = []
  for i in range(nodes):
    if chromosome[i] == 1:
      list.append(i)
  return list

# Takes in list of chromosomes, returns list of randomly selected parents
def rank_select(pop, pop_size, nodes, G, k, elites):
  fitnesses = [get_fitness(i, nodes, G, k) for i in pop]
  ranked = [sorted(fitnesses).index(i) + 1 for i in fitnesses]
  sum_ranks = sum(ranked)
  probabilities = [ranked[i] / sum_ranks for i in range(pop_size)]
  parents = random.choices(pop, probabilities, k = pop_size - elites)
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


def SGA(nodes, k, edge_prob, pop_size, elites):
  G = gen_graph(nodes, k, edge_prob)
  pop = gen_population(pop_size, nodes)
  parents = rank_select(pop, pop_size, nodes, G, k, elites)
  children = uniform_cross(parents)




random.seed()

NODES = 5
K = 3
EDGE_PROB = 0.2
POP_SIZE = 50
ELITES = 2

SGA(NODES, K, EDGE_PROB, POP_SIZE, ELITES)


chromosome = gen_chromosome(NODES)

list = chromosome_to_node_list(chromosome, NODES)

# nx.draw(G, with_labels=True) 
# plt.show() 
# nx.draw(G.subgraph(list), with_labels=True)
# plt.show()

# print("experimental fitness: ",  get_fitness(chromosome, NODES, G, True, K))
# print("normal fitness: ", get_fitness(chromosome, NODES, G, False, K))