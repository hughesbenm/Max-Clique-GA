import networkx as nx 
import matplotlib.pyplot as plt 
import random
import numpy as np

def graph_from_adj(adj):
  G = nx.Graph()
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
def get_fitness(chromosome, nodes, G, experimental, k):
  list = chromosome_to_node_list(chromosome, nodes)
  fitness = 0
  for i in list:
    fitness += G.degree[i]
  
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

def chromosome_to_node_list(chromosome, nodes):
  list = []
  for i in range(nodes):
    if chromosome[i] == 1:
      list.append(i)
  return list

random.seed()

NODES = 100
K = 2
EDGE_PROB = 0.45

G = gen_graph(NODES, K, EDGE_PROB)

chromosome = gen_chromosome(NODES)

list = chromosome_to_node_list(chromosome, NODES)

# nx.draw( G , with_labels=True) 
# plt.show() 
# nx.draw(G.subgraph(list), with_labels=True)
# plt.show()
print("experimental fitness: ",  get_fitness(chromosome, NODES, G, True, K))
print("normal fitness: ", get_fitness(chromosome, NODES, G, False, K))