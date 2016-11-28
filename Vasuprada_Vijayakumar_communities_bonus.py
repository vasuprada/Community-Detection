import matplotlib.pyplot as plt
import community
import networkx as nx
import operator
import random
import six
import copy
import sys
from matplotlib import colors

def girvan_newman(G):
    original_graph = G.copy()
    partitions = find_best_community(original_graph)
    return partitions

def assign_community(connected_components):
    community_label = 0
    partition_dict = {}
    for community in connected_components:
        community_label += 1
        for point in community:
            partition_dict[point] = community_label
    return partition_dict

def run_bfs(input_graph,source):
    reachable_nodes = []
    predecessor_dict = {}
    dict_cost = {}
    number_of_shortest_paths = {}

    # Initialize for new state of the graph
    for node in input_graph:
        predecessor_dict[node] = []
        number_of_shortest_paths[node] = 0.0

    #Initialize Source
    number_of_shortest_paths[source] = 1.0
    dict_cost[source] = 0
    #Initially queue has only source - LIFO ORDER
    queue = [source]

    # Perform BFS algorithm
    while queue:
        current_node = queue.pop()
        # Every visited node is reachable, add to list
        reachable_nodes.append(current_node)
        path_length = dict_cost[current_node]
        num_short_paths = number_of_shortest_paths[current_node]

        for adjacent_node in input_graph[current_node]:
            # Increment the path length by 1 for every new edge traversed in the path
            if adjacent_node not in dict_cost:
                dict_cost[adjacent_node] = path_length + 1
                queue.append(adjacent_node)

            if dict_cost[adjacent_node] == path_length + 1:
                number_of_shortest_paths[adjacent_node] += num_short_paths
                predecessor_dict[adjacent_node].append(current_node)

    # Return the list of reachable nodes,ancestors dictionary and total number of shortest paths to all other nodes
    return reachable_nodes,predecessor_dict,number_of_shortest_paths

def calculate_betweeness_value(betweeness_dict,reachable_nodes,predecessor_dict,number_of_shortest_paths,source_node):

    contribution_values = {}
    # Maintain dictionary for storing intermediate values of reachable nodes
    for node in reachable_nodes:
        contribution_values[node] = 0.0

    # Iterate through the reachable nodes list for a given source node
    while reachable_nodes:
        current_node = reachable_nodes.pop()
        # Using the formula given in slides
        fraction = (contribution_values[current_node] + 1.0) / number_of_shortest_paths[current_node]

        for parent in predecessor_dict[current_node]:
            # If there is more than one shortest path to a node, then use the fractional contribution
            value_passed = number_of_shortest_paths[parent] * fraction
            # Add to the dictionary if the edge does not exist 
            if (parent,current_node) not in betweeness_dict:
                betweeness_dict[(current_node,parent)] += value_passed
            else:
                betweeness_dict[(parent,current_node)] += value_passed

            # Summation over all values from edges below
            contribution_values[parent] += value_passed

        # Update the betweeness dictionary
        if current_node != source_node:
            betweeness_dict[current_node] += contribution_values[current_node]

    return betweeness_dict


def own_betweeness(input_graph):

    betweenness_dict = {}
    # Initializing betweeness values for nodes and edges for current state of graph
    for node in input_graph.nodes():
        betweenness_dict[node] = 0.0
    for edge in input_graph.edges():
        betweenness_dict[edge] = 0.0

    nodelist = input_graph.nodes()

    # Calculate betweeness by assigning every node as source
    for s in nodelist:
        reachable_nodes,predecessor_dict,number_of_shortest_paths = run_bfs(input_graph,s)
        betweenness_dict = calculate_betweeness_value(betweenness_dict,reachable_nodes,predecessor_dict,number_of_shortest_paths,s)

    # Remove betweeness values for nodes - Only need edges
    for s in nodelist:
        del betweenness_dict[s]

    # Normalize the betweeness values - since shortest paths will be counted twice
    for edge,value in betweenness_dict.iteritems():
        new_value = float(value) / 2.0
        betweenness_dict[edge] = new_value

    return betweenness_dict


def find_best_community(input_graph):
    max = -1
    modularity = -1
    count = 1
    subgraphs = {}
    communities=[]
    betweeness_dict = {}

    # Till no more edges left in the graph
    while count < nx.number_of_nodes(input_graph):
        connected_components_list = []

        # Get the remaining connected components after removal of edge
        connected_components = nx.connected_components(input_graph)

        # Add generated sets from graph object to a list
        for i in connected_components:
            connected_components_list.append(i)

        # Cluster the communities and assign labels
        subgraphs = assign_community(connected_components_list)

        # Find Modularity of the connected subgraphs
        modularity = community.modularity(subgraphs, input_graph)

        # Change the modularity value if it increases
        if modularity > max:
            max = modularity
            # Assign the new communities every iteration
            communities = copy.deepcopy(connected_components_list)

        count += 1

        # ----------- LIBRARY FUNCTION------------#
        # betweeness_dict = nx.edge_betweenness(input_graph)
        # ----------------------------------------#

        #------------  OWN FUNCTION -------------#
        betweeness_dict = own_betweeness(input_graph)
        #----------------------------------------#
        betweeness_items = betweeness_dict.items()

        # Get Edge with Maximum betweeness by sorting
        betweeness_items.sort(key=lambda x:x[1],reverse = True)
        edge_max_betweeness_value = betweeness_items[0][1]

        for edge in betweeness_items:
            # Removing edge with maximum betweeness
            if edge[1] == edge_max_betweeness_value:
                input_graph.remove_edge(edge[0][0],edge[0][1])

        if input_graph.number_of_edges() == 0:
            return communities

    return communities

# Get All possible colors from matplotlib
def get_colors():
    colors_ = list(six.iteritems(colors.cnames))
    for name, rgb in six.iteritems(colors.ColorConverter.colors):
        hex_ = colors.rgb2hex(rgb)
        colors_.append((name, hex_))
    return colors_

def draw_graph(communities,graph,image):
    pos = nx.spring_layout(graph)
    nx.draw_networkx_labels(graph, pos)
    k = 0
    for community in communities:
        nx.draw_networkx_nodes(graph, pos, community, node_size=300, node_color = get_colors()[k], with_labels = True)
        k += 1
    nx.draw_networkx_edges(graph, pos, alpha=0.5)
    plt.savefig(image)
    plt.show()



def main(argv):
    f = open(argv[1],'r')
    data = f.readlines()
    G = nx.Graph()
    graphdict = {}
    nodelist1 = []
    edgelist  = []
    #NODES
    for edge in data:
        nodes = edge.strip().split(" ")
        if nodes[0] not in nodelist1:
            nodelist1.append(nodes[0])
            graphdict[nodes[0]] = []
        if nodes[1] not in nodelist1:
            nodelist1.append(nodes[1])
            graphdict[nodes[1]] = []

    nodelist1.sort()
    # Adding the nodes to the Graph
    G.add_nodes_from(nodelist1)

    #EDGES
    for edge in data:
        nodes = edge.strip().split(" ")
        edgelist.append((nodes[0],nodes[1]))
        # Adding Edges to the Graph
        G.add_edge(nodes[0],nodes[1])
        graphdict[nodes[0]].append(nodes[1])

    final_partitions = girvan_newman(G)

    if final_partitions == []:

        pos = nx.spring_layout(G)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_nodes(G, pos, nodelist1, node_size=300, node_color='r', with_labels=True)
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        plt.savefig(argv[2])
        plt.show()

        print "%s" % sorted(map(int,nodelist1))

    else:
        final_communities = []
        draw_communities = []
        number_of_communities = 0

        for partition in final_partitions:
            number_of_communities += 1
            draw_communities.append(list(partition))
            final_communities.append(sorted(map(int,list(partition))))

        # Print Communities to Console

        for community in sorted(final_communities):
            print "%s" % community

        draw_graph(sorted(draw_communities),G,argv[2])

if __name__ == '__main__':
    main(sys.argv)