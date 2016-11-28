import matplotlib.pyplot as plt
import community
import sys
import networkx as nx
import operator
import random
import six
import copy
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

        # Get edge_betweeness - returns a dictionary
        betweeness_dict = nx.edge_betweenness(input_graph)
        #for k,v in betweeness_dict.iteritems():
        #    betweeness_dict[k] = round(v,12)

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