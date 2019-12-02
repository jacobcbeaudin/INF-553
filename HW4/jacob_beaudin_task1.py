from pyspark import SparkContext
from itertools import islice, combinations
from collections import deque, defaultdict
from sys import argv


def start_spark_context():
    sc = SparkContext('local[*]', 'hw4')
    sc.setLogLevel("OFF")

    return sc


def get_graph():
    graph = sc.textFile(argv[2]). \
        mapPartitionsWithIndex(lambda i, element: islice(element, 1, None) if i == 0 else element). \
        map(lambda element: element.split(',')).groupByKey(). \
        map(lambda element: (element[0], set(element[1]))). \
        filter(lambda element: element[1].__len__() >= int(argv[1])). \
        repartition(1). \
        mapPartitions(lambda element: combinations(tuple(element), 2)). \
        map(lambda element: ((element[0][0], element[1][0]), element[0][1].intersection(element[1][1]).__len__())). \
        filter(lambda element: element[1] >= int(argv[1])). \
        flatMap(lambda element: (element[0], (element[0][1], element[0][0]))).groupByKey(). \
        map(lambda element: (element[0], set(element[1]))).collect()

    return dict(graph)


def get_shortest_path_vals(root_node):
    # level, count
    node_min_path = dict()
    node_min_path['level'] = defaultdict(lambda: 0)
    node_min_path['count'] = defaultdict(lambda: 0)

    # keep track of parents nodes
    node_parent_dict = dict()
    # keep track of nodes to be checked
    exploring = deque([root_node],)
    # keep track of all explored nodes
    explored = deque()

    return explored, node_parent_dict, node_min_path, exploring


def BFS(graph_dict, exploring, explored, node_parent_dict, node_min_path, root_node):
    while exploring:
        node = exploring.popleft()

        if node == root_node:
            node_min_path['count'][node] = 1
            node_parent_dict[node] = 0,

        if node not in explored:
            for neighbor in graph_dict[node]:
                if neighbor not in explored:
                    if neighbor not in node_min_path['level']:
                        node_min_path['count'][neighbor] += node_min_path['count'][node]
                        node_parent_dict[neighbor] = [node, ]
                        node_min_path['level'][neighbor] = node_min_path['level'][node] + 1
                        exploring.append(neighbor)

                    else:
                        # make sure do not have same parent
                        if node_min_path['level'][neighbor] != node_min_path['level'][node]:
                            node_min_path['count'][neighbor] += node_min_path['count'][node]
                            node_parent_dict[neighbor].append(node, )

            explored.append(node)

    return explored, node_parent_dict, node_min_path


def get_node_intermittent_gn(node_parent_dict, node_min_path, explored):
    node_intermittent_gn = defaultdict(lambda: 1.0)
    while explored:
        node = explored.pop()
        for parent in node_parent_dict[node]:
            if parent:
                node_intermittent_gn[parent] += node_min_path['count'][parent]/\
                                                node_min_path['count'][node]*node_intermittent_gn[node]

    return node_intermittent_gn


def get_edge_betweeness_BFS(node_parent_dict, node_min_path, node_intermittent_gn, explored):
    edge_betweeness = defaultdict(lambda: 0.0)
    while explored:
        node = explored.pop()
        for parent in node_parent_dict[node]:
            if parent:
                edge_betweeness[tuple(sorted([node, parent]))] += \
                    node_min_path['count'][parent] / \
                    node_min_path['count'][node]*node_intermittent_gn[node] / 2

    return edge_betweeness


def girvan_newman(graph_dict, root_node):
    explored, node_parent_dict, node_min_path, exploring = get_shortest_path_vals(root_node)

    explored, node_parent_dict, node_min_path = BFS(graph_dict, exploring, explored, node_parent_dict, node_min_path, root_node)

    explored_copy = explored.copy()

    node_intermittent_gn = get_node_intermittent_gn(node_parent_dict, node_min_path, explored_copy)
    edge_betweeness_BFS = get_edge_betweeness_BFS(node_parent_dict, node_min_path, node_intermittent_gn, explored)

    return edge_betweeness_BFS


def write_edge_betweeness_graph(edge_betweeness):
    f = open(argv[3], 'w')
    for line in edge_betweeness:
        f.write(str(line)[1:-1]+'\n')
    f.close()

    return


def sort_edge_betweeness(edge_betweeness_dict):

    return sorted(edge_betweeness_dict.items(), key=lambda element: (-element[1], element[0]))


def get_edge_betweeness_graph(graph):
    edge_betweeness_dict = defaultdict(lambda: 0)
    for root_node in graph.keys():
        edge_betweeness_BFS = girvan_newman(graph, root_node)
        for edge, betweeness in edge_betweeness_BFS.items():
            edge_betweeness_dict[edge] += betweeness

    edge_betweeness = sort_edge_betweeness(edge_betweeness_dict)

    return edge_betweeness


def get_modularity_constants(graph):
    modularity_constants = dict()
    modularity_constants['A'] = defaultdict(lambda: 0)
    modularity_constants['k'] = dict()
    for i, js in graph.items():
        modularity_constants['k'][i] = js.__len__()
        for j in js:
            modularity_constants['A'][(i, j)] = 1

    modularity_constants['m'] = modularity_constants['A'].__len__() / 2
    return modularity_constants


def get_modularity(modularity_constants, communities):
    modularity = 0
    for community in communities:
        for i in community:
            for j in community:
                modularity += modularity_constants['A'][(i, j)] - \
                              modularity_constants['k'][i] * \
                              modularity_constants['k'][j] / \
                              (2*modularity_constants['m'])

    modularity = modularity/(2*modularity_constants['m'])

    return modularity


def get_max_modularity_communities(optimal_graph, communities, modularity):

    optimal_graph['modularity'] = modularity
    optimal_graph['communities'] = communities

    return optimal_graph


def remove_edges(edge_betweeness, graph):
    edge_betweeness = deque(edge_betweeness)
    max_betweeness = edge_betweeness[0][1]
    removing_edges = []

    while edge_betweeness[0][1] == max_betweeness:
        removing_edges.append(edge_betweeness.popleft()[0])
    for edge in removing_edges:

        graph[edge[0]].remove(edge[1])
        graph[edge[1]].remove(edge[0])
    return graph


def identify_communities(graph):
    communities = []
    explored_external = set()

    for root_node in graph:
        if root_node not in explored_external:

            exploring = deque([root_node],)
            explored_internal = set()

            while exploring:
                node = exploring.popleft()

                for neighbor in graph[node]:
                    if neighbor not in explored_internal:
                        exploring.append(neighbor,)

                explored_internal.add(node,)

            communities.append(frozenset(explored_internal))
            explored_external.union(explored_internal)

    communities = list(map(list, set(communities)))

    return communities


def write_community_output(communities):
    f = open(argv[4], 'w')
    for line in communities:
        f.write(str(line)[1:-1] + '\n')
    f.close()

    return


sc = start_spark_context()
graph = get_graph()
edge_betweeness = get_edge_betweeness_graph(graph)
write_edge_betweeness_graph(edge_betweeness)

modularity_constants = get_modularity_constants(graph)

communities = identify_communities(graph)
modularity = get_modularity(modularity_constants, communities)

optimal_graph = dict()
optimal_graph['modularity'] = modularity
optimal_graph['communities'] = communities

while edge_betweeness[0][1] != 1.0:

    edge_betweeness = get_edge_betweeness_graph(graph)
    if edge_betweeness[0][1] != 1.0:
        graph = remove_edges(edge_betweeness, graph)
        communities = identify_communities(graph)
    else:
        communities = tuple(map(lambda x: (x,), graph.keys()))

    modularity = get_modularity(modularity_constants, communities)

    if modularity > optimal_graph['modularity']:
        optimal_graph['modularity'] = modularity
        optimal_graph['communities'] = communities

communities = sorted(list(map(sorted, optimal_graph['communities'])), key=lambda element: (len(element), element))
write_community_output(communities)
