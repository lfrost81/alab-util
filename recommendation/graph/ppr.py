from recommendation.graph import generator
import networkx as nx
import pickle
import pprint
import csv
import re
import os

data_path = 'd:/data/netflix-prize-data/'


def load_probe():
    probe_dict = {}
    if os.path.exists('probe.bin'):
        with open('probe.bin', 'rb') as fp:
            return pickle.load(fp)

    with open(data_path + 'probe.txt') as probe_fp:
        reader = csv.reader(probe_fp)
        for row in reader:
            if row[0].endswith(':'):
                movie = int(row[0].replace(':', ''))
                if movie not in probe_dict:
                    probe_dict[movie] = set()
            else:
                user = int(row[0])
                probe_dict[movie].add(user)

    print('Load probe')

    with open('probe.bin', 'wb') as ofp:
        pickle.dump(probe_dict, ofp)

    return probe_dict


def generate_train_network(max_size=1000000, refresh=False):
    g = nx.Graph()
    probe_dict = load_probe()

    if refresh is False and os.path.exists('user-movie-bipartite.bin'):
        with open('user-movie-bipartite.bin', 'rb') as fp:
            return pickle.load(fp)

    files = [file for file in os.listdir(data_path)
             if re.search('combined', file)]

    files = sorted(files)
    for file in files:
        with open(data_path + file) as fp:
            reader = csv.reader(fp)
            for i, row in enumerate(reader):
                if i % 10000 == 0:
                    print(i, 'row processed')

                if len(row) == 1:
                    movie = 'm' + row[0].replace(':', '')
                    g.add_node(movie, b='m')
                else:
                    user, rating, date = row
                    if movie in probe_dict and user in probe_dict[movie]:
                        continue

                    # make it empty cell
                    user = int(user)
                    rating = int(rating)

                    g.add_node(user, b='u')
                    g.add_edge(movie, user, w=rating)

                if (i+1) % max_size == 0:
                    print(i+1, 'row processed')
                    break
        break

        print('Generate network')

    with open('user-movie-bipartite.bin', 'wb') as ofp:
        pickle.dump(g, ofp)

    return g


def test():
    b = generator.generate_sample_weighted_bipartite_graph()
    pr = nx.pagerank(b)
    pr = separate_pagerank(pr, extract_entity_node_names(b, entity=1))
    pprint.pprint(pr)

    g = directed_weighted_projection(b, extract_entity_node_names(b, 1))
    pr = nx.pagerank(g)
    pprint.pprint(pr)

    pprint.pprint(g.adj)


def directed_weighted_projection(b, nodes, weight='weight', normalize=True):
    dg = nx.DiGraph()
    for node in nodes:
        dg.add_node(node)
        adj_nodes = [adj_node for adj_node in b[node]]
        for adj_node_edge in b.edges(adj_nodes, data=True):
            _, adj_adj_node, w = adj_node_edge
            w = w[weight]
            if node == adj_adj_node:
                continue

            if adj_adj_node not in dg:
                dg.add_node(adj_adj_node)

            if adj_adj_node not in dg[node]:
                dg.add_edge(node, adj_adj_node)
                dg[node][adj_adj_node] = {weight: 0}
            dg[node][adj_adj_node][weight] += w

        if normalize:
            total_weight = 0
            for value in dg[node].values():
                total_weight += value[weight]

            for key in dg[node].keys():
                dg[node][key][weight] /= total_weight

    return dg


def separate_pagerank(pr, nodes):
    total = sum([pr[n] for n in nodes])
    return {n: pr[n]/total for n in nodes}


def extract_entity_node_names(b, entity, entity_attr_name='bipartite'):
    return [n[0] for n in b.nodes(data=True) if n[1][entity_attr_name] == entity]


def main():
    print('gen')
    b = generate_train_network(2000000, refresh=False)

    print('project')
    movie_nodes = extract_entity_node_names(b, entity='m', entity_attr_name='b')
    if not os.path.exists('user-movie-projected.bin'):
        g = directed_weighted_projection(b, movie_nodes, 'w', normalize=True)

        with open('user-movie-projected.bin', 'wb') as ofp:
            pickle.dump(g, ofp)

    else:
        with open('user-movie-projected.bin', 'rb') as fp:
            g = pickle.load(fp)

    print('pagerank')
    pr = nx.pagerank(g)

    preference = {
        'm1': 5,
        'm3': 4,
    }
    for k in pr:
        pr[k] = 0
    pr.update(preference)
    pr = nx.pagerank(g, alpha=0.9, weight='w', personalization=pr)

    a = sorted(list(pr.items()), key=lambda tup: -tup[1])[0:10]
    pprint.pprint(a)

    recovery_factor = 0
    for k, v in preference.items():
        recovery_factor += v / pr[k]

    recovery_factor /= len(preference)
    pr = {k: v * recovery_factor for k, v in pr.items()}

    a = sorted(list(pr.items()), key=lambda tup: -tup[1])[0:20]
    pprint.pprint(a)


if __name__ == '__main__':
    main()


