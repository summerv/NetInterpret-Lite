import csv
import networkx as nx
from networkx.algorithms.community import k_clique_communities
import matplotlib.pyplot as plt
import time
import numpy as np

def membership():
    in_file0 = open("sample_acti_rate_sample.csv", "r")
    csv_reader = csv.reader(in_file0)

    i = 0
    for line in csv_reader:
        if i == 0:
            feature_names = line[2:]
            i += 1
            continue

    in_file = open("membership.txt", "r")
    membership = []
    for line in in_file.readlines():
        line_list = line.strip().split(" ")
        line_list = [item for item in line_list if item != '']
        membership.extend(line_list)
    print(len(membership))

    group = {}
    for i in range(512):
        group.setdefault(membership[i], [])
        group[membership[i]].append(feature_names[i])
    print(len(group))
    print(group)
    for item in group.items():
        print("group has %d members, including:" % len(item[1]))
        print(item[1])


def gen_data():
    in_file = open("sample_acti_rate.csv", "r")
    mmap_file = "weight.mmap"
    csv_reader = csv.reader(in_file)

    i = 0
    weight = np.memmap(mmap_file, dtype=int, mode='w+', shape=(512, 512))
    for line in csv_reader:
        if i == 0:
            feature_names = line[2:]
            i += 1
            continue
        acti = line[2:]
        activated_units = [unit for unit in range(len(feature_names)) if float(acti[unit]) > 0.0]
        for i in range(len(activated_units)):
            for j in range(i + 1, len(activated_units)):
                unit_from = activated_units[i]
                unit_to = activated_units[j]
                weight[unit_from][unit_to] += 1.0
    csv_writer = csv.writer(open("matrix.csv", "w", newline=''))
    for i in range(512):
        csv_writer.writerow([float(w) for w in weight[i]])


def gen_networkx():
    start_time = time.time()
    in_file = open("sample_acti_rate_sample.csv", "r")
    csv_reader = csv.reader(in_file)

    i = 0
    G = nx.Graph()
    weighted_edges = {}
    for line in csv_reader:
        if i == 0:
            feature_names = line[2:]
            i += 1
            continue
        file = line[0]
        category = line[1]
        acti = line[2:]

        activated_units = [unit for unit in range(len(feature_names)) if float(acti[unit]) > 0.0]
        for i in range(len(activated_units)):
            for j in range(i+1, len(activated_units)):
                unit_from = activated_units[i]
                unit_to = activated_units[j]
                weighted_edges[(unit_from, unit_to)] = weighted_edges.setdefault((unit_from, unit_to), 0) + 1

    # for item in weighted_edges.items():
    #     print(item)

    cnt = 1
    for item in weighted_edges.items():
        if cnt % 50 == 0:
            print("adding edge No. %d" % cnt)
        cnt += 1
        G.add_edge(item[0][0], item[0][1], weight=item[1])
    #     nx.write_weighted_edgelist(G, 'test.weighted.edgelist')
    # print(len(feature_names))
    # G = nx.read_weighted_edgelist('test.weighted.edgelist')

    klist = list(k_clique_communities(G, 5))
    print(len(klist))
    pos = nx.spring_layout(G)
    plt.clf()
    nx.draw(G, pos=pos, with_labels=False)
    nx.draw(G, pos=pos, nodelist=klist[0], node_color='b')
    # nx.draw(G, pos=pos, nodelist=klist[1], node_color='y')
    # nx.draw(G, pos=pos, nodelist=klist[2], node_color='g')
    # nx.draw(G, pos=pos, nodelist=klist[3], node_color='c')
    plt.savefig('test.jpg')
    # plt.show()
    in_file.close()

    print("using time: %d s." % (time.time() - start_time))

if __name__ == '__main__':
    membership()