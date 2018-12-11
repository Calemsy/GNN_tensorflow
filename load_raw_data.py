import numpy as np
import os.path
import glob


def load_cni1():
    print("load cni1...")
    graph_size = 0
    graphs, labels, nodes_size_list, vertex_tag = [], [], [], []
    justify_count = 0
    with open("./graph_data/nci1/nci1.txt", "r") as f:
        line = f.readline()
        while line:
            line = list(map(int, line.strip().split()))
            if len(line) == 1:
                graph_size = line[0]
            elif len(line) == 2:
                graph, tag, node_size = [], [], line[0]
                labels.append(line[1])
                for cur_node in range(node_size):
                    line = list(map(int, f.readline().strip().split()))
                    tag.append(line[0])
                    for _ in range(line[1]):
                        graph.append([cur_node, line[_ + 2]])
                assert np.max(graph) + 1 == node_size
                justify_count += 1
                graphs.append(graph)
                vertex_tag.append(tag)
                nodes_size_list.append(node_size)
            line = f.readline()
    assert justify_count == graph_size
    assert graph_size == len(graphs) == len(labels) == len(vertex_tag)
    print("\tgraphs: ", len(graphs))
    print("\tmax nodes: %d, \n\tmin nodes: %d, \n\taverage node %.2f." %
          (np.max(nodes_size_list), np.min(nodes_size_list), np.average(nodes_size_list)))
    print("\tvertex tag: ", set(sum(vertex_tag, [])))
    data = {"graphs": graphs,
            "labels": labels,
            "nodes_size_list": nodes_size_list,
            "vertex_tag": vertex_tag,
            "index_from": 0,
            "dimension": None}
    return data


def load_mutag():
    print("load mutag...")
    file_list = []
    file_glob_pattern = os.path.join("graph_data", "mutag", "mutag*.graph")
    file_list.extend(glob.glob(file_glob_pattern))

    graphs, labels, nodes_size_list, vertex_tag, file_name_list = [], [], [], [], []
    for file in file_list:
        file_name_list.append(os.path.basename(file))
        with open(file, "r") as f:
            line = f.readline()
            if line.startswith("#v - vertex labels"):
                tags = []
                line = f.readline()
                while not line.startswith("#e - edge labels"):
                    tags.append(int(line.strip()))
                    line = f.readline()
                vertex_tag.append(tags)
            graph = []
            line = f.readline()
            while not line.startswith("#c - Class"):
                graph.append(list(map(int, line.strip().split(",")))[:2])
                line = f.readline()
            graphs.append(graph)
            labels.append(int(f.readline().strip()))
        nodes_size_list.append(np.max(graph))

    assert len(file_name_list) == len(nodes_size_list) == len(labels) == len(graphs)
    print("\tgraphs: ", len(graphs))
    print("\tmax nodes: %d, \n\tmin nodes: %d, \n\taverage node %.2f." %
          (np.max(nodes_size_list), np.min(nodes_size_list), np.average(nodes_size_list)))
    print("\tvertex tag: ", set(sum(vertex_tag, [])))
    data = {"graphs": graphs,
            "labels": labels,
            "nodes_size_list": nodes_size_list,
            "vertex_tag": vertex_tag,
            "index_from": 1,
            "dimension": None}
    return data

if __name__ == "__main__":
    load_mutag()
    load_cni1()
