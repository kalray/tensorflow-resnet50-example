import argparse
from tensorflow.core.framework import graph_pb2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_to_summarize", help="GraphProto binary frozen of the tensorflow network to summarize")
    args = parser.parse_args()

    path = args.graph_to_summarize
    with open(path, 'rb') as f:
        graph = graph_pb2.GraphDef()
        s = f.read()
        graph.ParseFromString(s)

    summary = {}
    # Find the input node name in the graph, if not, raise error
    for node in graph.node:
        summary[node.op] = summary.get(node.op, 0) + 1

    print("##################################")
    for o, c in sorted(summary.items()):
        print("Op {} : {}".format(o, c))