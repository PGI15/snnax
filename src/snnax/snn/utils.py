from typing import Sequence

def calc_input_dim(node_id, input_conn, input_layer_ids, dims, graph_input_dim) -> int:
    input_conn_nodes = input_conn[node_id]
    input_dims = [dims[input_id] for input_id in input_conn_nodes]
    input_dim = sum(input_dims)
    if node_id in input_layer_ids:
        input_dim += graph_input_dim
    return input_dim

