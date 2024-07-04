from typing import Sequence

def calc_input_dim(node_id, input_conn, input_layer_ids, dims, graph_input_dim) -> int:
    input_conn_nodes = input_conn[node_id]
    input_dims = [dims[input_id] for input_id in input_conn_nodes]
    input_dim = sum(input_dims)
    if node_id in input_layer_ids:
        input_dim += graph_input_dim
    return input_dim


# TODO this is more or less deprecated
def gen_output_connectivity(input_connectivity) -> Sequence[Sequence[int]]:
    output_connectivity = []
    num_layers = len(input_connectivity)
    for i in range(num_layers):
        out_conn_nodei = []
        for j, inp_conn_nodej in enumerate(input_connectivity):
            if i in inp_conn_nodej:
                out_conn_nodei.append(j)
        output_connectivity.append(out_conn_nodei)
    return output_connectivity

