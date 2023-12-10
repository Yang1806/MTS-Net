"""
    Utility file to select GraphNN model as
    selected by the user
"""
from nets.molecules_graph.graph_transformer_net import GraphTransformerNet

def GraphTransformer(net_params):
    return GraphTransformerNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'MTS-Net': GraphTransformer
    }
        
    return models[MODEL_NAME](net_params)