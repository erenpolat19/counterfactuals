
import networkx as nx
import matplotlib.pyplot as plt
import torch


def visualize(batch, expl_mask):
    # Extract node features and edge index from the batch
    x = batch.x
    edge_index = batch.edge_index

    print(f"Initial edge_index shape: {edge_index.shape}")  # Shape [2, N_edges]
    print(f"Initial expl_mask shape: {expl_mask.shape}")    # Shape [N_edges]

    G = nx.Graph()

    # Apply the batch mask to select nodes from the first graph (batch == 1)
    mask = batch.batch == 1  # Assuming batch.batch tells which graph each node belongs to
    node_indices = mask.nonzero(as_tuple=False).view(-1)
    node_features = x[node_indices]
    
    # Apply the node mask to filter the edges in edge_index
    edge_mask = mask[edge_index[0]]  # Select edges where the source node is part of the mask
    edge_index_filtered = edge_index[:, edge_mask]  # Filter the edge index

    # Also filter the explanation mask to keep only relevant edges
    expl_mask_filtered = expl_mask[edge_mask]  # Filter the explanation mask
    
    print(f"Filtered edge_index shape: {edge_index_filtered.shape}")  # Should be [2, M_edges]
    print(f"Filtered expl_mask shape: {expl_mask_filtered.shape}")    # Should be [M_edges]
    
    # Select top K edges based on the filtered explanation mask
    k = min((expl_mask_filtered.shape[0]), 10)
    top_k_values, top_k_indices = torch.topk(expl_mask_filtered, k)
    
    for i in range(k):
        u = edge_index_filtered[0][top_k_indices[i]]
        v = edge_index_filtered[1][top_k_indices[i]]
        G.add_edge(u.item(), v.item())

    # Identify and visualize the nodes
    print("Red node:", node_indices.tolist()[0])
    node_to_identify = None  # Example: node to highlight
    
    for i, node_idx in enumerate(node_indices):
        feature_tuple = tuple(node_features[i].tolist())
        G.add_node(node_idx.item(), feature=feature_tuple)

    node_colors = ['red' if node == node_to_identify else 'blue' for node in G.nodes()]
    nx.draw(G, with_labels=False, node_size=20, node_color=node_colors)
    plt.show()


# make batch size 1 then just do it once on all the test graphs
def visualize_old(batch, expl_mask):
    # print(batch)
    x = batch.x
    edge_index = batch.edge_index

    # print(edge_index.shape)
    G = nx.Graph()
    # get the first batch
    mask = batch.batch == 1

    edge_mask = mask[edge_index[0]]  # This gives the mask for the edges, resulting in 50 edges being selected

    # Step 2: Apply the mask to both edge_index and expl_mask
    edge_index_filtered = edge_index[:, edge_mask]  # Shape will be [2, 50]
    expl_mask_filtered = expl_mask[edge_mask]  # This will give the mask corresponding to the selected 50 edges

    # Step 3: Now you can visualize with the filtered edge index and explanation mask
    print("Filtered edge index shape:", edge_index_filtered.shape)  # Should be [2, 50]
    print("Filtered expl_mask shape:", expl_mask_filtered.shape)



    # node_indices = mask.nonzero(as_tuple=False).view(-1)
    # node_features = x[node_indices]
    # edge_index = edge_index_full[:, mask[edge_index_full[0]]]

    # print("Edge index shape:", edge_index.shape)  # Should be [2, 3200] or similar
    # print("Mask shape (applied to edge_index[0]):", mask[edge_index[0]].shape)  # Should match number of edges
    # print("Expl mask shape:", expl_mask.shape)  # Should be [3200]


    # edge_mask = mask[edge_index[0]]  
    # edge_mask = expl_mask[edge_mask] 

    # print(node_indices.shape)
    # # print(edge_index)
    # print(edge_index.shape)
    # print(expl_mask.shape)
    # print(mask.shape)


    #  expl_mask = expl_mask[mask[edge_index_full[0]]]

    k = 50
    top_k_values, top_k_indices = torch.topk(expl_mask, k)
    print(top_k_indices)
    for i in range(k):
        u = edge_index[0][top_k_indices[i]]
        v = edge_index[1][top_k_indices[i]]
        G.add_edge(u, v)

    print("Red node:", node_indices.tolist()[0])

    node_to_identify = None #node_indices.tolist()[0]


    for i, node_idx in enumerate(node_indices):
        feature_tuple = tuple(node_features[i].tolist())  
        G.add_node(node_idx.item(), feature=feature_tuple)


    node_colors = ['red' if node == node_to_identify else 'blue' for node in G.nodes()]
    nx.draw(G, with_labels=False, node_size=20, node_color=node_colors)
    plt.show()