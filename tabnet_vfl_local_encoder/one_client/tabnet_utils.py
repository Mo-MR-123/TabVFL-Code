import torch
import numpy as np
from torch.nn import Module

def initialize_non_glu(module: Module, input_dim: int, output_dim: int) -> None:
    """Initialize a linear layer with the Xavier initialization scheme. 
    This is used for the non-gated linear layers. 
    Xavier initialization is designed to keep the scale of the gradients roughly the same in all layers 
    to prevent the gradients from exploding or vanishing during backpropagation, 
    which can cause numerical stability issues. 

    Args:
        module (torch.nn.Module): The pytorch module of which to initialize the weights.
        input_dim (int): The input dimension of the module.
        output_dim (int): The output dimension of the module.
    """
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(4 * input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return

def check_list_groups(list_groups, input_dim):
    """
    Check that list groups:
        - is a list of list
        - does not contain twice the same feature in different groups
        - does not contain unknown features (>= input_dim)
        - does not contain empty groups
    Parameters
    ----------
    - list_groups : list of list of int
        Each element is a list representing features in the same group.
        One feature should appear in maximum one group.
        Feature that don't get assign a group will be in their own group of one feature.
    - input_dim : number of feature in the initial dataset
    """
    assert isinstance(list_groups, list), "list_groups must be a list of list."

    if len(list_groups) == 0:
        return
    else:
        for group_pos, group in enumerate(list_groups):
            msg = f"Groups must be given as a list of list, but found {group} in position {group_pos}."  # noqa
            assert isinstance(group, list), msg
            assert len(group) > 0, "Empty groups are forbidding please remove empty groups []"

    n_elements_in_groups = np.sum([len(group) for group in list_groups])
    flat_list = []
    for group in list_groups:
        flat_list.extend(group)
    unique_elements = np.unique(flat_list)
    n_unique_elements_in_groups = len(unique_elements)
    msg = f"One feature can only appear in one group, please check your grouped_features."
    assert n_unique_elements_in_groups == n_elements_in_groups, msg

    highest_feat = np.max(unique_elements)
    assert highest_feat < input_dim, f"Number of features is {input_dim} but one group contains {highest_feat}."  # noqa
    return

def create_group_matrix(list_groups, input_dim):
    """
    Create the group matrix corresponding to the given list_groups
    Parameters
    ----------
    - list_groups : list of list of int
        Each element is a list representing features in the same group.
        One feature should appear in maximum one group.
        Feature that don't get assigned a group will be in their own group of one feature.
    - input_dim : number of feature in the initial dataset
    Returns
    -------
    - group_matrix : torch matrix
        A matrix of size (n_groups, input_dim)
        where m_ij represents the importance of feature j in group i
        The rows must some to 1 as each group is equally important a priori.
    """
    check_list_groups(list_groups, input_dim)

    if len(list_groups) == 0:
        group_matrix = torch.eye(input_dim)
        return group_matrix
    else:
        n_groups = input_dim - int(np.sum([len(gp) - 1 for gp in list_groups]))
        group_matrix = torch.zeros((n_groups, input_dim))

        remaining_features = [feat_idx for feat_idx in range(input_dim)]

        current_group_idx = 0
        for group in list_groups:
            group_size = len(group)
            for elem_idx in group:
                # add importrance of element in group matrix and corresponding group
                group_matrix[current_group_idx, elem_idx] = 1 / group_size
                # remove features from list of features
                remaining_features.remove(elem_idx)
            # move to next group
            current_group_idx += 1
        # features not mentionned in list_groups get assigned their own group of singleton
        for remaining_feat_idx in remaining_features:
            group_matrix[current_group_idx, remaining_feat_idx] = 1
            current_group_idx += 1
        return group_matrix

def check_embedding_parameters(cat_dims, cat_idxs, cat_emb_dim):
    """
    Check parameters related to embeddings and rearrange them in a unique manner.
    """
    if (cat_dims == []) ^ (cat_idxs == []):
        if cat_dims == []:
            msg = "If cat_idxs is non-empty, cat_dims must be defined as a list of same length."
        else:
            msg = "If cat_dims is non-empty, cat_idxs must be defined as a list of same length."
        raise ValueError(msg)
    elif len(cat_dims) != len(cat_idxs):
        msg = "The lists cat_dims and cat_idxs must have the same length."
        raise ValueError(msg)

    if isinstance(cat_emb_dim, int):
        cat_emb_dims = [cat_emb_dim] * len(cat_idxs)
    else:
        cat_emb_dims = cat_emb_dim

    # check that all embeddings are provided
    if len(cat_emb_dims) != len(cat_dims):
        msg = f"""cat_emb_dim and cat_dims must be lists of same length, got {len(cat_emb_dims)}
                    and {len(cat_dims)}"""
        raise ValueError(msg)

    # Rearrange to get reproducible seeds with different ordering
    if len(cat_idxs) > 0:
        sorted_idxs = np.argsort(cat_idxs)
        cat_dims = [cat_dims[i] for i in sorted_idxs]
        cat_emb_dims = [cat_emb_dims[i] for i in sorted_idxs]

    return cat_dims, cat_idxs, cat_emb_dims