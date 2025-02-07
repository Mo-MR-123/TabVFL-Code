import torch
import numpy as np

def UnsupervisedLoss(y_pred, embedded_x, obf_vars, eps=1e-9):
    """
    Implements unsupervised loss function.
    This differs from orginal paper as it's scaled to be batch size independent
    and number of features reconstructed independent (by taking the mean)

    Parameters
    ----------
    y_pred : torch.Tensor or np.array
        Reconstructed prediction (with embeddings)
    embedded_x : torch.Tensor
        Original input embedded by network
    obf_vars : torch.Tensor
        Binary mask for obfuscated variables.
        1 means the variable was obfuscated so reconstruction is based on this.
    eps : float
        A small floating point to avoid ZeroDivisionError
        This can happen in degenerated case when a feature has only one value

    Returns
    -------
    loss : torch float
        Unsupervised loss, average value over batch samples.
    """
    errors = y_pred - embedded_x
    reconstruction_errors = torch.mul(errors, obf_vars) ** 2
    batch_means = torch.mean(embedded_x, dim=0)
    batch_means[batch_means == 0] = 1

    batch_stds = torch.std(embedded_x, dim=0) ** 2
    batch_stds[batch_stds == 0] = batch_means[batch_stds == 0]
    features_loss = torch.matmul(reconstruction_errors, 1 / batch_stds)
    # compute the number of obfuscated variables to reconstruct
    nb_reconstructed_variables = torch.sum(obf_vars, dim=1)
    # take the mean of the reconstructed variable errors
    features_loss = features_loss / (nb_reconstructed_variables + eps)
    # here we take the mean per batch, contrary to the paper
    loss = torch.mean(features_loss)
    return loss

class EmbeddingGenerator(torch.nn.Module):
    """
    Classical embeddings generator
    """

    def __init__(self, input_dim, cat_dims, cat_idxs, cat_emb_dims, group_matrix):
        """This is an embedding module for an entire set of features
        Parameters
        ----------
        input_dim : int
            Number of features coming as input (number of columns)
        cat_dims : list of int
            Number of modalities for each categorial features
            If the list is empty, no embeddings will be done
        cat_idxs : list of int
            Positional index for each categorical features in inputs
        cat_emb_dim : list of int
            Embedding dimension for each categorical features
            If int, the same embedding dimension will be used for all categorical features
        group_matrix : torch matrix
            Original group matrix before embeddings
        """
        super(EmbeddingGenerator, self).__init__()

        if cat_dims == [] and cat_idxs == []:
            self.skip_embedding = True
            self.post_embed_dim = input_dim
            self.embedding_group_matrix = group_matrix.to(group_matrix.device)
            return
        else:
            self.skip_embedding = False

        self.post_embed_dim = int(input_dim + np.sum(cat_emb_dims) - len(cat_emb_dims))

        self.embeddings = torch.nn.ModuleList()

        for cat_dim, emb_dim in zip(cat_dims, cat_emb_dims):
            self.embeddings.append(torch.nn.Embedding(cat_dim, emb_dim))

        # record continuous indices
        self.continuous_idx = torch.ones(input_dim, dtype=torch.bool)
        self.continuous_idx[cat_idxs] = 0

        # update group matrix
        n_groups = group_matrix.shape[0]
        self.embedding_group_matrix = torch.empty((n_groups, self.post_embed_dim),
                                                  device=group_matrix.device)
        for group_idx in range(n_groups):
            post_emb_idx = 0
            cat_feat_counter = 0
            for init_feat_idx in range(input_dim):
                if self.continuous_idx[init_feat_idx] == 1:
                    # this means that no embedding is applied to this column
                    self.embedding_group_matrix[group_idx, post_emb_idx] = group_matrix[group_idx, init_feat_idx]  # noqa
                    post_emb_idx += 1
                else:
                    # this is a categorical feature which creates multiple embeddings
                    n_embeddings = cat_emb_dims[cat_feat_counter]
                    self.embedding_group_matrix[group_idx, post_emb_idx:post_emb_idx+n_embeddings] = group_matrix[group_idx, init_feat_idx] / n_embeddings  # noqa
                    post_emb_idx += n_embeddings
                    cat_feat_counter += 1

    def forward(self, x):
        """
        Apply embeddings to inputs
        Inputs should be (batch_size, input_dim)
        Outputs will be of size (batch_size, self.post_embed_dim)
        """
        if self.skip_embedding:
            # no embeddings required
            return x

        cols = []
        cat_feat_counter = 0
        for feat_init_idx, is_continuous in enumerate(self.continuous_idx):
            # Enumerate through continuous idx boolean mask to apply embeddings
            if is_continuous:
                cols.append(x[:, feat_init_idx].float().view(-1, 1))
            else:
                cols.append(
                    self.embeddings[cat_feat_counter](x[:, feat_init_idx].long())
                )
                cat_feat_counter += 1
        # concat
        post_embeddings = torch.cat(cols, dim=1)
        return post_embeddings

class RandomObfuscator(torch.nn.Module):
    """
    Create and applies obfuscation masks.
    The obfuscation is done at group level to match attention.
    """

    def __init__(self, pretraining_ratio, group_matrix):
        """
        This create random obfuscation for self suppervised pretraining
        Parameters
        ----------
        pretraining_ratio : float
            Ratio of feature to randomly discard for reconstruction

        """
        super(RandomObfuscator, self).__init__()
        self.pretraining_ratio = pretraining_ratio
        # group matrix is set to boolean here to pass all posssible information
        self.group_matrix = (group_matrix > 0) + 0.
        self.num_groups = group_matrix.shape[0]

    def forward(self, x):
        """
        Generate random obfuscation mask.

        Returns
        -------
        masked input and obfuscated variables.
        """
        bs = x.shape[0]

        obfuscated_groups = torch.bernoulli(
            self.pretraining_ratio * torch.ones((bs, self.num_groups), device=x.device)
        )
        obfuscated_vars = torch.matmul(obfuscated_groups, self.group_matrix)
        masked_input = torch.mul(1 - obfuscated_vars, x)
        return masked_input, obfuscated_groups, obfuscated_vars
