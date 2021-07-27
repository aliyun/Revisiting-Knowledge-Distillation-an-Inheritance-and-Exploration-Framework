#! /usr/bin/env python
import pdb
import torch
import torch.nn as nn
import numpy as np


class CKALoss(nn.Module):
    def __init__(self, kernel_method='linear', threshold=1.0):
        super().__init__()
        assert kernel_method in ['linear', 'rbf'], ('Unknown kernel method:'
            '{}'.format(kernel_method))
        self.kernel_method = kernel_method
        self.threshold = threshold

    def forward(self, x, y):
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        if self.kernel_method == 'linear':
            return feature_space_linear_cka(x, y)
        else:
            th = self.threshold
            return cka(gram_rbf(x, th), gram_rbf(y, th))


def gram_linear(x):
    """ Compute Gram (kernel) matrix for a linear kernel.

    Args:
        x: A num_examples x num_features matrix of features.

    Returns:
        A num_examples x num_examples Gram matrix of examples.
    """
    return torch.mm(x, x.t())

def gram_rbf(x, threshold=1.0):
    """ Compute Gram (kernel) matrix for an RBF kernel.

    Args:
        x: A num_examples x num_features matrix of features.
        threshold: Fraction of median Euclidean distance to use as RBF kernel
        bandwidth.

    Returns:
        A num_examples x num_examples Gram matrix of exmaples.
    """
    dot_products = torch.mm(x, x.t())
    sq_norms = torch.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = torch.median(sq_distances)
    return torch.exp(-sq_distances / (2 * threshold **2 * sq_median_distance))

def center_gram(gram, unbiased=False):
    """ Center a symmetric Gram matrix.

    This is equvialent to centering the (possibly infinite-dimensional)
    features induced by the kernel before computing the Gram matrix.

    Args:
        gram: A num_examples x num_examples symmetric matrix.
        unbiased: Whether to adjust the Gram matrix in order to compute an
            unbiased estimate of HSIC. Note that this estimator may be negative,
            but it might be helpful if when the number of examples is small.

    Returns:
        A symmetric matrix with centered columns and rows.
    """
    if not torch.allclose(gram, gram.t()):
        raise ValueError('Input must be a symmetric matrix.')

    if unbiased:
        n = gram.shape[0]
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = torch.mean(gram, dim=0)
        means -= torch.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]

    return gram

def cka(gram_x, gram_y, debiased=False):
    """ Compute CKA.

    Args:
        gram_x: A num_examples x num_examples Gram matrix.
        gram_y: A num_examples x num_examples Gram matrix.
        debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
        The value of CKA between X and Y.
    """
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant)
    # or n*(n-3) (unbiased variant), but this cancels for CAL.
    scaled_hsic = gram_x.view(-1).dot(gram_y.view(-1))
    normalization_x = torch.norm(gram_x)
    normalization_y = torch.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)

def _debiased_dot_product_similarity_helper(xty, sum_squared_rows_x,
                                            sum_squared_rows_y, squared_norm_x,
                                            squared_norm_y, n):
    """Helper for computing debiased dot product similarity (i.e. linear
    HSIC)."""
    return (
        xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
        + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))

def feature_space_linear_cka(features_x, features_y, debiased=False):
    """ Compute CKA with a linear kernel, in feature space. This is faster than
    computing the Gram Matrix when there are fewer features than examples

    Args:
        features_x: A num_examples x num_features matrix of features
        features_y: A num_examples x num_features matrix of features
        debiased: Use unbiased estimator of dot product similarity. CKA may
            still be biased. Note that this estimator may be negative, but it might
            be helpful if the number of examples is small.

    Returns:
        The value of CKA between x and y.
    """
    features_x = features_x - torch.mean(features_x, 0, keepdim=True)
    features_y = features_y - torch.mean(features_y, 0, keepdim=True)

    dot_product_similarity = torch.norm(torch.mm(features_x.t(), features_y)) ** 2
    normalization_x = torch.norm(torch.mm(features_x.t(), features_x))
    normalization_y = torch.norm(torch.mm(features_y.t(), features_y))

    if debiased:
        n = features_x.shape[0]
        # Equivalent to np.sum(features_x **2, 1), but avoids an intermediate
        # array.
        sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
        sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
        squared_norm_x = np.sum(sum_squared_rows_x)
        squared_norm_y = np.sum(sum_squared_rows_y)

        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
            squared_norm_x, squared_norm_y, n)
        normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
            squared_norm_x, squared_norm_x, n))
        normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
            squared_norm_y, squared_norm_y, n))

    return dot_product_similarity / (normalization_x * normalization_y)


# Test CKA
if __name__ == '__main__':
    np.random.seed(1337)
    X = np.random.randn(100, 10)
    Y = np.random.randn(100, 10) + X

    # Linear CKA
    cka_from_examples = cka(gram_linear(X), gram_linear(Y))
    cka_from_features = feature_space_linear_cka(X, Y)
    print('Linear CKA from Examples: {:.5f}'.format(cka_from_examples))
    print('Linear CKA from Features: {:.5f}'.format(cka_from_features))
    np.testing.assert_almost_equal(cka_from_examples, cka_from_features)

    # RBF CKA
    rbf_cka = cka(gram_rbf(X, 0.5), gram_rbf(Y, 0.5))
    print('RBF CKA: {:.5f}'.format(rbf_cka))

    # Debiased CKA
    # If the number of examples is small, it might help to compute a 'debiased'
    # form of CKA.
    cka_from_examples_debiased = cka(gram_linear(X), gram_linear(Y), debiased=True)
    cka_from_features_debiased = feature_space_linear_cka(X, Y, debiased=True)
    print('Linear CKA from Examples (Debiased): {:.5f}'.format(
        cka_from_examples_debiased))
    print('Linear CKA from Features (Debiased): {:.5f}'.format(
        cka_from_features_debiased))
    np.testing.assert_almost_equal(cka_from_examples_debiased,
                                   cka_from_features_debiased)
