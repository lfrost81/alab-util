import numpy as np


def rank(mat: np.ndarray, iteration=10000, residue=0.85):
    score = np.ones([mat.shape[0], 1]) / mat.shape[0]
    base = np.ones([mat.shape[0], 1]) / mat.shape[0] * (1 - residue)
    mat = mat / mat.sum(axis=0) * residue
    for _ in np.arange(iteration):
        score = np.dot(mat, score) + base
    return score.flatten()


def rank_for_bipartite(mat: np.ndarray, iteration=10000, residue=0.85, border=None):
    score = rank(mat, iteration, residue).flatten()
    score_first = score[0:border] / np.sum(score[0:border])
    score_second = score[border:] / np.sum(score[border:])
    return score_first, score_second
