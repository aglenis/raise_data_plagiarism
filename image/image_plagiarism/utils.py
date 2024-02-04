from typing import List

import torch


def compute_scores(emb_one: torch.Tensor, emb_two: torch.Tensor) -> List:
    """
    A method that computes cosine similarity between two vectors.
    Args:
        emb_one (torch.Tensor): the first embedding
        emb_two (torch.Tensor): the second embedding

    Returns:
        list
    """
    scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)
    return scores.numpy().tolist()


def fetch_similar(query_embeddings, all_candidate_embeddings):
    """Fetches the `top_k` similar images with `image` as the query."""

    # Compute similarity scores with all the candidate images at one go.
    # We also create a mapping between the candidate image identifiers
    # and their similarity scores with the query image.
    similarity_mapping = {}
    for key, values in all_candidate_embeddings.items():
        if key in [query_embeddings[0]]:
            continue
        sim_scores = compute_scores(torch.Tensor(values), query_embeddings[1])

        similarity_mapping[key] = sim_scores[0]

    # Sort the mapping dictionary and return `top_k` candidates.
    similarity_mapping_sorted = dict(
        sorted(similarity_mapping.items(), key=lambda x: x[1], reverse=True)
    )
    id_entries = list(similarity_mapping_sorted.keys())
    # print(id_entries)
    return similarity_mapping_sorted
