from fastcluster import linkage
import logging
import numpy as np
import nmslib
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_distances
import time
from ann_linkage_clustering.helpers import TrackTrailing


def get_gene_neighborhood(central_gene_list, index, df, genes_remaining, starting_n_neighbors, max_dist, threads, max_neighborhood_size=10000):
    """
    Return the gene neighborhood for a particular gene.
    
    `central_gene_list`: The list of primary genes to consider.
    `index`: ANN index.
    `df`: Abundance DataFrame.
    `genes_remaining`: The set of genes that are remaining at this point of the analysis.
    `starting_n_neighbors`: Number of neighbors to find
    `max_dist`: Maximum distance threshold
    `max_neighborhood_size`: Limit the maximum size of a given gene neighborhood

    Return the second order neighbors of the central gene (as a set).
    """

    # Get the first order connections for this list of genes
    first_distances_list = index.knnQueryBatch(
        df.reindex(index=central_gene_list),
        k=starting_n_neighbors,
        num_threads=threads
    )

    list_of_neighborhoods = []

    for central_gene_ix, first_distances in zip(central_gene_list, first_distances_list):

        neighborhood = set([central_gene_ix])

        first_neighbors = [
            ix
            for ix, d in zip(first_distances[0], first_distances[1])
            if d <= max_dist and ix in genes_remaining and ix != central_gene_ix
        ]
        
        # Add the first neighbors to the neighborhood
        neighborhood |= set(first_neighbors)

        # Get the second order distances
        second_distances = index.knnQueryBatch(
            df.reindex(index=first_neighbors),
            k=starting_n_neighbors,
            num_threads=threads
        )

        # Add the second order neighbors to the neighborhood
        for l in second_distances:
            neighborhood |= {
                ix
                for ix, d in zip(l[0], l[1])
                if d <= max_dist and ix in genes_remaining
            }

        # Limit the total size of the neighborhood
        neighborhood = list(neighborhood)
        if len(neighborhood) > max_neighborhood_size:
            neighborhood = neighborhood[:max_neighborhood_size]

        list_of_neighborhoods.append(neighborhood)

    return list_of_neighborhoods


def complete_linkage_clustering(input_data):
    """Return the largest complete linkage group in this set of genes."""

    df, max_dist, distance_metric, linkage_type = input_data

    if df.shape[0] == 1:
        return [set(df.index.values)]

    # Get the flat clusters
    flat_clusters = find_flat_clusters(
        df,
        max_dist,
        distance_metric=distance_metric,
        linkage_type=linkage_type
    )
    
    # Reformat the clusters as a list of lists
    clusters = [
        [] for ix in range(flat_clusters.max() + 1)
    ]
    for cluster_ix, gene_name in zip(
        flat_clusters, df.index.values
    ):
        clusters[cluster_ix].append(gene_name)
    # Sort by size
    clusters = sorted(clusters, key=len, reverse=True)

    # Return the clusters that are at least 50% the size of the largest one
    largest_cluster = len(clusters[0])
    return [
        clusters[ix]
        for ix in range(len(clusters))
        if ix == 0 or len(clusters[ix]) > (largest_cluster / 2)
    ]


def make_cags_with_ann(
    index,
    max_dist,
    df,
    pool,
    threads=1,
    starting_n_neighbors=1000,
    distance_metric="cosine",
    linkage_type="average"
):
    """Make CAGs using the approximate nearest neighbor"""

    # Make a set of the genes remaining to be clustered
    genes_remaining = set(range(df.shape[0]))

    # Keep track of the names of each gene
    gene_names = list(df.index.values)

    # Delete the names of the index
    df = df.reset_index(drop=True)

    # Store CAGs as a dict of lists
    cags = {}
    cag_ix = 0

    # Keep track of the singletons
    singletons = set()

    # Set aside the singletons in the first round
    for gene_ix, gene_distances in enumerate(index.knnQueryBatch(
        df.values,
        k=2,
        num_threads=threads
    )):
        if any([
            neighbor_ix != gene_ix and neighbor_dist <= max_dist
            for neighbor_ix, neighbor_dist in zip(
                gene_distances[0],
                gene_distances[1]
            )
        ]) is False:
            singletons.add(gene_ix)

    # Remove the singletons from the genes remaining
    genes_remaining = genes_remaining - singletons

    logging.info("Masked a set of {:,} singletons in the first pass".format(
        len(singletons)
    ))

    # Keep track of the last 150 CAGs that were added
    trailing = TrackTrailing(n=500)

    # Keep track of the number of genes that were input
    n_genes_input = df.shape[0]

    # Keep clustering until everything is gone
    while len(genes_remaining) > 0 and trailing.max() > 10:

        # Get a list of neighborhoods to test
        logging.info("Getting a list of gene neighborhoods")
        list_of_neighborhoods = get_gene_neighborhood(
            np.random.choice(list(genes_remaining), int(threads * 10)),
            index,
            df,
            genes_remaining,
            starting_n_neighbors,
            max_dist,
            threads
        )

        # Add the singletons
        for n in list_of_neighborhoods:
            if len(n) == 1:
                singletons.add(n[0])
                if n[0] in genes_remaining:
                    genes_remaining.remove(n[0])

        list_of_neighborhoods = [n for n in list_of_neighborhoods if len(n) > 1]
        logging.info("Nonsingleton neighborhoods: {:,}".format(
            len(list_of_neighborhoods)
        ))

        if len(list_of_neighborhoods) == 0:
            continue

        # Find the linkage clusters in parallel
        for list_of_clusters in pool.imap_unordered(
            complete_linkage_clustering,
            [
                (
                    df.reindex(index=neighborhood),
                    max_dist, 
                    distance_metric, 
                    linkage_type
                )
                for neighborhood in list_of_neighborhoods
            ]
        ):
            for linkage_cluster in list_of_clusters:

                # Make a set for comparisons
                linkage_cluster_set = set(linkage_cluster)

                # Make sure that every member of this cluster still needs to be clustered
                if len(linkage_cluster) > 0 and linkage_cluster_set <= genes_remaining:

                    if len(linkage_cluster) == 1:
                        singletons.add(linkage_cluster[0])
                        genes_remaining.remove(linkage_cluster[0])
                        continue

                    logging.info("Adding a CAG with {:,} members, {:,} genes unclustered".format(
                        len(linkage_cluster),
                        len(genes_remaining) - len(linkage_cluster)
                    ))

                    # Add this CAG size to the trailing average
                    trailing.add(len(linkage_cluster))

                    cags[cag_ix] = linkage_cluster
                    cag_ix += 1

                    # Remove these genes from further consideration
                    genes_remaining = genes_remaining - linkage_cluster_set

                    if len(cags) % 1000 == 0:
                        start_time = time.time()
                        df = df.reindex(index=list(genes_remaining))
                        logging.info("Trimmed abundance data to {:,} genes - {:,} seconds".format(
                            df.shape[0], 
                            round(time.time() - start_time, 2)
                        ))

    # Add in all of the singletons
    logging.info("Adding in {:,} singletons that weren't clustered".format(
        len(genes_remaining) + len(singletons)
    ))
    for gene_name in list(genes_remaining) + list(singletons):
        cags[cag_ix] = [gene_name]
        cag_ix += 1

    # Convert all CAGs to gene names
    start_time = time.time()
    cags = {
        k: [
            gene_names[gene_ix]
            for gene_ix in v
        ]
        for k, v in cags.items()
    }
    logging.info("Added full names for CAGs - {:,} seconds elapsed".format(
        round(time.time() - start_time, 2)
    ))

    # Basic sanity checks
    assert all([len(v) > 0 for v in cags.values()])
    assert sum(map(len, cags.values())) == n_genes_input, (sum(
        map(len, cags.values())), n_genes_input)

    return cags


def join_overlapping_cags(cags, df, max_dist, distance_metric="cosine", linkage_type="average", threads=1):
    """Check to see if any CAGs are overlapping. If so, join them and combine the result."""

    # Make a dict linking each gene to its CAG - omitting singletons
    cag_dict = {
        gene_id: cag_id
        for cag_id, gene_id_list in cags.items()
        for gene_id in gene_id_list
    }

    # Make a DF with the mean abundance of each CAG
    logging.info("Computing mean abundances for {:,} CAGs and {:,} genes".format(
        len(set(cag_dict.values())), len(cag_dict)
    ))
    cag_df = pd.concat([
        df,
        pd.DataFrame({"cag": pd.Series(cag_dict)})
    ], axis=1).groupby("cag").mean()

    index = make_nmslib_index(cag_df)

    # Keep track of which CAGs have been regrouped
    already_regrouped = set()

    # Iterate over every CAG
    for cag_ix, distances in enumerate(index.knnQueryBatch(
        cag_df.values,
        k=2,
        num_threads=threads
    )):
        for neighbor_ix, d in zip(distances[0], distances[1]):
            if neighbor_ix != cag_ix and d <= max_dist and cag_ix:
                if cag_ix in already_regrouped or neighbor_ix in already_regrouped:
                    continue

                already_regrouped.add(cag_ix)
                already_regrouped.add(neighbor_ix)
                
                cag_name_1 = cag_df.index.values[cag_ix]
                cag_name_2 = cag_df.index.values[neighbor_ix]
                # Join the CAGs together
                logging.info("Joining CAGs with {:,} and {:,} genes, average distance: {}".format(
                    len(cags[cag_name_1]),
                    len(cags[cag_name_2]),
                    round(d, 4)
                ))

                cags[cag_name_1].extend(cags[cag_name_2])
                del cags[cag_name_2]


def iteratively_refine_cags(cags, df, max_dist, distance_metric="cosine", linkage_type="average", threads=1):
    """Refine the CAGs by merging all groups that are overlapping."""

    # Repeat until all overlapping CAGs are merged, maxing out after a few iterations
    n_cags_previously = len(cags) + 1
    n_iters = 0
    while n_cags_previously != len(cags):

        n_cags_previously = len(cags)

        # Merge any overlapping CAGs
        join_overlapping_cags(
            cags,
            df,
            max_dist,
            distance_metric=distance_metric,
            linkage_type=linkage_type,
            threads=threads
        )

        logging.info("Merging together {:,} CAGs yielded {:,} CAGs".format(
            n_cags_previously, len(cags)
        ))

        n_iters += 1
        if n_iters >= 10:
            logging.info("Done merging together CAGs")
            break

    logging.info("Returning a set of {:,} merged CAGs".format(len(cags)))


def find_flat_clusters(
    df, 
    max_dist,
    linkage_type="average",
    distance_metric="cosine",
    threads=1
):
    """Find the set of flat clusters for a given set of observations."""

    if threads == 1 or df.shape[0] < 2000:
        dm = pdist(df.values, metric=distance_metric)
    else:
        dm = pairwise_distances(df.values, metric=distance_metric, n_jobs=threads)

        # Transform into a condensed distance matrix
        dm = np.concatenate([
            dm[ix, (ix+1):]
            for ix in range(df.shape[0] - 1)
        ])

    # Fill in missing values with 1
    dm[np.isnan(dm)] = 1

    # Now compute the flat clusters
    flat_clusters = fcluster(
        linkage(
            dm,
            method=linkage_type,
            metric="precomputed",
        ), 
        max_dist, 
        criterion="distance"
    )

    assert len(flat_clusters) == df.shape[0]

    return flat_clusters


def make_nmslib_index(df, verbose=True):
    """Make the HNSW index"""
    if verbose:
        logging.info("Making the HNSW index")
    index = nmslib.init(method='hnsw', space='cosinesimil')

    if verbose:
        logging.info("Adding {:,} genes to the nmslib index".format(df.shape[0]))
    index.addDataPointBatch(df.values)
    if verbose:
        logging.info("Making the index")
    index.createIndex({'post': 2, "M": 200}, print_progress=False)

    return index
