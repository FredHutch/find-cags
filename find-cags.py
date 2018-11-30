#!/usr/bin/env python3

import argparse
import boto3
import copy
from collections import defaultdict
from fastcluster import linkage
import gzip
import io
import json
import logging
from math import ceil
from multiprocessing import Pool
import nmslib
import numpy as np
import os
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist, cdist
from scipy.stats import gmean
import shutil
from sklearn.metrics import pairwise_distances
import sys
import time
import traceback
import uuid


def exit_and_clean_up(temp_folder):
    """Log the error messages and delete the temporary folder."""
    # Capture the traceback
    logging.info("There was an unexpected failure")
    exc_type, exc_value, exc_traceback = sys.exc_info()
    for line in traceback.format_tb(exc_traceback):
        logging.info(line)

    # Delete any files that were created for this sample
    logging.info("Removing temporary folder: " + temp_folder)
    shutil.rmtree(temp_folder)

    # Exit
    logging.info("Exit type: {}".format(exc_type))
    logging.info("Exit code: {}".format(exc_value))
    sys.exit(exc_value)


def read_json(fp):
    assert fp.endswith((".json", ".json.gz"))
    logging.info("Reading in " + fp)
    if fp.startswith("s3://"):
        # Parse the S3 bucket and key
        bucket_name, key_name = fp[5:].split("/", 1)

        # Connect to the S3 boto3 client
        s3 = boto3.client('s3')

        # Download the object
        retr = s3.get_object(Bucket=bucket_name, Key=key_name)

        if fp.endswith(".gz"):
            # Parse GZIP
            bytestream = io.BytesIO(retr['Body'].read())
            got_text = gzip.GzipFile(
                None, 'rb', fileobj=bytestream).read().decode('utf-8')
        else:
            # Read text
            got_text = retr['Body'].read().decode('utf-8')

        # Parse the JSON
        dat = json.loads(got_text)

    else:
        assert os.path.exists(fp)

        if fp.endswith(".gz"):
            dat = json.load(gzip.open(fp, "rt"))
        else:
            dat = json.load(open(fp, "rt"))

    # Make sure that the sample sheet is a dictionary
    assert isinstance(dat, dict)

    return dat


def normalize_abundance_dataframe(df, normalization):
    """Normalize the raw depth values on a per-sample basis."""

    assert normalization in ["median", "sum", "clr"]
    logging.info("Normalizing the abundance values by " + normalization)

    # Normalize the abundance

    if normalization == "median":
        # Divide by the median on a per-sample basis
        df = df.apply(lambda v: v / v.loc[v > 0].median())

    elif normalization == "sum":
        # Divide by the median on a per-sample basis
        df = df / df.sum()

    elif normalization == "clr":
        # Divide by the median on a per-sample basis
        df = df.apply(lambda v: v / v.loc[v > 0].median())

        # Replace the zeros with the lowest non-zero value
        lowest_non_zero_value = df.apply(lambda v: v.loc[v > 0].min()).min()
        df.replace(to_replace={0: lowest_non_zero_value}, inplace=True)

        # Now take the log10
        df = df.apply(np.log10)

    # There are no NaN values
    assert df.shape[0] == df.dropna().shape[0]

    return df


def make_abundance_dataframe(sample_sheet, results_key, abundance_key, gene_id_key):
    """Make a single DataFrame with the abundance (depth) from all samples."""

    # Collect all of the abundance information in this single dict
    dat = {}

    # Iterate over each sample
    for sample_name, sample_path in sample_sheet.items():
        # Get the JSON for this particular sample
        sample_dat = read_json(sample_path)

        # Make sure that the key for the results is in this file
        assert results_key in sample_dat

        # Subset down to the list of results
        sample_dat = sample_dat[results_key]
        assert isinstance(sample_dat, list)

        # Make sure that every element in the list has the indicated keys
        for d in sample_dat:
            assert abundance_key in d
            assert gene_id_key in d

        # Format as a Series
        dat[sample_name] = pd.Series({
            d[gene_id_key]: np.float16(d[abundance_key])
            for d in sample_dat
        })

    logging.info("Formatting as a DataFrame")
    dat = pd.DataFrame(dat).fillna(np.float16(0))

    logging.info("Read in data for {:,} genes across {:,} samples".format(
        dat.shape[0],
        dat.shape[1]
    ))

    return dat


def make_summary_abund_df(df, cags):
    """Make a DataFrame with the average value for each CAG."""
    # Make sure that the members of every cag are rows in the `df`
    row_names = set(df.index.values)
    assert all([
        cag_member in row_names
        for cag_member_list in cags.values()
        for cag_member in cag_member_list
    ])

    summary_df = pd.DataFrame({
        cag_ix: df.loc[cag].mean()
        for cag_ix, cag in cags.items()
    }).T

    assert summary_df.shape[0] == len(cags)
    assert summary_df.shape[1] == df.shape[1]

    return summary_df


def return_results(df, summary_df, cags, log_fp, output_prefix, output_folder, temp_folder):
    """Write out all of the results to a file and copy to a final output directory"""

    # Make sure the output folder ends with a '/'
    if output_folder.endswith("/") is False:
        output_folder = output_folder + "/"

    if output_folder.startswith("s3://"):
        s3 = boto3.resource('s3')

    for suffix, obj in [
        (".feather", df),
        (".cags.feather", summary_df),
        (".cags.json.gz", cags),
        (".logs.txt", log_fp)
    ]:
        if obj is None:
            "Skipping {}{}, no data available".format(output_prefix, suffix)
            continue

        fp = os.path.join(temp_folder, output_prefix + suffix)
        if suffix.endswith(".feather"):
            obj.reset_index().to_feather(fp)
        elif suffix.endswith(".json.gz"):
            json.dump(obj, gzip.open(fp, "wt"))
        elif suffix.endswith(".txt"):
            with open(fp, "wt") as f:
                f.write(obj)
        else:
            raise Exception(
                "Object cannot be written, no method for " + suffix)

        if output_folder.startswith("s3://"):
            bucket, prefix = output_folder[5:].split("/", 1)

            # Make the full name of the destination key
            file_prefix = prefix + output_prefix + suffix

            # Copy the file
            logging.info("Copying {} to {}/{}".format(
                fp,
                bucket,
                file_prefix
            ))
            s3.Bucket(bucket).upload_file(fp, file_prefix)

        else:
            # Copy as a local file
            logging.info("Copying {} to {}".format(
                fp, output_folder
            ))
            shutil.copy(fp, output_folder)


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


def find_flat_clusters(
    df, 
    max_dist,
    linkage_type="average",
    distance_metric="cosine",
    threads=1
):
    """Find the set of flat clusters for a given set of observations."""

    assert df.isnull().any().any() == False, "NaN value(s) in DataFrame"

    if threads == 1 or df.shape[0] < 2000:
        try:
            dm = pdist(df.values, metric=distance_metric)
        except:
            logging.info("Fatal problem with dataframe")
            print(df.values)
            assert 1 == 0
    else:
        dm = pairwise_distances(df.values, metric=distance_metric, n_jobs=threads)

        # Transform into a condensed distance matrix
        dm = np.concatenate([
            dm[ix, (ix+1):]
            for ix in range(df.shape[0] - 1)
        ])

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


class TrackTrailing():

    def __init__(self, n=100, start=1000):
        self.cache = np.array([start for ix in range(n)])
        self.counter = 0
        self.n = n

    def add(self, new_size):
        self.cache[self.counter % self.n] = new_size

    def average(self):
        return np.mean(self.cache)

    def max(self):
        return np.max(self.cache)


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


def find_cags(
    sample_sheet=None,
    output_prefix=None,
    output_folder=None,
    normalization=None,
    max_dist=0.1,
    temp_folder="/scratch",
    results_key="results",
    abundance_key="depth",
    gene_id_key="id",
    threads=1,
    min_samples=1,
    test=False,
    clr_floor=None,
    distance_metric="cosine",
    linkage_type="average"
):
    # Make sure the temporary folder exists
    assert os.path.exists(temp_folder)

    # Make a new temp folder
    temp_folder = os.path.join(temp_folder, str(uuid.uuid4())[:8])
    os.mkdir(temp_folder)

    # Set up logging
    log_fp = os.path.join(temp_folder, "log.txt")
    logFormatter = logging.Formatter(
        '%(asctime)s %(levelname)-8s [find-cags] %(message)s'
    )
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    # Write to file
    fileHandler = logging.FileHandler(log_fp)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # Also write to STDOUT
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    # Set up the multiprocessing pool
    pool = Pool(threads)

    # READING IN DATA

    # Read in the sample_sheet
    logging.info("Reading in the sample sheet from " + sample_sheet)
    try:
        sample_sheet = read_json(sample_sheet)
    except:
        exit_and_clean_up(temp_folder)

    # Make the abundance DataFrame
    logging.info("Making the abundance DataFrame")
    try:
        df = make_abundance_dataframe(
            sample_sheet,
            results_key,
            abundance_key,
            gene_id_key
        )
    except:
        exit_and_clean_up(temp_folder)

    # NORMALIZING RAW ABUNDANCES

    # Normalize the raw depth abundances
    try:
        df = normalize_abundance_dataframe(df, normalization)
    except:
        exit_and_clean_up(temp_folder)

    # Make a copy of this abundance table, to be saved at the end
    unfiltered_abundance_df = copy.deepcopy(df)

    # Apply the clr_floor parameter, if applicable
    if clr_floor is not None and normalization == "clr":
        if clr_floor == "auto":
            clr_floor = df.min().max()
            logging.info("Automatically set the minimum CLR as {:,}".format(clr_floor))
        else:
            logging.info("User passed in {} as the minimum CLR value".format(clr_floor))
            try:
                clr_floor = float(clr_floor)
            except:
                logging.info("{:,} could not be evaluated as a float".format(clr_floor))
                exit_and_clean_up(temp_folder)

        logging.info("Applying the CLR floor: {}".format(clr_floor))
        df = df.applymap(lambda v: v if v > clr_floor else clr_floor)

    # If min_samples > 1, subset the genes
    logging.info(
        "Subsetting to genes found in at least {} samples".format(min_samples))

    # Keep track of the number of genes filtered, and the time elapsed
    n_before_filtering = df.shape[0]
    start_time = time.time()

    # Filter
    df = df.loc[(df > df.min().min()).sum(axis=1) >= min_samples]

    logging.info("{:,} / {:,} genes found in >= {:,} samples ({:,} seconds elapsed)".format(
        df.shape[0],
        n_before_filtering,
        min_samples,
        round(time.time() - start_time, 2)
    ))

    # If this is being run in testing mode, subset to 2,000 genes
    if test:
        logging.info("Running in testing mode, subset to 2,000 genes")
        df = df.head(2000)

    # Make sure that the lowest abundance is 0 (for clustering)
    logging.info("Shifting the lowest abundance to 0 (for the purpose of calculating distance metrics)")
    df = df - df.min().min()

    # CLUSTERING

    # Make the nmslib index
    logging.info("Making the nmslib index")
    index = make_nmslib_index(df)

    logging.info("Finding CAGs")
    # Make CAGs using the approximate nearest neighbor
    try:
        cags = make_cags_with_ann(
            index,
            max_dist,
            df.copy(),
            pool,
            threads=threads,
            distance_metric=distance_metric,
            linkage_type=linkage_type
        )
    except:
        exit_and_clean_up(temp_folder)

    logging.info("Closing the process pool")
    pool.close()

    # Refine the CAGs by merging overlapping groups
    try:
        iteratively_refine_cags(
            cags,
            df.copy(),
            max_dist,
            distance_metric=distance_metric,
            linkage_type=linkage_type,
            threads=threads
        )
    except:
        exit_and_clean_up(temp_folder)
    
    # Rename the CAGs
    cags = {
        ix: list_of_genes
        for ix, list_of_genes in enumerate(sorted(list(cags.values()), key=len, reverse=True))
    }

    # Print the number of total CAGs, number of singletons, etc.
    logging.info("Number of CAGs = {:,}".format(
        len(cags)
    ))
    # Print the number of total CAGs, number of singletons, etc.
    logging.info("Number of singletons = {:,} / {:,}".format(
        sum([len(v) == 1 for v in cags.values()]),
        sum(map(len, cags.values()))
    ))
    # Print the number of total CAGs, number of singletons, etc.
    logging.info("Largest CAG = {:,}".format(
        max(map(len, cags.values()))
    ))

    logging.info("Size distribution of CAGs:")
    logging.info(pd.Series(list(map(len, cags.values()))).describe())

    logging.info("Largest CAGs:")
    largest_cags = pd.Series(dict(zip(
        cags.keys(),
        map(len, cags.values())
    )))
    largest_cags.sort_values(ascending=False, inplace=True)
    for cag_id, cag_size in largest_cags.head(10).items():
        logging.info("{}: {:,}".format(cag_id, cag_size))

    # RETURN RESULTS

    # Now make a summary DF with the mean value for each combined CAG
    summary_df = make_summary_abund_df(unfiltered_abundance_df, cags)

    # Read in the logs
    logs = "\n".join(open(log_fp, "rt").readlines())

    # Return the results
    logging.info("Returning results to " + output_folder)
    try:
        return_results(
            unfiltered_abundance_df,
            summary_df,
            cags,
            logs,
            output_prefix,
            output_folder,
            temp_folder
        )
    except:
        exit_and_clean_up(temp_folder)

    # Delete any files that were created for this sample
    logging.info("Removing temporary folder: " + temp_folder)
    shutil.rmtree(temp_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Find a set of co-abundant genes"""
    )

    parser.add_argument("--sample-sheet",
                        type=str,
                        required=True,
                        help="""Location for sample sheet (.json[.gz]).""")
    parser.add_argument("--output-prefix",
                        type=str,
                        required=True,
                        help="""Prefix for output files.""")
    parser.add_argument("--output-folder",
                        type=str,
                        required=True,
                        help="""Folder to place results.
                                (Supported: s3://, or local path).""")
    parser.add_argument("--normalization",
                        type=str,
                        default=None,
                        help="Normalization factor per-sample (median, sum, or clr).")
    parser.add_argument("--max-dist",
                        type=float,
                        default=0.01,
                        help="Maximum cosine distance for clustering.")
    parser.add_argument("--temp-folder",
                        type=str,
                        default="/scratch",
                        help="Folder for temporary files.")
    parser.add_argument("--results-key",
                        type=str,
                        default="results",
                        help="Key identifying the list of gene abundances for each sample JSON.")
    parser.add_argument("--abundance-key",
                        type=str,
                        default="depth",
                        help="Key identifying the abundance value for each element in the results list.")
    parser.add_argument("--gene-id-key",
                        type=str,
                        default="id",
                        help="Key identifying the gene ID for each element in the results list.")
    parser.add_argument("--threads",
                        type=int,
                        default=1,
                        help="Number of threads to use.")
    parser.add_argument("--min-samples",
                        type=int,
                        default=1,
                        help="Filter genes by the number of samples they are found in.")
    parser.add_argument("--clr-floor",
                        type=str,
                        default=None,
                        help="Set a minimum CLR value, 'auto' will use the largest minimum value.")
    parser.add_argument("--test",
                        action="store_true",
                        help="Run in testing mode and only process a subset of 2,000 genes.")

    args = parser.parse_args(sys.argv[1:])

    # Sample sheet is in JSON format
    assert args.sample_sheet.endswith((".json", ".json.gz"))

    # Normalization factor is absent, 'median', or 'sum'
    assert args.normalization in [None, "median", "sum", "clr"]

    # max-dist is >=0
    assert args.max_dist >= 0

    assert args.threads >= 1

    assert args.min_samples >= 1

    # Make sure the temporary folder exists
    assert os.path.exists(args.temp_folder), args.temp_folder

    find_cags(
        **args.__dict__
    )
