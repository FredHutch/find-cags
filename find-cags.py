#!/usr/bin/env python3

import os
import io
import sys
import uuid
import copy
import time
import gzip
import json
import boto3
import shutil
import nmslib
import logging
import argparse
import traceback
import numpy as np
import pandas as pd
from scipy.stats import gmean
from multiprocessing import Pool
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster


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


def make_nmslib_index(df):
    """Make the HNSW index"""
    logging.info("Making the HNSW index")
    index = nmslib.init(method='hnsw', space='cosinesimil')

    logging.info("Adding {:,} genes to the nmslib index".format(df.shape[0]))
    index.addDataPointBatch(df.values)
    logging.info("Making the index")
    index.createIndex({'post': 2, "M": 200}, print_progress=False)

    return index


def get_gene_neighborhood(input_data):
    """
    Return the gene neighborhood for a particular gene.
    
    `nearest_neighbors`: A dict with the set of nearest neighbors, keyed by gene.
    `central_gene`: The primary gene to consider.
    `genes_remaining`: The set of genes that are remaining at this point of the analysis.

    Return the second order neighbors of the central gene (as a set).
    """

    central_gene, nearest_neighbors, genes_remaining = input_data

    if central_gene not in nearest_neighbors:
        return set()

    # Get the first and second order connections
    neighborhood = set([central_gene])
    for first_order_connection in nearest_neighbors[central_gene]:
        if first_order_connection in nearest_neighbors:
            neighborhood.add(first_order_connection)
            neighborhood |= nearest_neighbors[first_order_connection]
    neighborhood = neighborhood & genes_remaining

    return neighborhood


def complete_linkage_clustering(input_data):
    """Return the largest complete linkage group in this set of genes."""

    df, max_dist, distance_metric, linkage_type = input_data

    if df.shape[0] == 1:
        return set(df.index.values)

    # Compute the linkage (this will also calculate the pairwise distances)
    l = linkage(
        df,
        method=linkage_type,
        metric=distance_metric,
    )
    # Get the flat clusters
    flat_clusters = fcluster(l, max_dist, criterion="distance")

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

    # Return the largest cluster (or the one that is tied for the largest)
    return set(clusters[0])


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

    # Get the nearest neighbors for every gene
    logging.info("Starting with the closest {:,} neighbors for all genes".format(
        starting_n_neighbors))

    # Keep track of the genes that are singletons at this early stage
    singletons = set()

    # Keep track of which genes remain to be clustered
    genes_remaining = set()

    # Format the nearest neighbors as a dict of sets
    nearest_neighbors = {}

    # Iterate over every gene
    for gene_ix, gene_neighbors in enumerate(index.knnQueryBatch(
        df.values,
        k=starting_n_neighbors,
        num_threads=threads
    )):
        gene_name = df.index.values[gene_ix]
        nn = set([
            df.index.values[neighbor_ix]
            for neighbor_ix, neighbor_distance in zip(
                gene_neighbors[0],
                gene_neighbors[1]
            )
            if neighbor_distance <= max_dist
        ])
        if len(nn) > 1:
            nearest_neighbors[gene_name] = nn
            genes_remaining.add(gene_name)
        else:
            singletons.add(gene_name)

    logging.info("Formatted nearest neighbors for every input gene")
    logging.info("Genes with neighbors: {:,} -- Singletons: {:,}".format(
        len(nearest_neighbors), len(singletons)
    ))

    # Keep track of the number of genes that were input
    n_genes_input = df.shape[0]

    # Make sure that the number of genes adds up
    assert n_genes_input == len(genes_remaining) + len(singletons)

    # Find CAGs greedily, taking the complete linkage group for each gene in random order
    cags = {}
    cag_ix = 0

    # Keep clustering until everything is gone
    while len(genes_remaining) > 0:

        # Get some of the advantages of parallel processing

        # Find the linkage clusters in parallel
        for linkage_cluster in pool.imap_unordered(
            complete_linkage_clustering,
            [
                (
                    df.reindex(index=list(get_gene_neighborhood((
                        central_gene,
                        nearest_neighbors,
                        genes_remaining
                    )))),
                    max_dist, 
                    distance_metric, 
                    linkage_type
                )
                for central_gene in np.random.choice(list(genes_remaining), threads * 4)
            ]
        ):
            # Make sure that every member of this cluster still needs to be clustered
            if len(linkage_cluster) > 0 and linkage_cluster <= genes_remaining and len(genes_remaining) > 0:

                logging.info("Adding a CAG with {:,} members, {:,} genes unclustered".format(
                    len(linkage_cluster),
                    len(genes_remaining) - len(linkage_cluster)
                ))

                cags[cag_ix] = list(linkage_cluster)
                cag_ix += 1

                # Remove these genes from further consideration
                genes_remaining = genes_remaining - linkage_cluster
                df.drop(index=list(linkage_cluster), inplace=True)
                for gene_name in list(linkage_cluster):
                    if gene_name in nearest_neighbors:
                        del nearest_neighbors[gene_name]
                
    # Add in CAGs for the singletons
    for gene_name in list(singletons):
        cags[cag_ix] = list(gene_name)
        cag_ix += 1

    # Basic sanity checks
    assert all([len(v) > 0 for v in cags.values()])
    assert sum(map(len, cags.values())) == n_genes_input, (sum(
        map(len, cags.values())), n_genes_input)

    return cags


def join_overlapping_cags(cags, df, max_dist, distance_metric="cosine", linkage_type="average"):
    """Check to see if any CAGs are overlapping. If so, join them and combine the result."""

    # Make a dict linking each gene to its CAG - omitting singletons
    cag_dict = {
        gene_id: cag_id
        for cag_id, gene_id_list in cags.items()
        for gene_id in gene_id_list
        if len(gene_id_list) > 1
    }

    if len(cag_dict) == 1:
        logging.info("There is only 1 non-singleton CAG -- no need to cluster again")
        return

    # Make a DF with the mean abundance of each CAG
    logging.info("Computing mean abundances for {:,} CAGs and {:,} genes (omitting singletons)".format(
        len(set(cag_dict.values())), len(cag_dict)
    ))
    cag_df = pd.concat([
        df,
        pd.DataFrame({"cag": pd.Series(cag_dict)})
    ], axis=1).groupby("cag").mean()

    # Find the flat linkage clusters of the CAGs
    logging.info("Finding groups of CAGs")
    cag_clusters = fcluster(
        linkage(
            cag_df,
            method=linkage_type,
            metric=distance_metric,
        ), 
        max_dist, 
        criterion="distance"
    )

    # Iterate over the groups of CAGs
    for cag_cluster_ix in list(set(cag_clusters)):
        # Check to see if multiple CAGs were grouped together
        if (cag_clusters == cag_cluster_ix).sum() > 1:

            # Get the set of CAGs that will be regrouped
            cags_to_regroup = [
                cag_name
                for cag_name, ix in zip(
                    cag_df.index.values,
                    cag_clusters
                )
                if ix == cag_cluster_ix
            ]

            # Get the set of genes to regroup
            genes_to_regroup = [
                gene_name
                for cag_name in cags_to_regroup
                for gene_name in cags[cag_name]
            ]

            logging.info("Regrouping a set of {:,} CAGs and {:,} genes".format(
                len(cags_to_regroup),
                len(genes_to_regroup)
            ))

            # Make new groups
            new_cags = fcluster(
                linkage(
                    df.reindex(index=genes_to_regroup),
                    method=linkage_type,
                    metric=distance_metric,
                ),
                max_dist,
                criterion="distance"
            )

            logging.info("Made a new set of {:,} CAGs".format(
                len(set(new_cags))
            ))

            # Remove the old CAGs
            for cag_name in cags_to_regroup:
                del cags[cag_name]

            # Add the new CAGs
            new_cag_ix = 0
            for cag_name, genes_in_cag in pd.Series(genes_to_regroup).groupby(new_cags):
                while new_cag_ix in cags:
                    new_cag_ix += 1
                cags[new_cag_ix] = genes_in_cag.values.tolist()
                new_cag_ix += 1


def genes_are_overlapping(df1, df2, max_dist, distance_metric="cosine", linkage_type="average"):
    """Check to see if the two sets of genes are completely overlapping."""
    
    # Compute the linkage (this will also calculate the pairwise distances)
    l = linkage(
        pd.concat([df1, df2]),
        method=linkage_type,
        metric=distance_metric,
    )
    # Get the flat clusters
    flat_clusters = fcluster(l, max_dist, criterion="distance")

    # Return True if there is only a single cluster at this threshold
    return len(set(flat_clusters)) == 1


def iteratively_refine_cags(cags, df, max_dist, distance_metric="cosine", linkage_type="average"):
    """Refine the CAGs by merging all groups that are overlapping."""

    # Repeat until all overlapping CAGs are merged, maxing out after a few iterations
    n_cags_previously = len(cags) + 1
    n_iters = 0
    while n_cags_previously > len(cags):

        n_cags_previously = len(cags)

        # Merge any overlapping CAGs
        join_overlapping_cags(
            cags,
            df,
            max_dist,
            distance_metric=distance_metric,
            linkage_type=linkage_type
        )

        logging.info("Merging together {:,} CAGs yielded {:,} CAGs".format(
            n_cags_previously, len(cags)
        ))

        n_iters += 1
        if n_iters >= 5:
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
            linkage_type=linkage_type
        )
    except:
        exit_and_clean_up(temp_folder)

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
