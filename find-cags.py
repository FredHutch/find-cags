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
from scipy.spatial.distance import cdist


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


def greedy_complete_linkage_clustering(input_data):
    """Use a greedy approach to get the complete linkage group connected to the central gene."""

    central_gene, central_gene_ix, gene_names, df, max_dist = input_data

    if central_gene_ix % 1000 == 0:
        logging.info("Processing gene {:,}".format(central_gene_ix))

    # Singletons
    if len(gene_names) == 1:
        return central_gene, gene_names

    # Take the central gene out of the list of gene names
    gene_names = [g for g in gene_names if g != central_gene]

    # Calculate the distance of all genes against the central gene
    central_gene_dists = {
        g: d
        for g, d in zip(
            gene_names,
            cdist(df.loc[[central_gene]],
                  df.loc[gene_names], metric="cosine")[0]
        )
    }

    # Sort the genes by their closeness to the central one
    gene_names = sorted(gene_names, key=central_gene_dists.get)

    # Start the cluster
    cluster = [central_gene]

    # Add to the cluster until we're out
    for g in gene_names:

        # Skip genes that aren't close enough to the central gene
        if central_gene_dists[g] >= max_dist:
            continue

        # Keep track of whether this should be added to the cluster
        add_to_cluster = True

        # Check the distance against every member of the cluster
        for d in cdist(df.loc[[g]], df.loc[cluster], metric="cosine")[0]:
            if d >= max_dist:
                add_to_cluster = False
                break

        # Only add the gene if it's within the threshold for all previous genes
        if add_to_cluster:
            cluster.append(g)

    return central_gene, cluster


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def make_cags_with_ann(
    index,
    max_dist,
    df,
    pool,
    threads=1,
    starting_n_neighbors=1000
):
    """Make CAGs using the approximate nearest neighbor"""

    # Get the nearest neighbors for every gene
    logging.info("Starting with the closest {:,} neighbors for all genes".format(
        starting_n_neighbors))
    nearest_neighbors = index.knnQueryBatch(
        df.values, k=starting_n_neighbors, num_threads=threads)

    # Get the candidate groups for every gene
    logging.info("Making candidate groups for every gene")
    candidate_groups = []

    didnt_find_self = 0
    start_time = time.time()
    for gene_ix, gene_name in enumerate(df.index.values):
        if time.time() - start_time > 30:
            logging.info("Processed {:,} / {:,} genes".format(
                gene_ix, df.shape[0]
            ))
            start_time = time.time()

        # Add all genes with distance < threshold to the first order connections
        first_order_gene_neighbors = [
            ix
            for ix, d in zip(
                nearest_neighbors[gene_ix][0],
                nearest_neighbors[gene_ix][1]
            )
            if d < max_dist
        ]

        if gene_ix not in first_order_gene_neighbors:
            didnt_find_self += 1
            first_order_gene_neighbors.append(gene_ix)

        candidate_groups.append([
            df.index.values[ix]
            for ix in first_order_gene_neighbors
        ])

    logging.info("Didn't recall self for {:,} / {:,} genes".format(
        didnt_find_self, df.shape[0]
    ))

    # Now find the optimized CAGs with complete linkage
    logging.info("Finding optimized CAGs")
    all_cags = {}
    # Iterate over chunks of the data
    for gene_ix_list in chunks(range(df.shape[0]), threads * 10):
        all_cags = {
            **all_cags,
            **{
                gene_name: gene_group
                for gene_name, gene_group in pool.map(
                    greedy_complete_linkage_clustering,
                    [
                        (
                            df.index.values[gene_ix],
                            gene_ix,
                            candidate_groups[gene_ix],
                            df.loc[candidate_groups[gene_ix]],
                            max_dist
                        )
                        for gene_ix in gene_ix_list
                    ]
                )
            }
        }

    logging.info("Done performing complete linkage clustering")

    # Order the genes by the number of neighbors (descending)
    n_neighbors_per_gene = {
        gene_id: len(neighbors)
        for gene_id, neighbors in all_cags.items()
    }
    all_genes = sorted(
        df.index.values, key=n_neighbors_per_gene.get, reverse=True)

    # Now find the smallest set of CAGs which cover the largest number of genes
    logging.info("Picking non-overlapping CAGs from the complete set")
    # Store the CAGs as a dict of lists
    cags = {}
    cag_ix = 1

    # Make a set with all of the genes that need to be clustered
    to_cluster = set(all_genes)

    start_time = time.time()

    for gene_name in all_genes:
        # Skip genes that are already in a CAG
        if gene_name not in to_cluster:
            continue

        # Get the genes linked to this one
        new_cag = all_cags[gene_name]
        # Filter to those genes which haven't been clustered yet
        new_cag = list(set(new_cag) & to_cluster)

        # Now remove this set of genes from the list that needs to be clustered
        to_cluster -= set(new_cag)

        # Add CAGs to the running list
        cags["cag_{}".format(cag_ix)] = new_cag
        cag_ix += 1

    # Basic sanity checks
    assert all([len(v) > 0 for v in cags.values()])
    assert sum(map(len, cags.values())) == df.shape[0], (sum(
        map(len, cags.values())), df.shape[0])

    return cags


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

    # If min_samples > 1, subset the genes
    if min_samples > 1:
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

    # Apply the clr_floor parameter, if applicable
    if clr_floor is not None and normalization == "clr":
        logging.info("Applying the CLR floor: {}".format(clr_floor))
        df = df.applymap(lambda v: v if v > clr_floor else clr_floor)

    # Make sure that the lowest abundance is 0 (for clustering)
    logging.info("Setting the lowest abundance as 0")
    df = df - df.min().min()

    # Get rid of any genes that do not rise above the floor
    ix = df.max(axis=1) > df.min().min()
    logging.info("{:,} / {:,} genes are found above an abundance of {:,}".format(
        ix.sum(),
        ix.shape[0],
        df.min().min()
    ))
    df = df.loc[ix]

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
            df,
            pool,
            threads=threads
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
                        type=float,
                        default=None,
                        help="Set the floor for the CLR as -1 (gmean / 10.).")
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
