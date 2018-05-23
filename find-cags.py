#!/usr/bin/env python3

import os
import io
import sys
import uuid
import time
import gzip
import json
import boto3
import shutil
import logging
import argparse
import traceback
import pandas as pd
from multiprocessing import Pool
from collections import defaultdict
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


def make_abundance_dataframe(sample_sheet, results_key, abundance_key, gene_id_key, normalization):
    """Make a single DataFrame with the abundance from all samples."""

    # Normalize each sample's data
    if normalization is not None:
        assert normalization in ["median", "sum"]
        logging.info("Normalizing the abundance values by " + normalization)

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
        sample_dat = pd.Series({
            d[gene_id_key]: d[abundance_key]
            for d in sample_dat
        })

        # Normalize the abundance
        if normalization == "median":
            sample_dat = sample_dat / sample_dat.median()
        elif normalization == "sum":
            sample_dat = sample_dat / sample_dat.sum()

        # Add the data to the total
        dat[sample_name] = sample_dat

    logging.info("Formatting as a DataFrame")
    dat = pd.DataFrame(dat).fillna(0)

    logging.info("Read in data for {:,} genes across {:,} samples".format(
        dat.shape[0],
        dat.shape[1]
    ))

    return dat


def find_pairwise_connections_worker(input_data):
    d1, d2, metric, max_dist = input_data

    df1, df1_index = d1
    df2, df2_index = d2

    d = cdist(df1, df2, metric=metric)

    return [
        (name1, name2)
        for ix1, name1 in enumerate(df1_index)
        for ix2, name2 in enumerate(df2_index)
        if d[ix1, ix2] <= max_dist
    ]


def generate_combinations(chunks, metric, max_dist):
    """Iteratively yield all pairwise combinations of these chunks."""

    # Calculate the total number of combinations
    n_possible_combinations = int(len(chunks) * (len(chunks) - 1) / 2)

    # Keep track of the number of combinations that have been yielded
    ix = 0
    # Keep track of the time it takes to yield successive combinations
    start_time = time.time()

    for chunk_1_ix in range(len(chunks)):
        for chunk_2_ix in range(chunk_1_ix + 1, len(chunks)):
            yield (chunks[chunk_1_ix], chunks[chunk_2_ix], metric, max_dist)
            ix += 1
            if ix % 1000 == 0:
                logging.info("Analyzed {:,} / {:,} combinations ({:,} seconds elapsed)".format(
                    ix, n_possible_combinations, 
                    round(time.time() - start_time, 2)
                ))
                start_time = time.time()

    logging.info("Analyzed {:,} / {:,} combinations ({:,} seconds elapsed)".format(
        ix, n_possible_combinations, 
        round(time.time() - start_time, 2)
    ))


def find_pairwise_connections(df, metric, max_dist, p, chunk_size=100000):
    gene_names = df.index.values
    # Break up the DataFrame into chunks
    chunks = [
        (df.values[n:(n + chunk_size)], gene_names[n:(n + chunk_size)])
        for n in range(0, df.shape[0], chunk_size)
    ]

    logging.info("Made {:,} groups of {:,} genes each".format(
        len(chunks), chunk_size
    ))

    connections = [    
        c
        for conn in p.imap_unordered(
            find_pairwise_connections_worker,
            generate_combinations(chunks, metric, max_dist)
        )
        for c in conn
    ]

    logging.info("Number of connections below the threshold: {:,}".format(
        len(connections)
    ))
    logging.info("Getting the list of singletons")

    # Calculate which genes are singletons with no connections
    singletons = set(gene_names) - set(
        [
            gene_id
            for connection in connections
            for gene_id in connection
        ]
    )
    return connections, singletons


def single_linkage_clustering(connections):
    gene_clusters = {}
    next_cluster_id = 0

    for g1, g2 in connections:
        if g1 in gene_clusters:
            if g2 in gene_clusters:
                # Already part of the same cluster
                if gene_clusters[g1] == gene_clusters[g2]:
                    pass

                # Need to combine two clusters
                else:
                    genes_to_combine = [
                        gene_id
                        for gene_id, cluster_id in gene_clusters.items()
                        if cluster_id in [gene_clusters[g1], gene_clusters[g2]]
                    ]
                    for gene_id in genes_to_combine:
                        gene_clusters[gene_id] = next_cluster_id
                    # Increment the cluster counter
                    next_cluster_id += 1

            # Add g2 to the cluster for g1
            else:
                gene_clusters[g2] = gene_clusters[g1]

        # g1 is not part of any cluster
        else:

            # g2 is already in a cluster
            if g2 in gene_clusters:
                # Assign g1 to the cluster for g1
                gene_clusters[g1] = gene_clusters[g2]
            
            # Neither gene is in a cluster yet
            else:
                gene_clusters[g1] = next_cluster_id
                gene_clusters[g2] = next_cluster_id
                # Increment the cluster counter
                next_cluster_id += 1

    # Reformat the clusters as a dict of sets
    cags = defaultdict(set)
    for gene_id, cluster_id in gene_clusters.items():
        cags[cluster_id].add(gene_id)

    # Return a dict of lists (numbering from 0)
    return {
        "cag_{}".format(ix): list(s)
        for ix, s in enumerate(cags.values())
    }


def make_summary_abund_df(df, cags, singletons):
    """Make a DataFrame with the average value for each CAG."""
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
        fp = os.path.join(temp_folder, output_prefix + suffix)
        if suffix.endswith(".feather"):
            obj.reset_index().to_feather(fp)
        elif suffix.endswith(".json.gz"):
            json.dump(obj, gzip.open(fp, "wt"))
        elif suffix.endswith(".txt"):
            with open(fp, "wt") as f:
                f.write(obj)
        else:
            raise Exception("Object cannot be written, no method for " + suffix)

        if output_folder.startswith("s3://"):
            bucket, prefix = output_folder[5:].split("/", 1)

            # Make the full name of the destination key
            file_prefix = prefix + output_prefix + suffix

            # Open a file handle to copy the file up to S3
            data = open(fp, 'rb')

            # Copy the file
            logging.info("Copying {} to {}/{}".format(
                fp,
                bucket,
                file_prefix
            ))
            s3.Bucket(bucket).put_object(Key=file_prefix, Body=data)


        else:
            # Copy as a local file
            logging.info("Copying {} to {}".format(
                fp, output_folder
            ))
            shutil.copy(fp, output_folder)


def find_cags(
    sample_sheet=None,
    output_prefix=None,
    output_folder=None,
    metric="euclidean",
    normalization=None,
    max_dist=0.1,
    temp_folder="/scratch",
    results_key="results",
    abundance_key="depth",
    gene_id_key="id",
    threads=1,
    min_samples=1,
    test=False,
    chunk_size=1000,
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

    # Read in the sample_sheet
    logging.info("Reading in the sample sheet from " + sample_sheet)
    try:
        sample_sheet = read_json(sample_sheet)
    except:
        exit_and_clean_up(temp_folder)

    # Make a pool of workers
    logging.info("Making a pool of {} workers".format(threads))
    p = Pool(threads)

    # Make the abundance DataFrame
    logging.info("Making the abundance DataFrame")
    try:
        df = make_abundance_dataframe(
            sample_sheet,
            results_key,
            abundance_key,
            gene_id_key,
            normalization
        )
    except:
        exit_and_clean_up(temp_folder)

    # If min_samples > 1, subset the genes
    if min_samples > 1:
        logging.info("Subsetting to genes found in at least {} samples".format(min_samples))

        # Keep track of the number of genes filtered, and the time elapsed
        n_before_filtering = df.shape[0]
        start_time = time.time()

        # Filter
        df = df.loc[(df > 0).sum(axis=1) >= min_samples]

        logging.info("{:,} / {:,} genes found in >= {:,} samples ({:,} seconds elapsed)".format(
            df.shape[0],
            n_before_filtering,
            min_samples,
            round(time.time() - start_time, 2)
        ))

    # If this is being run in testing mode, subset to 1,000 genes
    if test:
        logging.info("Running in testing mode, subset to 1,000 genes")
        df = df.head(1000)

    # Get the pairwise distances under the threshold
    logging.info("Finding pairwise connections with {} <= {}".format(metric, max_dist))
    try:
        connections, singletons = find_pairwise_connections(
            df,
            metric,
            max_dist,
            p,
            chunk_size=chunk_size)
    except:
        exit_and_clean_up(temp_folder)

    logging.info("Found {:,} pairwise connections, and {:,} singletons".format(
        len(connections),
        len(singletons)
    ))

    # Make the CAGs with single-linkage clustering
    logging.info("Making CAGs from {:,} pairwise connections".format(len(connections)))
    try:
        cags = single_linkage_clustering(connections)
    except:
        exit_and_clean_up(temp_folder)
    logging.info("Found {:,} CAGs".format(len(cags)))

    # Make a DataFrame summarizing the CAG and singleton abundances
    logging.info("Making summary DataFrame")
    try:
        summary_df = make_summary_abund_df(df, cags, singletons)
    except:
        exit_and_clean_up(temp_folder)

    # Read in the logs
    logs = "\n".join(open(log_fp, "rt").readlines())

    # Return the results
    logging.info("Returning results to " + output_folder)
    try:
        return_results(df, summary_df, cags, logs, output_prefix, output_folder, temp_folder)
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
    parser.add_argument("--metric",
                        type=str,
                        default="euclidean",
                        help="Distance metric calculation method, see scipy.spatial.distance.")
    parser.add_argument("--normalization",
                        type=str,
                        default=None,
                        help="Normalization factor per-sample (median or sum).")
    parser.add_argument("--max-dist",
                        type=float,
                        default=0.01,
                        help="Maximum distance for single-linkage clustering.")
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
    parser.add_argument("--chunk-size",
                        type=int,
                        default=100000,
                        help="Size of chunks to break abundance table into.")
    parser.add_argument("--min-samples",
                        type=int,
                        default=1,
                        help="Filter genes by the number of samples they are found in.")
    parser.add_argument("--test",
                        action="store_true",
                        help="Run in testing mode and only process a subset of 1,000 genes.")

    args = parser.parse_args(sys.argv[1:])

    # Sample sheet is in JSON format
    assert args.sample_sheet.endswith((".json", ".json.gz"))

    # Normalization factor is absent, 'median', or 'sum'
    assert args.normalization in [None, "median", "sum"]

    # max-dist is >=0
    assert args.max_dist >= 0

    assert args.threads >= 1

    assert args.min_samples >= 1

    # Make sure the temporary folder exists
    assert os.path.exists(args.temp_folder)

    find_cags(
        **args.__dict__
    )
