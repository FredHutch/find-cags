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
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from multiprocessing import Pool
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.stats import gmean


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
        assert normalization in ["median", "sum", "clr"]
        logging.info("Normalizing the abundance values by " + normalization)

    # Collect all of the abundance information in this single dict
    dat = {}

    # Keep track of the lowest value across all samples
    lowest_value = None

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
            d[gene_id_key]: np.float16(d[abundance_key])
            for d in sample_dat
        })

        # Normalize the abundance
        if normalization == "median":
            sample_dat = sample_dat / sample_dat.median()
        elif normalization == "sum":
            sample_dat = sample_dat / sample_dat.sum()
        elif normalization == "clr":
            sample_dat = (
                sample_dat / gmean(sample_dat)
            ).apply(np.log10).apply(np.float16)

        # Keep track of the lowest value across all samples
        if lowest_value is None:
            lowest_value = sample_dat.min()
        else:
            lowest_value = np.min([lowest_value, sample_dat.min()])

        # Add the data to the total
        dat[sample_name] = sample_dat

    logging.info("Formatting as a DataFrame")
    if normalization in ["median", "sum"] or normalization is None:
        dat = pd.DataFrame(dat).fillna(np.float16(0))
    else:
        assert normalization == "clr", normalization
        assert lowest_value is not None
        dat = pd.DataFrame(dat).fillna(lowest_value)

    logging.info("Read in data for {:,} genes across {:,} samples".format(
        dat.shape[0],
        dat.shape[1]
    ))

    return dat


def make_summary_abund_df(df, cags, singletons):
    """Make a DataFrame with the average value for each CAG, as well as the singletons."""
    assert len(set(cags.keys()) & set(df.index.values)) == 0, "Collision between protein/CAG names"

    summary_df = pd.concat([
        pd.DataFrame({
            cag_ix: df.loc[cag].mean()
            for cag_ix, cag in cags.items()
        }).T,
        df.loc[singletons]
    ])

    assert summary_df.shape[0] == len(cags) + len(singletons)
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
            raise Exception("Object cannot be written, no method for " + suffix)

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


def make_annoy_index(df, metric, n_trees=100):
    """Make the annoy index"""
    logging.info("Making the annoy index")
    index_handle = AnnoyIndex(df.shape[1], metric=metric)
    logging.info("Adding {:,} genes to the annoy index".format(df.shape[0]))
    for gene_ix in range(df.shape[0]):
        if gene_ix > 0 and gene_ix % 100000 == 0:
            logging.info("Added {:,} / {:,} genes to the index".format(
                gene_ix, df.shape[0]
            ))
        index_handle.add_item(gene_ix, df.iloc[gene_ix].values)
    logging.info("Building trees in the index")
    index_handle.build(n_trees)
    return index_handle


def get_single_cag(
    index_handle,
    gene_ix,
    n_samples,
    metric,
    max_dist,
    recursion_depth=0,
    max_recursion_depth=10
):
    """Get the CAGs for a single gene. This will always return a set including gene_ix"""

    n_neighbors_to_check = 10
    neighbors = index_handle.get_nns_by_item(
        gene_ix,
        n_neighbors_to_check,
        include_distances=True
    )
    # Make sure we've got all of the neighbors within the threshold
    while neighbors[1][-1] < max_dist:
        n_neighbors_to_check = n_neighbors_to_check * 2
        neighbors = index_handle.get_nns_by_item(
            gene_ix,
            n_neighbors_to_check,
            include_distances=True
        )

    # Now subset to those genes within the threshold
    neighbors = [
        ix
        for ix, d in zip(neighbors[0], neighbors[1])
        if d < max_dist
    ]
    if gene_ix not in neighbors:
        neighbors.append(gene_ix)

    # No neighbors were found
    if len(neighbors) == 1:
        return [gene_ix]

    # Make a distance matrix
    dm = pd.DataFrame({
        ix_1: {
            ix_2: index_handle.get_distance(ix_1, ix_2)
            for ix_2 in neighbors
        }
        for ix_1 in neighbors
    })
    
    # Pick a centroid as the gene with the lowest median distance
    centroid = dm.median().sort_values().index.values[0]

    # If the centroid is very close to the starting place, return this set
    if index_handle.get_distance(gene_ix, centroid) < max_dist / 10:
        return neighbors

    # If we've reached our maximum recursion depth, return this set
    if recursion_depth >= max_recursion_depth:
        return neighbors

    # Otherwise, pick the CAGs based on this new centroid
    new_neighbors = get_single_cag(
        index_handle,
        centroid,
        n_samples,
        metric,
        max_dist,
        recursion_depth=recursion_depth+1
    )
    # If the gene_ix is in the new set of neighbors, return that
    if gene_ix in new_neighbors:
        return new_neighbors
    # Otherwise, return the original set of neighbors
    return neighbors
    

def make_cags_with_ann(
    index_handle,
    metric,
    max_dist,
    n_genes,
    n_samples,
    gene_names
):
    """Make CAGs using the approximate nearest neighbor"""

    # Make a set with all of the genes that need to be clustered
    to_cluster = set(list(range(n_genes)))

    # Store the CAGs as a dict of lists
    cag_ix = 0
    cags = {}

    start_time = time.time()

    while len(to_cluster) > 0:
        if time.time() - start_time > 10:
            logging.info("Number of CAGs: {:,} -- Genes remaining to be clustered: {:,}".format(
                len(cags), len(to_cluster)
            ))
            start_time = time.time()
        # Get a single random gene index
        gene_ix = to_cluster.pop()
        to_cluster.add(gene_ix)

        # Get the genes linked to this one
        new_cag = get_single_cag(index_handle, gene_ix, n_samples, metric, max_dist)
        # Filter to those genes which haven't been clustered yet
        new_cag = list(set(new_cag) & to_cluster)

        # Now remove this set of genes from the list that needs to be clustered
        to_cluster -= set(new_cag)

        # Add CAGs with >1 member to the running list
        if len(new_cag) > 1:
            cags["cag_{}".format(cag_ix)] = [
                gene_names[ix]
                for ix in new_cag
            ]
            cag_ix += 1

    return cags


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
    iterations=1,
    test=False,
    chunk_size=1000
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

    assert isinstance(iterations, int)
    assert iterations >= 1
    assert iterations <= 999, "Don't iterate > 999 times"

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

    # If this is being run in testing mode, subset to 2,000 genes
    if test:
        logging.info("Running in testing mode, subset to 2,000 genes")
        df = df.head(2000)

    # Make the annoy index
    index_handle = make_annoy_index(df, metric)

    # Make CAGs using the approximate nearest neighbor
    cags = make_cags_with_ann(
        index_handle,
        metric,
        max_dist,
        df.shape[0],
        df.shape[1],
        df.index.values
    )

    # Get the singletons that aren't in any of the CAGs
    genes_in_cags = set([
        gene_id
        for cag_members in cags.values()
        for gene_id in cag_members
    ])
    all_genes = set(df.index.values)
    singletons = list(all_genes - genes_in_cags)

    logging.info("Number of genes in CAGs: {:,} / {:,}".format(
        len(genes_in_cags),
        len(all_genes)
    ))

    logging.info("Number of CAGs: {:,}".format(len(cags)))

    assert len(singletons) == len(all_genes) - len(genes_in_cags)

    # Now make a summary DF with the mean value for each combined CAG
    summary_df = make_summary_abund_df(df, cags, singletons)

    # Read in the logs
    logs = "\n".join(open(log_fp, "rt").readlines())

    # Return the results
    logging.info("Returning results to " + output_folder)
    try:
        return_results(
            df, summary_df, cags, logs, output_prefix, output_folder, temp_folder)
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
                        default="angular",
                        help="Distance metric calculation method, see spotify/annoy.")
    parser.add_argument("--normalization",
                        type=str,
                        default=None,
                        help="Normalization factor per-sample (median, sum, or clr).")
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
