#!/usr/bin/env python3

import os
import sys
import uuid
import time
import gzip
import json
import shutil
import logging
import argparse
import traceback


def exit_and_clean_up(temp_folder):
    """Log the error messages and delete the temporary folder."""
    # Capture the traceback
    logging.info("There was an unexpected failure")
    exc_type, exc_value, exc_traceback = sys.exc_info()
    for line in traceback.format_tb(exc_traceback):
        logging.info(line.encode("utf-8"))

    # Delete any files that were created for this sample
    logging.info("Removing temporary folder: " + temp_folder)
    shutil.rmtree(temp_folder)

    # Exit
    logging.info("Exit type: {}".format(exc_type))
    logging.info("Exit code: {}".format(exc_value))
    sys.exit(exc_value)


def read_json(fp):
    return json.load(open(fp, "rt"))


def make_abundance_dataframe(sample_sheet):
    pass


def find_pairwise_connections(df, metric, max_dist):
    return connections, singletons


def single_linkage_clustering(connections):
    pass


def make_summary_abund_df(df, cags, singletons):
    pass


def return_results(df, summary_df, cags, log_fp, output_prefix, output_folder):
    pass


def find_cags(
    sample_sheet=None,
    output_prefix=None,
    output_folder=None,
    metric="euclidean",
    normalization=None,
    max_dist=0.1,
    temp_folder="/scratch"
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

    # Make the abundance DataFrame
    logging.info("Making the abundance DataFrame")
    try:
        df = make_abundance_dataframe(sample_sheet)
    except:
        exit_and_clean_up(temp_folder)

    # Normalize the abundance DataFrame
    if normalization is not None:
        assert normalization in ["median", "sum"]
        logging.info("Normalizing the abundance values by " + normalization)

        if normalization == "median":
            try:
                df = df / df.median()
            except:
                exit_and_clean_up(temp_folder)
        elif normalization == "sum":
            try:
                df = df / df.sum()
            except:
                exit_and_clean_up(temp_folder)

    # Get the pairwise distances under the threshold
    logging.info("Finding pairwise connections with {} <= {}".format(metric, max_dist))
    try:
        connections, singletons = find_pairwise_connections(df, metric, max_dist)
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

    # Return the results
    logging.info("Returning results to " + output_folder)
    try:
        return_results(df, summary_df, cags, log_fp, output_prefix, output_folder)
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
                        default=0.1,
                        help="Maximum distance for single-linkage clustering.")
    parser.add_argument("--temp-folder",
                        type=str,
                        default="/scratch",
                        help="Folder for temporary files.")

    args = parser.parse_args(sys.argv[1:])

    # Normalization factor is absent, 'median', or 'sum'
    assert args.normalization in [None, "median", "sum"]

    # max-dist is >=0
    assert args.max_dist >= 0

    # Make sure the temporary folder exists
    assert os.path.exists(args.temp_folder)

    find_cags(
        **args.__dict__
    )
