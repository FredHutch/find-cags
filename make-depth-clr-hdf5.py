#!/usr/bin/env python3
"""Make a single HDF5 with all of the gene abundances for a set of samples"""

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
import logging
import argparse
import traceback
import subprocess
import numpy as np
import pandas as pd


def run_cmds(commands, retry=0, catchExcept=False, stdout=None):
    """Run commands and write out the log, combining STDOUT & STDERR."""
    logging.info("Commands:")
    logging.info(' '.join(commands))
    p = subprocess.Popen(commands,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    stdout, stderr = p.communicate()
    exitcode = p.wait()
    if stdout:
        logging.info("Standard output of subprocess:")
        for line in stdout.decode("utf-8").split('\n'):
            logging.info(line)
    if stderr:
        logging.info("Standard error of subprocess:")
        for line in stderr.split('\n'):
            logging.info(line)

    # Check the exit code
    if exitcode != 0 and retry > 0:
        msg = "Exit code {}, retrying {} more times".format(exitcode, retry)
        logging.info(msg)
        run_cmds(commands, retry=retry - 1)
    elif exitcode != 0 and catchExcept:
        msg = "Exit code was {}, but we will continue anyway"
        logging.info(msg.format(exitcode))
    else:
        assert exitcode == 0, "Exit code {}".format(exitcode)


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



def write_all_samples_to_hdf5(sample_sheet, hdf5_fp, results_key, abundance_key, gene_id_key):
    """Make a single HDF5 file with the abundance (raw depth and CLR) from all samples."""

    # Connect to the HDF5 store
    store = pd.HDFStore(hdf5_fp)

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

        # Format as a DataFrame
        df = pd.DataFrame([
            {
                gene_id_key: d[gene_id_key],
                abundance_key: np.float16(d[abundance_key])
            }
            for d in sample_dat
        ])
        logging.info("Read in {:,} genes for {}".format(df.shape[0], sample_name))

        # Calculate the median
        median_abund = df.loc[
            df[abundance_key] > 0,
            abundance_key
        ].median()

        # Calculate the CLR
        if abundance_key != "clr":
            logging.info("Calculating CLR")
            df["clr"] = (df[abundance_key] / median_abund).apply(np.log10).apply(np.float16)

        assert df.shape[0] == df.dropna().shape[0], "There are NaN values in the table for " + sample_name

        # Write the DataFrame to the HDF5 file
        logging.info("Writing {} to HDF5".format(sample_name))
        df.to_hdf(store, sample_name, format="table", complevel=5, data_columns=[gene_id_key])

    logging.info("Closing connection to " + hdf5_fp)
    store.close()

def return_results(hdf5_fp, log_fp, output_prefix, output_folder, temp_folder):
    """Write out all of the results to a file and copy to a final output directory"""

    # Make sure the output folder ends with a '/'
    if output_folder.endswith("/") is False:
        output_folder = output_folder + "/"

    if output_folder.startswith("s3://"):
        s3 = boto3.resource('s3')

    for fp in [hdf5_fp, log_fp]:
        
        if output_folder.startswith("s3://"):
            bucket, prefix = output_folder[5:].split("/", 1)

            # Make the full name of the destination key
            file_prefix = os.path.join(prefix, fp.split("/")[-1])

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

def make_depth_clr_hdf5(
    sample_sheet=None,
    output_prefix=None,
    output_folder=None,
    temp_folder="/scratch",
    results_key="results",
    abundance_key="depth",
    gene_id_key="id"
):
    # Make sure the temporary folder exists
    assert os.path.exists(temp_folder)

    # Make a new temp folder
    temp_folder = os.path.join(temp_folder, str(uuid.uuid4())[:8])
    os.mkdir(temp_folder)

    # Set up logging
    log_fp = os.path.join(temp_folder, output_prefix + ".txt")
    logFormatter = logging.Formatter(
        '%(asctime)s %(levelname)-8s [make-depth-clr-hdf5] %(message)s'
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

    logging.info("Temporary files being placed in " + temp_folder)

    # READING IN DATA

    # Read in the sample_sheet
    logging.info("Reading in the sample sheet from " + sample_sheet)
    try:
        sample_sheet = read_json(sample_sheet)
    except:
        exit_and_clean_up(temp_folder)

    # Make the abundance DataFrame
    logging.info("Making the HDF5 file with depth and CLR for every sample")
    hdf5_fp = os.path.join(temp_folder, output_prefix + ".hdf5")
    try:
        write_all_samples_to_hdf5(
            sample_sheet,
            hdf5_fp,
            results_key,
            abundance_key,
            gene_id_key
        )
    except:
        exit_and_clean_up(temp_folder)

    logging.info("Repacking the HDF5 file")
    try:
        run_cmds(["h5repack", "-i", hdf5_fp, "-o", hdf5_fp + '.repacked.hdf5', "-f", "GZIP=5"])
    except:
        exit_and_clean_up(temp_folder)
    try:
        run_cmds(["mv", hdf5_fp + '.repacked.hdf5', hdf5_fp])
    except:
        exit_and_clean_up(temp_folder)


    # Return the results
    logging.info("Returning results to " + output_folder)
    try:
        return_results(
            hdf5_fp,
            log_fp,
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
        description="""Make a single HDF5 with all of the gene abundances for a set of samples"""
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

    args = parser.parse_args(sys.argv[1:])

    # Sample sheet is in JSON format
    assert args.sample_sheet.endswith((".json", ".json.gz"))

    # Make sure the temporary folder exists
    assert os.path.exists(args.temp_folder), args.temp_folder

    make_depth_clr_hdf5(
        **args.__dict__
    )
