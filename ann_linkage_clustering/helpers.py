import boto3
from collections import defaultdict
import gzip
import io
import json
import logging
import numpy as np
import os
import pandas as pd
import shutil
import sys
import time
import traceback


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


def make_abundance_dataframe(sample_sheet, results_key, abundance_key, gene_id_key, normalization, min_samples):
    """Make a single DataFrame with the abundance (depth) from all samples."""

    assert isinstance(min_samples, int) and min_samples > 0
    assert min_samples <= len(sample_sheet)

    # Collect all of the abundance information in this single dict
    dat = {}

    # Iterate over each sample
    for sample_name, sample_path in sample_sheet.items():
        # Get the JSON for this particular sample
        sample_dat = read_json(sample_path)

        # If this is a dict, make sure that the key for the results is in this file
        if isinstance(sample_dat, dict):
            assert results_key in sample_dat

            # Subset down to the list of results
            sample_dat = sample_dat[results_key]

        assert isinstance(sample_dat, list)

        # Make sure that every element in the list has the indicated keys
        for d in sample_dat:
            assert abundance_key in d
            assert gene_id_key in d

        # Format as a dict, while normalizing abundances
        dat[sample_name] = normalize_abundance_vector(pd.Series({
            d[gene_id_key]: np.float16(d[abundance_key])
            for d in sample_dat
        }), normalization).to_dict()

    # Filter to the minimum number of samples
    logging.info(
        "Subsetting to genes found in at least {} samples".format(min_samples))

    # Count the number of times each gene is seen
    gene_counts = defaultdict(int)
    for sample_name in dat:
        for gene_id in dat[sample_name].keys():
            gene_counts[gene_id] += 1

    genes_to_keep = set([
        gene_id
        for gene_id, gene_count in gene_counts.items()
        if gene_count >= min_samples
    ])

    # Keep track of the number of genes filtered, and the time elapsed
    n_before_filtering = len(gene_counts)
    start_time = time.time()

    # Filter
    dat = {
        sample_name: {
            gene_id: gene_abund
            for gene_id, gene_abund in sample_dat.items()
            if gene_id in genes_to_keep
        }
        for sample_name, sample_dat in dat.items()
    }

    logging.info("{:,} / {:,} genes found in >= {:,} samples ({:,} seconds elapsed)".format(
        len(genes_to_keep),
        n_before_filtering,
        min_samples,
        round(time.time() - start_time, 2)
    ))


    logging.info("Converting to float16")
    dat = {
        sample_name: pd.Series(sample_dat).apply(np.float16)
        for sample_name, sample_dat in dat.items()
    }
    logging.info("Formatting as a DataFrame")
    dat = pd.DataFrame(dat)

    logging.info("Filling missing values")
    dat.fillna(
        np.min([dat.min().min(), np.float16(0)]),
        inplace=True
    )

    logging.info("Read in data for {:,} genes across {:,} samples".format(
        dat.shape[0],
        dat.shape[1]
    ))

    return dat


def normalize_abundance_vector(vec, normalization):
    """Normalize the raw depth values for a single sample."""

    assert normalization in ["median", "sum", "clr"]

    # Normalize the abundance

    if normalization == "median":
        # Divide by the median
        return vec / vec.median()

    elif normalization == "sum":
        # Divide by the sum
        return vec / vec.sum()

    elif normalization == "clr":
        # Divide by the median and take log10
        return (vec / vec.sum()).apply(np.log10)


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

    # Check to see if there are NaN values
    if df.isnull().any().any():
        logging.info("Found NaN values in the abundance dataframe")
        logging.info("Number of rows with NaN values: {:,}".format(
            df.isnull().any(axis=1).sum()
        ))
        logging.info("Number of columns with NaN values: {:,}".format(
            df.isnull().any(axis=0).sum()
        ))
        logging.info(df.loc[
            df.isnull().any(axis=1),
            df.isnull().any(axis=0)
        ].to_string())

        if normalization == "clr":
            logging.info("Filling in missing values with lowest value ({})".format(
                lowest_non_zero_value
            ))
            df.fillna(lowest_non_zero_value, inplace=True)
        else:
            logging.info("Filling missing values with 0")
            df.fillna(0, inplace=True)

    assert df.isnull().any().any() == False

    return df


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

    return dat
