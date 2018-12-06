## Find Co-Abundant Groups of Genes

[![Docker Repository on Quay](https://quay.io/repository/fhcrc-microbiome/find-cags/status "Docker Repository on Quay")](https://quay.io/repository/fhcrc-microbiome/find-cags)

#### Purpose

Analyze gene abundance data from a large set of samples and calculate
which sets of genes are found at a similar abundance across all samples.
Those genes are expected to be biologically linked, such as the case of
metagenomic analysis via whole-genome shotgun sequences, where genes
from the same genome tend to be found at a similar abundance.


#### Code Availability

The code in this repository is provided in two different formats. There
is a library of Python code (`ann_linkage_clustering` in PyPI) that can
be used to make CAGs directly from a Pandas DataFrame. There is also a
Docker image that is intended to be run with the script `find-cags.py`.
The documentation below describes the end-to-end workflow that is available
with that Docker image and the single wrapper script. 


#### Input Data Format

It is assumed that all input data will be in JSON format (gzip optional).
The pertinent data for each individual sample is an abundance metric for
each sample. The input file must contain a `list` in which each element
is a `dict` that contains the gene ID with one `key` and the abundance
metric with another `key`. 

For initial development we will assume that each input file is a single
`dict`, with the results located at a single `key` within that `dict`. 
In the future we may end up supporting more flexibility in extracting
results from files with different structures, but for the first pass we'll
just go with this.

Therefore the features that must be specified by the user are:

  * Key for the list of gene abundances within the JSON (e.g. "results")
  * Key for the gene_id within each element of the list (e.g. "id")
  * Key for the abundance metric within each element (e.g. "depth")

Here is an example of what that might look like in JSON format:

```json
{
  "results": [
    {
      "id": "gene_1",
      "depth": 1.1
    },
    {
      "id": "gene_2",
      "depth": 0.2
    },
    {
      "id": "gene_3",
      "depth": 3000.015
    },
  ],
  "logs": [
    "any other data",
    "that you would like",
    "to include in this file is just fine."
  ]
}
```

**NOTE**: All abundance metric values must be >= 0


#### Running from any DataFrame

If you have any other format of data, you can use this code to find CAGs as well.
The big difference is that this script does some data normalization that is very
helpful. For example, if you are using cosine distance, it's best to have the value
indicating absence to be zero. So if you are using the centered log-ratio (clr)
normalization approach, you really need to set a standard cuttoff across all samples,
trim the lowest value to that, and then set that lowest value to zero. This is all
done automatically by `find-cags.py`, but you can absolutely use the same functions
to make CAGs with any other input data format or normalization approach.


You can follow the workflow in the `find-cags.py` script, which basically follows
this workflow (assuming that `df` is your DataFrame of abundance data, with genes
in rows and samples in columns):

```python
from multiprocessing import Pool
from ann_linkage_clustering.lib import make_cags_with_ann
from ann_linkage_clustering.lib import iteratively_refine_cags
from ann_linkage_clustering.lib import make_nmslib_index

# Maximum distance threshold (use any value)
max_dist=0.2

# Distance metric (only 'cosine' is supported)
distance_metric="cosine"

# Multiprocessing pool (pick any number of threads, in this case `1`)
threads = 1
pool = Pool(threads)

# Linkage type (only `average` is fully supported)
linkage_type = "average"

# Make the ANN index
index = make_nmslib_index(df)

# Make the CAGs in the first round
cags = make_cags_with_ann(
    index,
    max_dist,
    df,
    pool,
    threads=threads,
    distance_metric=distance_metric,
    linkage_type=linkage_type
)

# Iteratively refine the CAGs (this is the part that is hardedcoded to 
# use average linkage clustering, while the step above could technically
# use any of `complete`, `single`, `average`, etc.)
iteratively_refine_cags(
    cags,
    df.copy(),
    max_dist,
    distance_metric=distance_metric,
    linkage_type=linkage_type,
    threads=threads
)
```

At the end of all of that, the `cags` object is a dictionary containing
all of the identified groups.


#### Sample Sheet

To link individual files with sample names, the user will specify a
sample sheet, which is a JSON file formatted as a `dict`, with sample
names as key and file locations as values. 


#### Data Locations

At the moment we will support data found in either the local file system 
or AWS S3.


#### Test Dataset

For testing, I will use a set of JSONs which contain the abundance of
individual genes for a set of microbiome samples. That data is found in the
`tests/` folder. There is also a JSON file indicating which sample goes
with which file, which is formatted as a simple dict (keys are sample names
and values are file locations) and located in `tests/sample_sheet.json`.


#### Normalization

The `--normalize` metric accepts three values, `clr`, `median`, and `sum`. In each case
the abundance metric for each gene within each sample is divided by either
the `median` or the `sum` of the abundance metrics for all genes within that
sample. When calculating the `median`, only genes with non-zero abundances
are considered. For `clr`, each value is divided by the geometric mean for the
sample, and then the log10 is taken. All zero values are filled with the minimum
value for the entire dataset (so that they are equal across samples, and not
sensitive to sequencing depth).


#### Approximate Nearest Neighbor

The Approximate Nearest Neighbor algorithm as implemented by 
[nmslib](https://nmslib.github.io/nmslib/index.html) is being used to create the CAGs.
This implementation has a high performance in an independent 
[benchmark](http://ann-benchmarks.com/).


#### Distance Metric

The distance metric is now hard-coded to be the cosine similarity. This is limited by the
available functionality of ANN in `nmslib`, and therefore has been standardized to the
other parts of the codebase as well.


#### Refinements

After finding CAGs, the algorithm will iteratively join CAGs that are very similar to each
other in aggregate. This increases the fidelity of the final CAGs and mitigates some of the
sensitivity limitations of ANN.


#### Invocation

```
usage: find-cags.py [-h] --sample-sheet SAMPLE_SHEET --output-prefix
                    OUTPUT_PREFIX --output-folder OUTPUT_FOLDER
                    [--normalization NORMALIZATION] [--max-dist MAX_DIST]
                    [--temp-folder TEMP_FOLDER] [--results-key RESULTS_KEY]
                    [--abundance-key ABUNDANCE_KEY]
                    [--gene-id-key GENE_ID_KEY] [--threads THREADS]
                    [--min-samples MIN_SAMPLES] [--clr-floor CLR_FLOOR]
                    [--test]

Find a set of co-abundant genes

optional arguments:
  -h, --help            show this help message and exit
  --sample-sheet SAMPLE_SHEET
                        Location for sample sheet (.json[.gz]).
  --output-prefix OUTPUT_PREFIX
                        Prefix for output files.
  --output-folder OUTPUT_FOLDER
                        Folder to place results. (Supported: s3://, or local
                        path).
  --normalization NORMALIZATION
                        Normalization factor per-sample (median, sum, or clr).
  --max-dist MAX_DIST   Maximum cosine distance for clustering.
  --temp-folder TEMP_FOLDER
                        Folder for temporary files.
  --results-key RESULTS_KEY
                        Key identifying the list of gene abundances for each
                        sample JSON.
  --abundance-key ABUNDANCE_KEY
                        Key identifying the abundance value for each element
                        in the results list.
  --gene-id-key GENE_ID_KEY
                        Key identifying the gene ID for each element in the
                        results list.
  --threads THREADS     Number of threads to use.
  --min-samples MIN_SAMPLES
                        Filter genes by the number of samples they are found
                        in.
  --clr-floor CLR_FLOOR
                        Set a minimum CLR value, 'auto' will use the largest
                        minimum value.
  --test                Run in testing mode and only process a subset of 2,000
                        genes.

  ```
