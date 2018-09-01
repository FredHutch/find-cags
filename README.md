## Find Co-Abundant Groups of Genes

[![Docker Repository on Quay](https://quay.io/repository/fhcrc-microbiome/find-cags/status "Docker Repository on Quay")](https://quay.io/repository/fhcrc-microbiome/find-cags)

#### Purpose

Analyze gene abundance data from a large set of samples and calculate
which sets of genes are found at a similar abundance across all samples.
Those genes are expected to be biologically linked, such as the case of
metagenomic analysis via whole-genome shotgun sequences, where genes
from the same genome tend to be found at a similar abundance.


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

**NOTE**: All abundance metric values must be >= 0

#### Sample Sheet

To link individual files with sample names, the user will specify a
sample sheet, which is a JSON file formatted as a `dict`, with sample
names as key and file locations as values. 


#### Data Locations

At the moment we will support data found in (a) the local file system 
or (b) AWS S3.


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

The distance metric is now hard-coded to be the cosine similarity.


#### Iterations

The algorithm can be run iteratively, which may help with some of the noise and
imperfect recall of the ANN algorithm. Use the `--iterations` flag to set the number
of iterations to attemp (less than 1,000).


#### Invocation

```
usage: find-cags.py [-h] --sample-sheet SAMPLE_SHEET --output-prefix
                    OUTPUT_PREFIX --output-folder OUTPUT_FOLDER
                    [--normalization NORMALIZATION] [--max-dist MAX_DIST]
                    [--temp-folder TEMP_FOLDER] [--results-key RESULTS_KEY]
                    [--abundance-key ABUNDANCE_KEY]
                    [--gene-id-key GENE_ID_KEY] [--iterations ITERATIONS]
                    [--threads THREADS] [--min-samples MIN_SAMPLES] [--test]

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
  --iterations ITERATIONS
                        Number of iterations to run.
  --threads THREADS     Number of threads to use.
  --min-samples MIN_SAMPLES
                        Filter genes by the number of samples they are found
                        in.
  --test                Run in testing mode and only process a subset of 2,000
                        genes.
  ```

#### Helper Scripts

```
usage: make-cag-feather.py [-h] --cag-json-fp CAG_JSON_FP --sample-sheet
                           SAMPLE_SHEET --output-prefix OUTPUT_PREFIX
                           --output-folder OUTPUT_FOLDER
                           [--normalization NORMALIZATION]
                           [--temp-folder TEMP_FOLDER]
                           [--results-key RESULTS_KEY]
                           [--abundance-key ABUNDANCE_KEY]
                           [--gene-id-key GENE_ID_KEY]

Read in a set of samples and make a feather file with the CAG abundances

optional arguments:
  -h, --help            show this help message and exit
  --cag-json-fp CAG_JSON_FP
                        Location for CAGs (.json[.gz]).
  --sample-sheet SAMPLE_SHEET
                        Location for sample sheet (.json[.gz]).
  --output-prefix OUTPUT_PREFIX
                        Prefix for output files.
  --output-folder OUTPUT_FOLDER
                        Folder to place results. (Supported: s3://, or local
                        path).
  --normalization NORMALIZATION
                        Normalization factor per-sample (median, sum, or clr).
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
```
