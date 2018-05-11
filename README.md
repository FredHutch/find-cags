## Find Co-Abundant Groups of Genes

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
