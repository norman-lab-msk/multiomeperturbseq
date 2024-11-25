import os
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import argparse
import datetime

from collections import Counter
from tqdm import tqdm
import csv
from joblib import Parallel, delayed

import scanpy as sc
import anndata as ad

from Bio.SeqIO.QualityIO import FastqGeneralIterator, FastqPhredIterator
from Bio.Seq import Seq

from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression
import hdbscan
import re
import pysam

import sys
sys.path.append("/scratch/eli/")
from isotonic_barcode_calling import *

plt.rcdefaults()

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--features", default = '/data/norman/eli/T7/202404_SIRLOIN_multiome/library_QC/epi16_library_build.csv', dest = 'EXPECTED_FEATURES', help = 'feature reference with guide sequences and targets')
parser.add_argument("--id", dest = "run_id", help = 'provide a unique run id for output filenames')
parser.add_argument("--samplesheet", dest = 'samplesheet', help = 'provide a sample sheet csv with 4 columns: sample_name, gex_path, cellranger_name, and fastqs. cellranger_name is the --sample argument when cellranger is called on guide lib fastqs, so must match the first part of the fastq filenames. fastqs are the comma-separated paths to folders containing the guide lib fastqs')
parser.add_argument("-v", dest = 'verbose', action='store_true')
parser.add_argument("--workdir", dest = "WORKING_DIRECTORY", default = os.getcwd(), help = 'working directory')
parser.add_argument("--entropy", dest = 'entropy', action = 'store_true', help = 'whether to add guide UMI entropy for id calling')
parser.add_argument("--call_only", dest = 'call_only', action = 'store_true', help = 'call identities from existing cellranger output; assumes that umi csvs are in the working directory')
parser.add_argument("--umi_cutoff", dest = 'umi_cutoff', default = 3, help = 'umi cutoff for top IDs (this is different than MIN_LOG_UMI for calling!)')
parser.add_argument("--eps", dest = 'eps', default = 0.5, help = 'epsilon for id calling', type = float)
parser.add_argument("--sample_reads", dest = 'sample_reads', default = None, type = float, help = 'number of reads to downsample to, if any')
parser.add_argument("--preflight", dest = 'preflight', action = 'store_true', help = 'print cellranger command and exit')
args = parser.parse_args()

# define args that shouldn't be changed
BARCODE_SEQUENCE_CONTEXT = 'gtgttttgagactataaGtatcccttggagaaCCAcctTGTTGNNNNNNNNNNNNNNNNNNNNGTTTaAGAGCTAaGCTGGAAACAGCATAGCAAGTTtAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGC'
CELLRANGER_PATH = '/scratch/eli/cellranger-7.1.0/cellranger'
TRANSCRIPTOME_PATH = '/fscratch/eli/genomes/refdata-gex-GRCh38-2020-A'
MATCHING_THRESHOLD = 95
NUM_WORKERS = 6
NUM_CHUNKS = 1
MIN_LOG_UMI = 1.5

# define samples
try:
    samplesheet = pd.read_csv(args.samplesheet)
    sample_dict = dict(zip(samplesheet['sample_name'].tolist(), samplesheet['gex_path'].tolist()))
    cellranger_sample_names = dict(zip(samplesheet['sample_name'].tolist(), samplesheet['cellranger_name'].tolist()))
    fastqs = dict(zip(samplesheet['sample_name'].tolist(), samplesheet['fastqs'].tolist()))
except:
    raise ValueError("Invalid samplesheet!")

try:
    depth = dict(zip(samplesheet['sample_name'].tolist(), samplesheet['depth'].tolist()))
except:
    if args.sample_reads is not None:
        depth = {name: args.sample_reads for name in sample_dict.keys()}
    else:
        depth = {}
        print("No sequencing depth specified, calling guides with all available reads...")

# get expected cell barcodes from gex data
with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    filtered_UMI_counts = dict()
    target_barcodes = dict()
    raw_UMI_counts = dict()
    raw_target_barcodes = dict()

    for sample_name, gex_path in sample_dict.items():
        filtered_pop = sc.read_10x_h5(os.path.join(gex_path, 'filtered_feature_bc_matrix.h5'))

        filtered_barcodes = filtered_pop.obs.index.map(lambda x: x.split('-')[0])
        filtered_UMI_counts[sample_name] = pd.Series(filtered_pop.X.sum(axis=1).A.flatten(), index=filtered_barcodes)
        
        target_barcodes[sample_name] = filtered_barcodes

        del filtered_pop
        
        raw_pop = sc.read_10x_h5(os.path.join(gex_path, 'raw_feature_bc_matrix.h5'))

        raw_barcodes = raw_pop.obs.index.map(lambda x: x.split('-')[0])
        raw_UMI_counts[sample_name] = pd.Series(raw_pop.X.sum(axis=1).A.flatten(), index=raw_barcodes)
        
        raw_target_barcodes[sample_name] = raw_barcodes
        
        del raw_pop

if args.verbose:
    print('Found cell barcodes:')
    print({name: len(bcs) for name, bcs in target_barcodes.items()})

if not args.call_only:

    # read in features
    features = pd.read_csv(args.EXPECTED_FEATURES)
    features['id'] = features['protospacer']
    features['full_sequence'] = features['protospacer'].map(lambda x: BARCODE_SEQUENCE_CONTEXT.replace('NNNNNNNNNNNNNNNNNNNN', x))
    features = features.drop_duplicates("protospacer")
    features.index = features.apply(lambda df: df.target_gene + "_" + df.protospacer, axis = 1)

    # run cellranger, produces a bam with barcode-labeled reads
    if args.verbose:
        print("Running cellranger...")

    command = '''{0} count --chemistry=ARC-v1 --transcriptome={1}
                    --fastqs={2}
                    --id={3}
                    --sample={4}
                    --localcores=8 --localmem=64'''

    def run_cellranger(sample_name, sample_cellranger_name, fastq_folder):
        os.chdir(args.WORKING_DIRECTORY)
        os.system(command.format(CELLRANGER_PATH, TRANSCRIPTOME_PATH, fastq_folder, sample_name, sample_cellranger_name).replace('\n', ''))

    if args.preflight:
        for name, cellranger_name in cellranger_sample_names.items():
            print(command.format(CELLRANGER_PATH, TRANSCRIPTOME_PATH, fastqs[name], sample_name, cellranger_name).replace('\n', ''))
        sys.exit(0)
    else:
        Parallel(n_jobs=4, verbose=args.verbose)(delayed(run_cellranger)(name, cellranger_name, fastqs[name]) for name, cellranger_name in cellranger_sample_names.items())

    # use fuzzy string matching to align guide reads
    def extract_sequence_from_read_name(read_name):
        pattern = r":r(.{20})"
        match = re.search(pattern, read_name)
        if match:
            return match.group(1)
        else:
            return None

    def barcode_bam_to_dual_guide_df(bam_file, sequence_context, verbose=False, sample = None):
        reads = list()
        barcode_start = sequence_context.find('N')
        barcode_end = sequence_context.rfind('N') + 1
        
        with pysam.AlignmentFile(bam_file, "rb", threads=4) as f:
            for read in f.fetch(until_eof=True):
                try:
                    cb = read.get_tag('CB')
                    ub = read.get_tag('UB')
                    sequence = extract_sequence_from_read_name(read.query_name)
                except KeyError:
                    continue
                reads.append({'CB': cb, 'UB': ub, 'seq': read.seq[barcode_start:barcode_end], 'index_seq': sequence})
        print(len(reads))   
        return pd.DataFrame(reads) if sample is None else pd.DataFrame(reads).sample(n = int(sample))

    def align_dual_barcode_umis(sample, choices, gex_filter_cb, depth = None):
        sample_path = os.path.join(args.WORKING_DIRECTORY, sample, 'outs', 'possorted_genome_bam.bam')
        print('{0}: {1}'.format(sample, sample_path))
        
        # loads all reads from bam into a dataframe with cell barcode and UMI as columns and protospacer as sequence
        reads = barcode_bam_to_dual_guide_df(sample_path, BARCODE_SEQUENCE_CONTEXT, sample = depth)
        # we only need to align actual sequences and can then just use that to label the reads array
        unique_sequences = reads['seq'].unique()    
        
        # do fuzzy string matching to identify protospacers
        alignments = chunked_call_alignments(unique_sequences, choices, num_chunks=NUM_CHUNKS, num_workers=NUM_WORKERS)
        
        # apply identity back to reads
        reads['identity'] = reads['seq'].map(alignments['identity'])
        reads['score'] = reads['seq'].map(alignments['score'])

        # remove '-1' from cell barcodes
        reads['CB'] = reads['CB'].map(lambda x: x.split('-')[0])
        
        # collapse to UMIs (only counting reads with good alignments)
        umis = reads[(reads['score'] >= MATCHING_THRESHOLD)].groupby(['CB', 'UB', 'identity']).agg({'seq': 'size',
                                                'score': 'mean'}).rename(columns={'seq': 'count',
                                                                                    'score': 'mean_score'})

        umis.to_csv(os.path.join(args.WORKING_DIRECTORY, '{0}_raw_barcode_umis.csv.gz'.format(sample)))
        umis[umis.index.get_level_values(0).isin(gex_filter_cb)].to_csv(os.path.join(args.WORKING_DIRECTORY, '{0}_filtered_barcode_umis.csv.gz'.format(sample)))   

    if args.verbose:
        print("Aligning guides...")
    Parallel(n_jobs=4, verbose=args.verbose)(delayed(align_dual_barcode_umis)(sample, features['protospacer'], target_barcodes[sample], depth = depth.get(sample, None)) for sample in sample_dict)

# merge and process umi files
reads_files = {sample: os.path.join(args.WORKING_DIRECTORY, '{0}_filtered_barcode_umis.csv.gz'.format(sample)) for sample in sample_dict}
raw_reads_files = {sample: os.path.join(args.WORKING_DIRECTORY, '{0}_raw_barcode_umis.csv.gz'.format(sample)) for sample in sample_dict}

def get_dual_reads(reads_file):
    raw_reads = pd.read_csv(reads_file, index_col=[0,1,2])
    return raw_reads

dfs = {sample: get_dual_reads(sample_path) for sample, sample_path in reads_files.items()}
raw_dfs = {sample: get_dual_reads(sample_path) for sample, sample_path in raw_reads_files.items()}

if args.verbose:
    print("Calculating identity stats...")

def get_identity_stats_new(raw_reads):
    
    identities = raw_reads.groupby(level=[0, 2])[['count']].count()
    identities.columns = ['UMI']
    identities['UMI_fraction'] = identities.groupby(level=0)['UMI'].transform(lambda x: x/x.sum())
    identities['total_reads'] = raw_reads.groupby(level=[0, 2])['count'].sum()

    identities['mean_score'] = raw_reads.groupby(level=[0, 2])['mean_score'].mean()
    #identities['anscombe_median_support'] = raw_reads.groupby(level=[0, 2])['anscombe'].median()
    identities['num_identities'] = identities.groupby(level=0)['UMI'].transform('count')
    identities['read_fraction'] = identities.groupby(level=0)['total_reads'].transform(lambda x: x/x.sum())
    identities['guide_UMI_entropy'] = identities.groupby(level=0)["UMI_fraction"].transform(entropy)
    identities['total_log_UMI'] = identities.groupby(level=0)["UMI"].transform(lambda x: np.log2(x.sum()))
    
    return identities

ids = {sample: get_identity_stats_new(reads) for sample, reads in tqdm(dfs.items())}
raw_ids = {sample: get_identity_stats_new(reads) for sample, reads in tqdm(raw_dfs.items())}

# call ids
def return_called_ids(ids, sample, eps=args.eps, min_samples=10, umi_cutoff=args.umi_cutoff, model_entropy = args.entropy):

    all_ids = ids[['UMI', 'total_reads', "guide_UMI_entropy"]].copy() if model_entropy else ids[['UMI', 'total_reads']].copy()
    all_ids["UMI"] = np.log2(all_ids["UMI"])
    all_ids["total_reads"] = np.log2(all_ids["total_reads"])
    top_ids = all_ids.query(f'UMI >= {umi_cutoff}')

    # Define and fit DBSCAN model
    dbscan = hdbscan.HDBSCAN(cluster_selection_epsilon = eps, min_samples = min_samples, metric = 'l1')
    dbscan.fit(top_ids)

    # Count the labels
    labels, counts = np.unique(dbscan.labels_, return_counts=True)
    label_counts = dict(zip(labels, counts))
    label_counts.pop(-1)

    # Sort the labels based on counts
    sorted_labels = sorted(label_counts, key=label_counts.get, reverse=True)

    # Get top 2 labels
    top_clusters = sorted_labels[:2]

    # Remove minority clusters
    labels_to_keep = np.isin(dbscan.labels_, top_clusters)
    top_ids_filtered = top_ids[labels_to_keep]

    # Get the new labels for the data
    filtered_labels = dbscan.labels_[labels_to_keep]

    # Identify which cluster has higher UMI values on average
    if top_ids_filtered[filtered_labels == top_clusters[0]].UMI.mean() > top_ids_filtered[filtered_labels == top_clusters[1]].UMI.mean():
        high_umi_cluster = top_clusters[0]
        low_umi_cluster = top_clusters[1]
    else:
        high_umi_cluster = top_clusters[1]
        low_umi_cluster = top_clusters[0]

    # Relabel so the cluster with higher average UMI is always 1
    filtered_labels = np.where(filtered_labels == high_umi_cluster, 1, 0)

    # 2. Fit a logistic regression model to this filtered data to identify a separating hyperplane.
    log_reg = LogisticRegression()
    log_reg.fit(top_ids_filtered, filtered_labels)

    # 3. Use this classifier to extend the labels to a larger dataset in 'all_ids'
    all_ids_labels = log_reg.predict(all_ids)
    all_ids['cluster'] = all_ids_labels
    

    # plot id calls
    if not model_entropy:

        fig, ax = plt.subplots(ncols = 2, sharex = True, sharey = True, figsize = (8, 4))
        ax[0].scatter(top_ids['UMI'], top_ids['total_reads'], label = 'top id', s=1, c=dbscan.labels_)
        ax[0].set_title(f"Top IDs, UMI cutoff = {umi_cutoff}")
        ax[0].set_xlim(-0.5,8.5)
        ax[1].set_ylim(-0.25,16.25)

        ax[1].scatter(all_ids['UMI'], all_ids['total_reads'], label = 'top id', s=1, c=all_ids['cluster'])
        ax[1].scatter(all_ids.query('UMI < {0}'.format(MIN_LOG_UMI))['UMI'], all_ids.query('UMI < {0}'.format(MIN_LOG_UMI))['total_reads'], s=1, label='top id', c='gray')
        ax[1].set_title(f"All IDs")
        ax[1].set_xlim(-0.5,8.5)
        ax[1].set_ylim(-0.5,16.25)
    
    else:
        
        fig, ax = plt.subplots(ncols = 2, nrows = 2, sharex = True, sharey = True, figsize = (12, 8))
        ax[0,0].scatter(top_ids['UMI'], top_ids['total_reads'], label = 'top id', s=1, c=dbscan.labels_)
        ax[0,0].set_title(f"Top IDs, UMI cutoff = {umi_cutoff}")

        ax[1,0].scatter(all_ids['UMI'], all_ids['total_reads'], label = 'top id', s=1, c=all_ids['cluster'])
        ax[1,0].scatter(all_ids.query('UMI < {0}'.format(MIN_LOG_UMI))['UMI'], all_ids.query('UMI < {0}'.format(MIN_LOG_UMI))['total_reads'], s=1, label='top id', c='gray')
        ax[1,0].set_title(f"All IDs")

        ax[0,1].scatter(top_ids['UMI'], top_ids['total_reads'], label = 'top id', s=1, c=top_ids['guide_UMI_entropy'])
        ax[0,1].set_title(f"Top IDs, UMI cutoff = {umi_cutoff}, colored by guide UMI entropy in cell of origin")

        ax[1,1].scatter(all_ids['UMI'], all_ids['total_reads'], label = 'top id', s=1, c=all_ids['guide_UMI_entropy'])
        ax[1,1].scatter(all_ids.query('UMI < {0}'.format(MIN_LOG_UMI))['UMI'], all_ids.query('UMI < {0}'.format(MIN_LOG_UMI))['total_reads'], s=1, label='top id', c='gray')
        ax[1,1].set_title(f"All IDs, colored by guide UMI entropy in cell of origin")

    plt.xlabel("log2(ID UMI)")
    plt.ylabel("log2(ID total reads)")
    plt.suptitle(f"{sample}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.WORKING_DIRECTORY,f"{sample}_id_clusters.pdf"), format = 'pdf', bbox_inches='tight')

    plt.figure(figsize=(5,4))
    plt.hexbin(x = all_ids['UMI'], y = all_ids['total_reads'], vmax = 150, gridsize = 30, cmap = 'viridis', mincnt=1)
    plt.title(f"{sample} ID density")
    plt.xlabel("log2 guide UMI")
    plt.ylabel("log2 total guide reads")
    plt.tight_layout()
    plt.savefig(os.path.join(args.WORKING_DIRECTORY, f"{sample}_id_density.pdf"), format = 'pdf', bbox_inches = 'tight')
    
    return all_ids.query('cluster==1 and UMI >= {0}'.format(MIN_LOG_UMI))

called_ids = dict()
for sample in sample_dict.keys():
    called_ids[sample] = return_called_ids(ids[sample], sample)

# output called ids by sample
for name, the_ids in called_ids.items():
    the_ids.to_csv(os.path.join(args.WORKING_DIRECTORY, '{0}_called_ids.csv'.format(name)))

# output percent cells by called guide
id_counts = {key: val.groupby('CB')['UMI'].count() for key, val in called_ids.items()}
id_counts = {key: val.reindex(target_barcodes[key]).fillna(0).value_counts().sort_index().to_frame('count') for key, val in id_counts.items()}
for k, v in id_counts.items():
    v["sample"] = k

df_id_counts = pd.concat([v for v in id_counts.values()], axis = 0)
df_id_counts["pct_cells"] = df_id_counts.groupby('sample').transform(lambda x: 100 * x / x.sum())
df_id_counts["n_guides"] = df_id_counts.index.astype(int).map(lambda n: '4+' if n > 3 else n)
df_id_counts = df_id_counts.sort_values("sample")
df_id_counts.to_csv(os.path.join(args.WORKING_DIRECTORY,f"{args.run_id}_guide_call_summary.csv"))

plt.figure(figsize = [5,4])
sns.barplot(data = df_id_counts, y = "n_guides", x = 'pct_cells', hue = 'sample', orient = 'h', palette = 'mako')
plt.ylabel("Called guides")
plt.tight_layout()
plt.savefig(os.path.join(args.WORKING_DIRECTORY,f"{args.run_id}_guide_calls.pdf"), dpi = 200)

# write pipeline summary
with open(os.path.join(args.WORKING_DIRECTORY,f"{args.run_id}_cropseq_call.txt"), 'w') as summary:
    args_dict = vars(args)
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    summary.write(f"CROP-seq guide calling pipeline: {args.run_id}, {now}\n")
    summary.write("Parameters:\n")
    summary.write(f"Min log UMI: {MIN_LOG_UMI}\n")
    for k, v in args_dict.items():
        summary.write(f"{k}: {v}\n")