import heapq
import numpy as np
import pysam
from rapidfuzz import process, string_metric
import pandas as pd
from tqdm import tqdm

"""
Copyright 2020 Google LLC.
SPDX-License-Identifier: Apache-2.0
"""
def isotonic_regression_l1_total_order(y, w):
    """Finds a non-decreasing fit for the specified `y` under L1 norm.

    The O(n log n) algorithm is described in:
    "Isotonic Regression by Dynamic Programming", Gunter Rote, SOSA@SODA 2019.

    Args:
        y: The values to be fitted, 1d-numpy array.
        w: The loss weights vector, 1d-numpy array.

    Returns:
        An isotonic fit for the specified `y` which minimizies the weighted
        L1 norm of the fit's residual.
    """
    h = []    # max heap of values
    p = np.zeros_like(y)    # breaking position
    for i in range(y.size):
        a_i = y[i]
        w_i = w[i]
        heapq.heappush(h, (-a_i, 2 * w_i))
        s = -w_i
        b_position, b_value = h[0]
        while s + b_value <= 0:
            s += b_value
            heapq.heappop(h)
            b_position, b_value = h[0]
        b_value += s
        h[0] = (b_position, b_value)
        p[i] = -b_position
    z = np.flip(np.minimum.accumulate(np.flip(p)))    # right_to_left_cumulative_min
    return z

def isotonic_threshold(data, sorted_reindex=None):
    pred = isotonic_regression_l1_total_order(data.values, np.ones(len(data)))
    if sorted_reindex is None:
        return pd.Series(pred, data.index)
    else:
        if sorted(sorted_reindex) != list(sorted_reindex):
            raise ValueError('Warning: index must be sorted for isotonic regression')
        return pd.Series(pred, data.index).reindex(sorted_reindex).fillna(method='ffill').fillna(method='bfill')

def barcode_bam_to_df(bam_file, sequence_context, verbose=False):
    reads = list()
    barcode_start = sequence_context.find('N')
    barcode_end = sequence_context.rfind('N') + 1

    with pysam.AlignmentFile(bam_file, "rb", threads=4) as f:
        for read in f.fetch(until_eof=True):
            try:
                cb = read.get_tag('CB')
                ub = read.get_tag('UB')
            except KeyError:
                continue
            reads.append({'CB': cb, 'UB': ub, 'seq': read.seq[barcode_start:barcode_end]})
    return pd.DataFrame(reads)

# def barcode_bam_to_df_dual_guide(bam_file, sequence_context_1, sequence_context_2, verbose=False):
#     reads = list()
#     barcode_start_1 = sequence_context_1.find('N') -5
#     barcode_end_1 = sequence_context_1.rfind('N') + 6
#
#     barcode_start_2 = sequence_context_2.find('N') -5
#     barcode_end_2 = sequence_context_2.rfind('N') + 6
#
#     with pysam.AlignmentFile(bam_file, "rb", threads=4) as f:
#         for read in f.fetch(until_eof=True):
#             try:
#                 cb = read.get_tag('CB')
#                 ub = read.get_tag('UB')
#             except KeyError:
#                 continue
#             reads.append({'CB': cb, 'UB': ub, 'seq_1': read.seq[barcode_start_1:barcode_end_1],
#                                               'seq_2': read.seq[barcode_start_2:barcode_end_2]})
#     return pd.DataFrame(reads)

# def rapidfuzz_align(seqs, choices):
#     aligned_reads = dict()
#     aligned_scores = dict()

#     for seq in tqdm(seqs, total=len(seqs)):
#         res = process.extractOne(seq, choices)
#         fz_gx = res[2]
#         fz_score = res[1]
#         aligned_reads[seq] = fz_gx
#         aligned_scores[seq] = fz_score
#     return pd.DataFrame([aligned_reads, aligned_scores], index=['identity', 'score']).T

def call_alignments(unique_sequences, choices, scorer=string_metric.normalized_levenshtein, num_workers=1):
    res = process.cdist(unique_sequences, choices, workers=num_workers, scorer=scorer)
    res = pd.DataFrame(res, index=unique_sequences, columns=choices.index)

    max_scores = res.max(axis=1)
    max_indices = res.idxmax(axis=1)

    return pd.DataFrame([max_scores, max_indices], columns=unique_sequences, index=['score', 'identity']).T

def chunked_call_alignments(unique_sequences, choices, scorer=string_metric.normalized_levenshtein, num_chunks=1, num_workers=1):
    unique_chunks = np.array_split(unique_sequences, num_chunks)
    called_identities = list()

    for chunk in tqdm(unique_chunks):
        called_identities.append(call_alignments(chunk, choices, scorer=scorer, num_workers=num_workers))
    called_identities = pd.concat(called_identities, verify_integrity=False)

    return called_identities

def get_background_umis(identities, possibly_real, num_samples, hard_umi_threshold):
    num_identities = identities['num_identities'].value_counts()
    num_identities = num_identities[num_identities > num_samples].sort_index()

    probably_background = dict()

    for n in num_identities.index:
        res = identities[identities['num_identities']==n]
        probably_background[n] = res.sort_values(['UMI', 'UMI_fraction', 'anscombe_median_support'], ascending=False).groupby(level=0).tail(n-possibly_real)

    total_probably_background = pd.concat(probably_background.values()).sort_index()
    total_probably_background = total_probably_background[total_probably_background['UMI'] < hard_umi_threshold]
    return total_probably_background

def get_isotonic_thresholds(identities, probably_background, chosen_quantile):
    possible_number_identities = sorted(identities['num_identities'].unique())
    quantile_stats = probably_background.groupby('num_identities').quantile(chosen_quantile)
    umi_threshold = isotonic_threshold(quantile_stats['UMI'], sorted_reindex=possible_number_identities)
    fraction_threshold = -isotonic_threshold(-quantile_stats['UMI_fraction'], sorted_reindex=possible_number_identities)
    read_fraction_threshold = -isotonic_threshold(-quantile_stats['read_fraction'], sorted_reindex=possible_number_identities)
    return umi_threshold, fraction_threshold, read_fraction_threshold

def get_identity_stats(raw_reads):
    identities = raw_reads.groupby(level=[0, 2])[['count']].count()
    identities.columns = ['UMI']
    identities['UMI_fraction'] = identities.groupby(level=0)['UMI'].transform(lambda x: x/x.sum())
    identities['total_reads'] = raw_reads.groupby(level=[0, 2])['count'].sum()

    identities['mean_score'] = raw_reads.groupby(level=[0, 2])['mean_score'].mean()
    identities['anscombe_median_support'] = raw_reads.groupby(level=[0, 2])['anscombe'].median()
    identities['num_identities'] = identities.groupby(level=0)['UMI'].transform('count')
    identities['read_fraction'] = identities.groupby(level=0)['total_reads'].transform(lambda x: x/x.sum())
    return identities

def get_identity_stats_dual_guide(raw_reads):
    identities = raw_reads.groupby(level=[0, 2])[['count']].count()
    identities.columns = ['UMI']
    identities['UMI_fraction'] = identities.groupby(level=0)['UMI'].transform(lambda x: x/x.sum())
    identities['total_reads'] = raw_reads.groupby(level=[0, 2])['count'].sum()

    identities['mean_score_1'] = raw_reads.groupby(level=[0, 2])['mean_score_1'].mean()
    identities['mean_score_2'] = raw_reads.groupby(level=[0, 2])['mean_score_2'].mean()
    identities['anscombe_median_support'] = raw_reads.groupby(level=[0, 2])['anscombe'].median()
    identities['num_identities'] = identities.groupby(level=0)['UMI'].transform('count')
    identities['read_fraction'] = identities.groupby(level=0)['total_reads'].transform(lambda x: x/x.sum())
    return identities

def get_reads(reads_file):
    raw_reads = pd.read_csv(reads_file, index_col=[0,1,2])
    raw_reads['anscombe'] = 2*np.sqrt(raw_reads['count'] + 3/8)
    return raw_reads

def format_identity_table(called_identities, target_barcodes):
    res = dict()
    top_identities = called_identities.reset_index(level=1).sort_values(['UMI', 'UMI_fraction', 'anscombe_median_support'], ascending=False).groupby(level=0).head(1)
    sorted_identities = called_identities.reset_index(level=1).sort_values(['UMI', 'UMI_fraction', 'anscombe_median_support'], ascending=False)
    observed_cbs = top_identities.index

    for cb in target_barcodes:
        if cb in observed_cbs:
            cb_info = top_identities.loc[cb]
            # cell with a single confidently called identity
            if (cb_info['num_called_confident_identities'] == 1) & (cb_info['num_called_identities'] == 1):
                res[cb] = {'cell_BC': cb, 'guide_identity': cb_info['identity'], 'UMI_count': cb_info['UMI'], 'read_count': cb_info['total_reads'], 'good_coverage': True, 'number_of_cells': cb_info['num_called_identities'], 'mean_score': cb_info['mean_score']}
            # weak positive
            elif (cb_info['num_called_confident_identities'] == 0) & (cb_info['num_called_identities'] >= 1):
                res[cb] = {'cell_BC': cb, 'guide_identity': cb_info['identity'], 'UMI_count': cb_info['UMI'], 'read_count': cb_info['total_reads'], 'good_coverage': False, 'number_of_cells': cb_info['num_called_identities'], 'mean_score': cb_info['mean_score']}
            # multiple called identities -- merge
            elif (cb_info['num_called_identities'] > 1):
                res[cb] = {'cell_BC': cb, 'guide_identity': '/'.join(sorted_identities.loc[cb, 'identity']), 'UMI_count': cb_info['UMI'], 'read_count': cb_info['total_reads'], 'good_coverage': False, 'number_of_cells': cb_info['num_called_identities'], 'mean_score': np.nan}
            # guides detected, no identity found
            elif (cb_info['num_called_identities'] == 0):
                res[cb] = {'cell_BC': cb, 'guide_identity': 'no_confident_identity', 'UMI_count': cb_info['UMI'], 'read_count': cb_info['total_reads'], 'good_coverage': False,  'number_of_cells': cb_info['num_called_identities'], 'mean_score': np.nan}
        else:
            # cell barcode not in guide sequencing
            res[cb] = {'cell_BC': cb, 'guide_identity': 'missing', 'UMI_count': 0, 'read_count': 0, 'good_coverage': False, 'number_of_cells': 0, 'mean_score': np.nan}
    return pd.DataFrame(res).T.set_index('cell_BC').sort_index()

def format_identity_table_dual_guide(called_identities, target_barcodes):
    res = dict()
    top_identities = called_identities.reset_index(level=1).sort_values(['UMI', 'UMI_fraction', 'anscombe_median_support'], ascending=False).groupby(level=0).head(1)
    sorted_identities = called_identities.reset_index(level=1).sort_values(['UMI', 'UMI_fraction', 'anscombe_median_support'], ascending=False)
    observed_cbs = top_identities.index

    for cb in target_barcodes:
        if cb in observed_cbs:
            cb_info = top_identities.loc[cb]
            # cell with a single confidently called identity
            if (cb_info['num_called_confident_identities'] == 1) & (cb_info['num_called_identities'] == 1):
                res[cb] = {'cell_BC': cb, 'guide_identity': cb_info['identity'], 'UMI_count': cb_info['UMI'], 'read_count': cb_info['total_reads'], 'good_coverage': True, 'number_of_cells': cb_info['num_called_identities'], 'mean_score': cb_info['mean_score_1']}
            # weak positive
            elif (cb_info['num_called_confident_identities'] == 0) & (cb_info['num_called_identities'] >= 1):
                res[cb] = {'cell_BC': cb, 'guide_identity': cb_info['identity'], 'UMI_count': cb_info['UMI'], 'read_count': cb_info['total_reads'], 'good_coverage': False, 'number_of_cells': cb_info['num_called_identities'], 'mean_score': cb_info['mean_score_1']}
            # multiple called identities -- merge
            elif (cb_info['num_called_identities'] > 1):
                res[cb] = {'cell_BC': cb, 'guide_identity': '/'.join(sorted_identities.loc[cb, 'identity']), 'UMI_count': cb_info['UMI'], 'read_count': cb_info['total_reads'], 'good_coverage': False, 'number_of_cells': cb_info['num_called_identities'], 'mean_score': np.nan}
            # guides detected, no identity found
            elif (cb_info['num_called_identities'] == 0):
                res[cb] = {'cell_BC': cb, 'guide_identity': 'no_confident_identity', 'UMI_count': cb_info['UMI'], 'read_count': cb_info['total_reads'], 'good_coverage': False,  'number_of_cells': cb_info['num_called_identities'], 'mean_score': np.nan}
        else:
            # cell barcode not in guide sequencing
            res[cb] = {'cell_BC': cb, 'guide_identity': 'missing', 'UMI_count': 0, 'read_count': 0, 'good_coverage': False, 'number_of_cells': 0, 'mean_score': np.nan}
    return pd.DataFrame(res).T.set_index('cell_BC').sort_index()

def _vars(a, axis=None):
    """ Variance of sparse matrix a
    var = mean(a**2) - mean(a)**2
    """
    a_squared = a.copy()
    a_squared.data **= 2
    return a_squared.mean(axis) - np.square(a.mean(axis))

def stds(a, axis=None):
    """ Standard deviation of sparse matrix a
    std = sqrt(var(a))
    """
    return np.sqrt(_vars(a, axis)).A.flatten()
