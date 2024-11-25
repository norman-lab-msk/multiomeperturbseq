import pandas as pd
import numpy as np
import scanpy as sc
import snapatac2 as snap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import pyranges as pr
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from scipy.stats import spearmanr, false_discovery_control
import time
import sys
import argparse
plt.rcdefaults()
warnings.simplefilter("ignore")

def cut_qc_metrics_by_called(sample_name, figsize = (15,5), mode = 'gex'):

    id_files = [file for file in os.scandir("/data/norman/eli/T7/202404_SIRLOIN_multiome/guide_calling/T7_outs") if "called_ids" in file.name]
    called_ids = pd.read_csv([f for f in id_files if  sample_name in f.name][0]).reset_index().groupby("CB").UMI.count().to_frame("n_guides")
    called_ids.index = called_ids.index.map(lambda cb: cb + "-1")

    if mode == 'gex':

        adata_files = [file for file in os.scandir("/data/norman/eli/T7/202404_SIRLOIN_multiome/") if sample_name in file.name]
        adata_path = adata_files[0].path + '/outs/filtered_feature_bc_matrix.h5'

        adata = sc.read_10x_h5(adata_path)
        adata.var_names_make_unique()
        adata.var["mito"] = adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, inplace=True, qc_vars = ["mito"])
        
        merge = adata.obs.join(called_ids, how = 'left')
        merge["called"] = merge["n_guides"].fillna(0).map(lambda n: n > 0)
        merge.drop_duplicates(inplace = True)

        fig, ax = plt.subplots(1, 3, figsize=figsize)
        sns.histplot(data = merge, x = 'log1p_total_counts', kde = True, hue = 'called', ax = ax[0])
        sns.histplot(data = merge, x = 'log1p_n_genes_by_counts', kde = True, hue = 'called', ax = ax[1])
        sns.histplot(data = merge, x = 'pct_counts_mito', kde = True, hue = 'called', ax = ax[2])
        ax[2].set_xlim([-2,50])
        fig.suptitle(f"{sample_name} GEX QC metrics by guide call")
        fig.show()

    elif mode == 'atac':

        adata_files = [file for file in os.scandir("/data/norman/eli/T7/202404_SIRLOIN_multiome/figs/intermediate_files") if sample_name in file.name and 'preprocessed_atac' in file.name]
        adata = snap.read(adata_files[0].path, backed = None)
        merge = adata.obs.join(called_ids, how = 'left')
        merge["called"] = merge["n_guides"].fillna(0).map(lambda n: n > 0)
        merge.drop_duplicates(inplace = True)

        fig, ax = plt.subplots(1, 2, figsize=figsize)
        sns.histplot(data = merge, x = 'n_fragment', kde = True, hue = 'called', ax = ax[0])
        ax[0].set_xlim([0,1e5])
        sns.histplot(data = merge, x = 'tsse', kde = True, hue = 'called', ax = ax[1])
        fig.suptitle(f"{sample_name} ATAC QC metrics by guide call")
        fig.show()

    return merge

parser = argparse.ArgumentParser()
parser.add_argument("--genes", type = str, help = "Path to gene list (one gene per line, txt file)")
parser.add_argument("--outdir", type = str, help = "Path to output directory for models and predictions")
parser.add_argument("--good_quality", action = 'store_true', help = "Use good_quality autogluon preset instead of medium_quality (takes about 20x longer per gene and is very resource-intensive)")
parser.add_argument("--scale_features", action = 'store_true', help = "Scale ATAC features with MinMaxScaler")
parser.add_argument("--refit_full", action = 'store_true', help = "Refit full model after CV")
args = parser.parse_args()

adata = snap.read("/data/norman/eli/T7/202404_SIRLOIN_multiome/figs/intermediate_files/Lane1_040_preprocessed_atac.h5ad", backed = None)
gexqc_040 = cut_qc_metrics_by_called("Lane1_040")
atacqc_040 = cut_qc_metrics_by_called("Lane1_040", mode = 'atac')
qc040 = gexqc_040.join(atacqc_040.drop(["n_guides", "called"], axis = 1), how = 'inner')
qc040.n_fragment = qc040.n_fragment.astype(int)
singlets_040 = qc040.query("n_guides == 1 and log1p_total_counts > 6 and log1p_n_genes_by_counts > 6 and n_fragment > 1000")
ids_040 = pd.read_csv("/data/norman/eli/T7/202404_SIRLOIN_multiome/guide_calling/T7_outs/Lane1_040_called_ids.csv")
ids_040['n_guides'] = ids_040.groupby('CB').UMI.transform('count')
ids_040 = ids_040.query('n_guides == 1')
ids_040['guide_target'] = ids_040['identity'].map(lambda s: s.split("_")[0])
ids_040['CB'] = ids_040.CB.map(lambda b: b + "-1")
ids_040.set_index('CB', inplace = True)
singlets_040 = singlets_040.join(ids_040['guide_target'], how = 'left')
assert len(singlets_040) == 4724

atac_cluster = adata[adata.obs.index.isin(singlets_040.index), adata.var.selected].copy()
atac_cluster.obs = atac_cluster.obs.join(singlets_040.guide_target)
snap.tl.spectral(atac_cluster, n_comps = 11)
df = pd.DataFrame(atac_cluster.obsm['X_spectral'], index = atac_cluster.obs.index)
df_first_comp_removed = df.iloc[:,1:].join(singlets_040.guide_target)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    dfs = []
    for guide in singlets_040.guide_target.unique():
        km = KMeans(n_clusters = 10)
        df_cluster = df_first_comp_removed.query(f"guide_target == '{guide}'").drop("guide_target", axis = 1)
        df_cluster['cluster'] = km.fit_predict(df_cluster)
        df_cluster['cluster'] = df_cluster.cluster.map(lambda c: f"{guide}_{c}")
        dfs.append(df_cluster)
    df = pd.concat(dfs)
    singlets_040 = singlets_040.join(df.cluster)

gex = sc.read_10x_h5("/data/norman/eli/T7/202404_SIRLOIN_multiome/Lane1_040/outs/filtered_feature_bc_matrix.h5")
gex.var_names_make_unique()
gex = gex[gex.obs.index.isin(singlets_040.index)]
gex.obs = gex.obs.join(singlets_040, how = 'right')
sc.pp.filter_genes(gex, min_cells = 100)
gex = gex[:, ~gex.var_names.str.contains("MT-")]
sc.pp.normalize_total(gex, target_sum=1e4)

gtf = pr.read_gtf('/fscratch/eli/genomes/refdata-gex-GRCh38-2020-A/genes/genes.gtf')[["Chromosome", "Feature", "Start", "End", "gene_id", "gene_type", "gene_name"]]
gtf = gtf[gtf.Feature == 'gene']

with open(args.genes, 'r') as f:
    genes_final = f.read().splitlines()
print(f"{len(genes_final)} total genes")

zs = get_hyperparameter_config('zeroshot')
zs.pop('NN_TORCH')

promoters = pd.read_csv("/home/normant1/notebooks/methylation variation/epdNewHuman006_extended_promoter_regions.bed", sep = '\t', header=None).iloc[:,[0,1,2]]
promoters.columns =["Chromosome", "Start", "End"]
promoters['reg'] = 'prom'
reg_elements = pd.read_csv("/data/norman/eli/T7/202404_SIRLOIN_multiome/figs/intermediate_files/encodeCcreCombined.bed", sep = '\t', header = None).iloc[:,[0,1,2,12]]
reg_elements.columns = ["Chromosome", "Start", "End", "reg"]
reg_elements = reg_elements.query("reg != 'prom'")
reg_elements = pr.PyRanges(pd.concat([reg_elements, promoters], axis = 0))

df_preds = []
for target_gene in genes_final:
    try:
        chrom = gtf.df.query('gene_name == @target_gene').Chromosome.item()
        start = int(np.round(gtf.df.query('gene_name == @target_gene').Start.item() / 500) * 500 - 2.5e5)
        end = int(np.round(gtf.df.query('gene_name == @target_gene').End.item() / 500) * 500 + 2.5e5)

        mtx = adata[adata.obs.index.isin(singlets_040.index)]
        mtx.obs = mtx.obs.join(singlets_040.cluster)
        mtx.obs = mtx.obs.join(singlets_040.guide_target)
        mtx.var["Chromosome"] = mtx.var.index.map(lambda s: s.split(":")[0])
        mtx.var["Start"] = mtx.var.index.map(lambda s: int(s.split(":")[1].split("-")[0]))
        mtx.var["End"] = mtx.var.index.map(lambda s: int(s.split(":")[1].split("-")[1]))

        model_mtx = mtx[:,mtx.var.index.isin(mtx.var.query("Chromosome == @chrom and Start >= @start and End <= @end").index)]
        model_mtx = model_mtx.to_df().div(model_mtx.obs.n_fragment, axis = 0).mul(1e6)
        model_mtx = model_mtx.join(singlets_040.cluster)
        
        model_mtx_cluster = model_mtx.groupby("cluster").mean().reset_index()
        model_mtx_cluster['guide_target'] = model_mtx_cluster.cluster.map(lambda s: s.split("_")[0])
        model_mtx_cluster['complex'] = model_mtx_cluster.guide_target.map(lambda g: 'NuA4' if g in ['ACTL6A', 'DMAP1', 'EP400'] else 'Poly' if g in ['EZH2', 'SUZ12', 'YY1'] else 'NTC' if g == 'NTC' else 'BAF')

        target_expr = gex[:,gex.var.index == target_gene].to_df().join(gex.obs.cluster).groupby("cluster").mean()
        target_expr.columns = ['target_expr']
        model_mtx_cluster = model_mtx_cluster.set_index("cluster").join(target_expr)
        train, test = train_test_split(model_mtx_cluster, test_size = 0.2, random_state = 42, stratify = model_mtx_cluster.guide_target)

        scaler = MinMaxScaler()
        train['target_expr_scaled'] = scaler.fit_transform(train.target_expr.to_numpy().reshape(-1,1))

        if args.scale_features:
            print('scaling features')
            feature_cols = [c for c in model_mtx.columns if c not in ['cluster', 'guide_target', 'complex']]
            feature_scaler = MinMaxScaler()
            train[feature_cols] = feature_scaler.fit_transform(train[feature_cols])
            print(train[feature_cols].describe())

        train = train.drop("target_expr", axis = 1)
        train = pd.get_dummies(train, columns = ['guide_target', 'complex'])

        if os.path.exists(os.path.join(args.outdir, target_gene)):
            print(f"Skipping {target_gene}: model already exists", flush = True)
            continue
        
        print(f"Training model for {target_gene}...", flush = True)
        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ag_train = TabularDataset(train)
            predictor = TabularPredictor(
                label='target_expr_scaled', 
                path = os.path.join(args.outdir, target_gene), 
                eval_metric = 'mse'
            )
            predictor.fit(
                ag_train, 
                presets = 'medium_quality' if not args.good_quality else 'good_quality', 
                ag_args_fit={'num_gpus': 1}, 
                ds_args={'memory_safe_fits': False}, 
                hyperparameters=zs, 
                verbosity = 0, 
                num_cpus=24, 
                refit_full = args.refit_full, 
                set_best_to_refit_full = args.refit_full,
            )
            
        test['target_expr_scaled'] = scaler.transform(test.target_expr.to_numpy().reshape(-1,1))
        if args.scale_features:
            test[feature_cols] = feature_scaler.transform(test[feature_cols])

        test = test.drop("target_expr", axis = 1)
        test = pd.get_dummies(test, columns = ['guide_target', 'complex'])

        y_pred = predictor.predict(test.drop(columns=['target_expr_scaled']))
        sp = spearmanr(test.target_expr_scaled, y_pred).statistic
        print(f"Target gene {target_gene}: test Spearman {sp}", flush = True)

        test_plot = test.copy()
        guide_target_cols = [c for c in test_plot.columns if c.startswith('guide_target')]
        complex_cols = [c for c in test_plot.columns if c.startswith('complex')]
        test_plot['guide_target'] = pd.from_dummies(test_plot[guide_target_cols], sep = '_')
        test_plot['complex'] = pd.from_dummies(test_plot[complex_cols], sep = '_')

        plt.figure(figsize = (8,6))
        sns.scatterplot(x = test_plot.target_expr_scaled, y = y_pred, hue = test_plot.guide_target)
        plt.xlabel("Observed expression")
        plt.ylabel("Predicted expression")
        plt.axline([0,0], [1,1])
        plt.savefig(os.path.join(args.outdir, f"{target_gene}.png"))
        plt.close()

        df_feature_importance = predictor.feature_importance(test, time_limit=600)
        df_feature_importance['fdr'] = false_discovery_control(df_feature_importance.p_value)
        df_feature_importance['target_gene'] = target_gene

        guide_target_feature = df_feature_importance.query("index.str.startswith('guide_target')")
        complex_feature = df_feature_importance.query("index.str.startswith('complex')")

        df_feature_importance = df_feature_importance.query("index.str.startswith('chr')")
        df_feature_importance['Chromosome'] = df_feature_importance.index.map(lambda s: s.split(":")[0])
        df_feature_importance['Start'] = df_feature_importance.index.map(lambda s: int(s.split(":")[1].split("-")[0]))
        df_feature_importance['End'] = df_feature_importance.index.map(lambda s: int(s.split(":")[1].split("-")[1]))

        df_feature_importance = pr.PyRanges(df_feature_importance).join(reg_elements, how = 'left', report_overlap = 'True').df
        df_feature_importance = df_feature_importance.drop(['Start_b', 'End_b'], axis = 1)
        df_feature_importance = pd.concat([df_feature_importance, guide_target_feature, complex_feature], axis = 0)
        df_feature_importance.to_csv(os.path.join(args.outdir, f"{target_gene}_feature_importance.csv"))

        df_pred = pd.DataFrame({"obs_expr":test.target_expr_scaled, "pred_expr":y_pred, 'target_gene': target_gene, 'test_spearman': sp})
        df_preds.append(df_pred)

        # do this within loop so predictions update live
        df = pd.concat(df_preds)
        df.to_csv(os.path.join(args.outdir, "test_preds.csv"))
        
        end = time.time()
        print(f"Training and feature importance done in {end - start} seconds", flush = True)

    except Exception as e:
        print(f"Error during training for {target_gene}: {e}")
        continue

df = pd.concat(df_preds)
df.to_csv(os.path.join(args.outdir, "test_preds.csv"))
