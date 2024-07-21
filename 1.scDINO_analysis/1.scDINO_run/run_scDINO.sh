#!/bin/bash
# this script runs the pre-processing and scDINO pipeline

conda activate scDINO_env

echo "Starting the scDINO analysis..."

# make logs directory if it doesn't exist
mkdir -p logs



# calculate the number of cores this machine has and use all but 2
# unless it has only 2 cores, in which case use 1 core
NCORES=$(nproc)
if [ $NCORES -gt 2 ]; then
    NCORES=$(($NCORES - 2))
else
    NCORES=1
fi

snakemake -s only_downstream_snakefile --configfile="only_downstream_analyses.yaml" --until compute_CLS_features --cores $NCORES --rerun-incomplete > logs/CLS_token_run.log 2>&1 > logs/CLS_token_run.log 2>&1

snakemake -s only_downstream_snakefile --configfile="only_downstream_analyses.yaml" --cores $NCORES --rerun-incomplete > logs/downstream_run.log 2>&1

echo "scDINO analysis complete"

conda deactivate
