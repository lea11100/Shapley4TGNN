
# simulate_v1, simulate_v2, wikipedia, reddit
source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate condafact
for i in 0, 1, 2
do
    echo "${i}-th run\n"

    dataset=simulate_v2
    python learn_simulate.py -d ${dataset} --bs 256 --n_degree 10 --n_epoch 100 --agg_method attn --attn_mode prod --gpu 1 --n_head 2 --prefix ${dataset}
    
    dataset=reddit
    python learn_edge.py -d ${dataset} --bs 512 --n_degree 10 --n_epoch 10 --agg_method attn --attn_mode prod --gpu 1 --n_head 2 --prefix ${dataset}


done
