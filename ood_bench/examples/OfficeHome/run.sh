data_dir=$1
dataset="OfficeHome"
shared_args="--n_trials 4 --parallel --data_dir $data_dir --dataset $dataset --backbone resnet-18 --pretrained_model_path auto --calibrate"

python -m ood_bench.scripts.main $shared_args --envs_p 1 2 3 --envs_q 0 --output_dir ood_bench/examples/$dataset/outputs/A
python -m ood_bench.scripts.main $shared_args --envs_p 0 2 3 --envs_q 1 --output_dir ood_bench/examples/$dataset/outputs/C
python -m ood_bench.scripts.main $shared_args --envs_p 0 1 3 --envs_q 2 --output_dir ood_bench/examples/$dataset/outputs/P
python -m ood_bench.scripts.main $shared_args --envs_p 0 1 2 --envs_q 3 --output_dir ood_bench/examples/$dataset/outputs/R