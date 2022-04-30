data_dir=$1
dataset="DomainNet"
shared_args="--n_trials 4 --parallel --data_dir $data_dir --dataset $dataset --backbone resnet-18 --pretrained_model_path auto --calibrate"

python -m ood_bench.scripts.main $shared_args --envs_p 1 2 3 4 5 --envs_q 0 --output_dir ood_bench/examples/$dataset/outputs/clip
python -m ood_bench.scripts.main $shared_args --envs_p 0 2 3 4 5 --envs_q 1 --output_dir ood_bench/examples/$dataset/outputs/info
python -m ood_bench.scripts.main $shared_args --envs_p 0 1 3 4 5 --envs_q 2 --output_dir ood_bench/examples/$dataset/outputs/paint
python -m ood_bench.scripts.main $shared_args --envs_p 0 1 2 4 5 --envs_q 3 --output_dir ood_bench/examples/$dataset/outputs/quick
python -m ood_bench.scripts.main $shared_args --envs_p 0 1 2 3 5 --envs_q 4 --output_dir ood_bench/examples/$dataset/outputs/real
python -m ood_bench.scripts.main $shared_args --envs_p 0 1 2 3 4 --envs_q 5 --output_dir ood_bench/examples/$dataset/outputs/sketch