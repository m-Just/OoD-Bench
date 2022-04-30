data_dir=$1
dataset="CelebA_Blond"
python -m ood_bench.scripts.main --n_trials 8 --parallel --data_dir $data_dir --dataset $dataset --envs_p 0 1 --envs_q 2 --backbone resnet-18 --pretrained_model_path auto --output_dir ood_bench/examples/$dataset/outputs --calibrate