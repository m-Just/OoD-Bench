data_dir=$1
dataset="ImageNet_R"
python -m ood_bench.scripts.main --n_trials 8 --parallel --data_dir $data_dir --dataset $dataset --envs_p 0 --envs_q 1 --backbone resnet-18 --pretrained_model_path auto --output_dir ood_bench/examples/$dataset/outputs --calibrate