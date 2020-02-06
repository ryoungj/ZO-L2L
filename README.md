# Learning to learn by zeroth-order oracle
Readme is coming soon...

python main_attack.py --exp_name ZO_attack_mnist --train_task ZOL2L-Attack --gpu_num 0 --train optimizer_attack
python main_attack.py --exp_name VarReduced_ZO_attack_mnist --train_task VarReducedZOL2L-Attack --gpu_num 0 --train optimizer_attack --warm_start_ckpt ./output/ZO_attack_mnist/ckpt_best