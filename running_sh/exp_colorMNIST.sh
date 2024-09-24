DSET=ColoredMNIST 
ETHR=0.5
EMAR=0.4
DTHR=0.3
SEED=2024
ROOT=/home/aiotlab/ducntm/DATA/

LRMUL=5
DTHR=0.5
MODEL=resnet18_bn
INTERVAL=30
ETHR=1.0
EMAR=1.0
GPU=0,1,2,3
python3 ../main_update.py --method no_adapt --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --new_criteria False --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method tent --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --new_criteria False --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method sar --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --new_criteria False --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --new_criteria False --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.05 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.1 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist
 
python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.15 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.2 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.25 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.3 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.35 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.4 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.45 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.5 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.55 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.6 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.65 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.7 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.75 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.8 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.85 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.9 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.95 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --new_criteria False --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.05 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.1 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.15 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.2 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.25 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.3 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.35 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.4 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.45 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.5 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.55 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.6 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.65 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.7 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.75 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.8 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.85 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.9 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.95 --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle  --gpu $GPU  --output ../output_mnist