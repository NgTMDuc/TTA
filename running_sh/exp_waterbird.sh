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

python3 ../main_update.py --method no_adapt --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --new_criteria False

python3 ../main_update.py --method tent --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --new_criteria False

python3 ../main_update.py --method sar --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --new_criteria False

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --new_criteria False

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.05

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.1

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.15

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.2

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.25

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.3

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.35

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.4

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.45

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.5

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.55

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.6

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.65

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.7

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.75

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.8

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.85

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.9

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.95

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --new_criteria False

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.05

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.1

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.15

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.2

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.25

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.3

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.35

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.4

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.45

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.5

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.55

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.6

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.65

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.7

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.75

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.8

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.85

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.9

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.95