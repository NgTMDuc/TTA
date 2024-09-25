DSET=Waterbirds 
ETHR=0.5
EMAR=0.4
DTHR=0.3
SEED=2024

LRMUL=5
DTHR=0.5
MODEL=resnet50_bn_torch
INTERVAL=10
ETHR=1.0
EMAR=1.0
GPU=0,1,2,3
# python3 ../main_update.py --method no_adapt --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --new_criteria False --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

# python3 ../main_update.py --method tent --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --new_criteria False --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

# python3 ../main_update.py --method sar --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --new_criteria False --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

# python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --new_criteria False --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.05 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.1 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.15 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.2 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.25 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.3 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.35 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.4 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.45 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.5 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.55 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.6 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.65 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.7 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.75 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.8 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.85 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.9 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method eata --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.95 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

# python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --new_criteria False --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.05 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.1 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.15 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.2 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.25 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.3 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.35 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.4 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.45 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.5 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.55 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.6 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.65 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.7 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.75 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.8 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.85 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.9 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird

python3 ../main_update.py --method deyo --dset $DSET --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --num_sim 3 --alpha_cap 0.95 --gpu $GPU --pretrained_path /home/aiotlab/ducntm/DeYO/pretrained/waterbirds_pretrained_model.pickle --output ../output_waterbird