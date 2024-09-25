ROOT=aiotlab/DATA

MODEL=vitbase_timm
GPU=1
ETHR=0.5
EMAR=0.4
DTHR=0.3
INTERVAL=100
SEED=2024
EXP=normal
LRMUL=1
NUM_SIM=1001
BATCH=64
###Exp blur
OUTPUT=output/output_imagenet/blur
LEVEL=1
python3 main_update_blur.py --method no_adapt --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method tent --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method sar --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.05 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.1 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.15 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.2 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.25 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.3 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.35 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.4 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.45 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.5 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.55 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.6 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.65 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.7 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.75 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.8 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.85 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.9 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.95 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.05 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.1 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.15 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.2 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.25 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.3 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.35 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.4 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.45 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.5 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.55 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.6 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.65 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.7 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.75 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.8 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.85 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.9 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.95 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

LEVEL=2
python3 main_update_blur.py --method no_adapt --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method tent --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method sar --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.05 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.1 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.15 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.2 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.25 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.3 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.35 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.4 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.45 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.5 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.55 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.6 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.65 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.7 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.75 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.8 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.85 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.9 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.95 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.05 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.1 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.15 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.2 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.25 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.3 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.35 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.4 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.45 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.5 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.55 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.6 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.65 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.7 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.75 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.8 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.85 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.9 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.95 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

LEVEL=3
python3 main_update_blur.py --method no_adapt --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method tent --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method sar --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.05 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.1 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.15 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.2 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.25 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.3 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.35 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.4 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.45 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.5 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.55 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.6 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.65 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.7 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.75 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.8 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.85 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.9 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.95 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.05 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.1 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.15 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.2 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.25 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.3 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.35 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.4 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.45 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.5 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.55 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.6 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.65 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.7 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.75 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.8 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.85 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.9 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.95 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

LEVEL=4
python3 main_update_blur.py --method no_adapt --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method tent --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method sar --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.05 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.1 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.15 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.2 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.25 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.3 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.35 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.4 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.45 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.5 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.55 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.6 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.65 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.7 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.75 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.8 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.85 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.9 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.95 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.05 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.1 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.15 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.2 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.25 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.3 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.35 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.4 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.45 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.5 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.55 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.6 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.65 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.7 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.75 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.8 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.85 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.9 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.95 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

LEVEL=5
python3 main_update_blur.py --method no_adapt --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method tent --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method sar --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.05 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.1 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.15 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.2 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.25 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.3 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.35 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.4 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.45 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.5 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.55 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.6 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.65 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.7 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.75 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.8 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.85 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.9 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.95 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.05 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.1 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.15 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.2 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.25 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.3 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.35 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.4 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.45 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.5 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.55 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.6 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.65 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.7 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.75 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.8 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.85 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.9 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH

python3 main_update_blur.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.95 --output $OUTPUT --level $LEVEL --test_batch_size $BATCH


# #Exp noise
# OUTPUT=output/output_imagenet/noise
# python3 main_update_noise.py --method no_adapt --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT

# python3 main_update_noise.py --method tent --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT

# python3 main_update_noise.py --method sar --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT

# python3 main_update_noise.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT

# python3 main_update_noise.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --new_criteria False --output $OUTPUT

# python3 main_update_noise.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.05 --output $OUTPUT

# python3 main_update_noise.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.1 --output $OUTPUT

# python3 main_update_noise.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.15 --output $OUTPUT

# python3 main_update_noise.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.2 --output $OUTPUT

# python3 main_update_noise.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.25 --output $OUTPUT

# python3 main_update_noise.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.3 --output $OUTPUT

# python3 main_update_noise.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.35 --output $OUTPUT

# python3 main_update_noise.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.4 --output $OUTPUT

# python3 main_update_noise.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.45 --output $OUTPUT

# python3 main_update_noise.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.5 --output $OUTPUT

# python3 main_update_noise.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.55 --output $OUTPUT

# python3 main_update_noise.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.6 --output $OUTPUT

# python3 main_update_noise.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.65 --output $OUTPUT

# python3 main_update_noise.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.7 --output $OUTPUT

# python3 main_update_noise.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.75 --output $OUTPUT

# python3 main_update_noise.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.8 --output $OUTPUT

# python3 main_update_noise.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.85 --output $OUTPUT

# python3 main_update_noise.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.9 --output $OUTPUT

# python3 main_update_noise.py --method eata --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.95 --output $OUTPUT

# python3 main_update_noise.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.05 --output $OUTPUT

# python3 main_update_noise.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.1 --output $OUTPUT

# python3 main_update_noise.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.15 --output $OUTPUT

# python3 main_update_noise.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.2 --output $OUTPUT

# python3 main_update_noise.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.25 --output $OUTPUT

# python3 main_update_noise.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.3 --output $OUTPUT

# python3 main_update_noise.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.35 --output $OUTPUT

# python3 main_update_noise.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.4 --output $OUTPUT

# python3 main_update_noise.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.45 --output $OUTPUT

# python3 main_update_noise.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.5 --output $OUTPUT

# python3 main_update_noise.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.55 --output $OUTPUT

# python3 main_update_noise.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.6 --output $OUTPUT

# python3 main_update_noise.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.65 --output $OUTPUT

# python3 main_update_noise.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.7 --output $OUTPUT

# python3 main_update_noise.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.75 --output $OUTPUT

# python3 main_update_noise.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.8 --output $OUTPUT

# python3 main_update_noise.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.85 --output $OUTPUT

# python3 main_update_noise.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.9 --output $OUTPUT

# python3 main_update_noise.py --method deyo --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type normal --model $MODEL --wandb_interval $INTERVAL --seed $SEED --lr_mul $LRMUL --num_sim $NUM_SIM --alpha_cap 0.95 --output $OUTPUT