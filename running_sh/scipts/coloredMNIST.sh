export LC_NUMERIC="C"

DSET=ColoredMNIST 
SEED=2024
GPU=0
ROOT=/mnt/disk2/ducntm/DATA
LRMUL=5
OUTPUT=/mnt/ducntm/TTA/output_colorMNIST_10_10/
INTERVAL=10
BATCH=64
MODEL=resnet18_bn
PRETRAINED=/mnt/disk2/ducntm/DeYO/pretrained/ColoredMNIST_model.pickle
EXP_TYPE=spurious
#EATA
EATA_FISHERS=1
FISHER_SIZE=2000
FISHER_ALPHA=2000
E_MARGIN=0.4
D_MARGIN=0.5

#SAR
SAR_MARGIN_E0=0.4
IMBALANCE_RATIO=500000

#DEYO
AUG_TYPE=patch
DEYO_MARGIN=0.5
DEYO_MARGIN_0=0.4
THRSH=0.5

FILTER_ENT=1
FILTER_PLPD=1
REWEIGHT_ENT=1
REWEIGHT_PLPD=1

# python3 ../../main_update.py --exp_type $EXP_TYPE --method no_adapt --dset $DSET --wandb_interval $INTERVAL --model $MODEL --seed $SEED --pretrained_path $PRETRAINED  --gpu $GPU --output $OUTPUT --data_root $ROOT --lr_mul $LRMUL

# python3 ../../main_update.py --exp_type $EXP_TYPE --method tent --dset $DSET --wandb_interval $INTERVAL --model $MODEL --seed $SEED --pretrained_path $PRETRAINED  --gpu $GPU --output $OUTPUT --data_root $ROOT --lr_mul $LRMUL

# python3 ../../main_update.py --exp_type $EXP_TYPE --method eata --dset $DSET --wandb_interval $INTERVAL --model $MODEL --seed $SEED --pretrained_path $PRETRAINED  --gpu $GPU --output $OUTPUT --data_root $ROOT --fisher_alpha $FISHER_ALPHA --e_margin $E_MARGIN --d_margin $D_MARGIN --fisher_size $FISHER_SIZE --eata_fishers $EATA_FISHERS --lr_mul $LRMUL

# python3 ../../main_update.py --exp_type $EXP_TYPE --method sar --dset $DSET --wandb_interval $INTERVAL --model $MODEL --seed $SEED --pretrained_path $PRETRAINED --gpu $GPU --output $OUTPUT --data_root $ROOT --sar_margin_e0  $SAR_MARGIN_E0 --imbalance_ratio $IMBALANCE_RATIO --lr_mul $LRMUL

# python3 ../../main_update.py --exp_type $EXP_TYPE --method deyo --dset $DSET --wandb_interval $INTERVAL --model $MODEL --seed $SEED --pretrained_path $PRETRAINED --gpu $GPU --output $OUTPUT --data_root $ROOT --deyo_margin $DEYO_MARGIN --deyo_margin_e0 $DEYO_MARGIN_0 --plpd_threshold $THRSH --filter_ent $FILTER_ENT --filter_plpd $FILTER_PLPD --reweight_ent $REWEIGHT_ENT --reweight_plpd $REWEIGHT_PLPD  --lr_mul $LRMUL

#Propose
K=3

START=0.9
END=0.99
STEP=0.01

for i in $(seq $START $STEP $END)
do 
    ALPHA_CAP=$(printf "%.2f" $i)
    python3 ../../main_update.py --exp_type $EXP_TYPE --method deyo --dset $DSET --wandb_interval $INTERVAL --model $MODEL --seed $SEED --pretrained_path $PRETRAINED --gpu $GPU --output $OUTPUT --data_root $ROOT --deyo_margin $DEYO_MARGIN --deyo_margin_e0 $DEYO_MARGIN_0 --plpd_threshold $THRSH --filter_ent $FILTER_ENT --filter_plpd $FILTER_PLPD --reweight_ent $REWEIGHT_ENT --reweight_plpd $REWEIGHT_PLPD  --lr_mul $LRMUL --new_criteria True --alpha_cap $ALPHA_CAP --num_sim $K
    python3 ../../main_update.py --exp_type $EXP_TYPE --method eata --dset $DSET --wandb_interval $INTERVAL --model $MODEL --seed $SEED --pretrained_path $PRETRAINED --gpu $GPU --output $OUTPUT --data_root $ROOT --deyo_margin $DEYO_MARGIN --deyo_margin_e0 $DEYO_MARGIN_0 --plpd_threshold $THRSH --filter_ent $FILTER_ENT --filter_plpd $FILTER_PLPD --reweight_ent $REWEIGHT_ENT --reweight_plpd $REWEIGHT_PLPD  --lr_mul $LRMUL --new_criteria True --alpha_cap $ALPHA_CAP --num_sim $K
done