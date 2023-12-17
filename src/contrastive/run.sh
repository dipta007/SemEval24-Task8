sbatch ada.sh \
    python train.py \
        --batch_size=32 \
        --exp_name=senformer_adamw

sbatch ada.sh \
    python train.py \
        --batch_size=32 \
        --loss_weight_gen_text=1.0 \
        --exp_name=senformer_gen_text

sbatch ada.sh \
    python train.py \
        --batch_size=32 \
        --loss_weight_gen_text=0.5 \
        --exp_name=senformer_gloss_0.5
