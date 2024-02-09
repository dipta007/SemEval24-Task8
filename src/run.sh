# sbatch ada.sh \
#     python train.py \
#         --batch_size=32 \
#         --exp_name=senformer_adamw

# sbatch ada.sh \
#     python train.py \
#         --batch_size=32 \
#         --loss_weight_gen_text=1.0 \
#         --exp_name=senformer_gen_text

# sbatch ada.sh \
#     python train.py \
#         --batch_size=32 \
#         --loss_weight_gen_text=0.5 \
#         --exp_name=senformer_gloss_0.5

# sbatch ada.sh python train.py --accumulate_grad_batches=32 --cls_dropout=0.6 --loss_weight_con=0.7 --loss_weight_gen_text=0.8 --loss_weight_text=1 --lr=0.00004693643463272506 --weight_decay=0.01 --encoder_type=sen --model_name='jpwahle/longformer-base-plagiarism-detection' --batch_size=2 --max_sen_len=4096 --exp_name=best_sweep_1.1

# sbatch ada.sh python train.py --accumulate_grad_batches=8 --cls_dropout=0.3 --loss_weight_con=0.2 --loss_weight_gen_text=0.9 --loss_weight_text=0.5 --lr=0.00001228049804238282 --weight_decay=0.001 --encoder_type=sen --model_name="jpwahle/longformer-base-plagiarism-detection" --batch_size=2 --max_sen_len=4096 --exp_name=best_sweep_2.1

# sbatch ada.sh python train.py --accumulate_grad_batches=16 --cls_dropout=0.5 --loss_weight_con=0.8 --loss_weight_gen_text=0.6 --loss_weight_text=0.5 --lr=0.00002560204941817421 --weight_decay=0 --encoder_type=sen --model_name=jpwahle/longformer-base-plagiarism-detection --batch_size=2 --max_sen_len=4096 --exp_name=best_sweep_3

# sbatch ada.sh python train.py --accumulate_grad_batches=8 --cls_dropout=0.3 --loss_weight_con=0.5 --loss_weight_gen_text=0.1 --loss_weight_text=1.5 --lr=0.00003364843101428523 --weight_decay=0.0001 --encoder_type=sen --model_name=jpwahle/longformer-base-plagiarism-detection --batch_size=2 --max_sen_len=4096 --exp_name=best_sweep_4

# sbatch ada.sh python train.py --encoder_type=sen --model_name="jpwahle/longformer-base-plagiarism-detection" --batch_size=2 --max_sen_len=4096 --accumulate_grad_batches=8 --cls_dropout=0.3 --loss_weight_con=0.9 --loss_weight_gen_text=0.2 --loss_weight_text=0.6 --lr=0.00001 --weight_decay=0.0 --exp_name=best_1

# sbatch ada.sh python train.py --encoder_type=sen --model_name=jpwahle/longformer-base-plagiarism-detection --batch_size=2 --max_sen_len=4096 --accumulate_grad_batches=16 --cls_dropout=0.6 --loss_weight_con=0.7 --loss_weight_gen_text=0.1 --loss_weight_text=0.8 --lr=0.00001 --weight_decay=0.0 --exp_name=best_2


# sbatch ada.sh python train.py --encoder_type=sen --model_name="jpwahle/longformer-base-plagiarism-detection" --batch_size=4 --max_sen_len=4096 --accumulate_grad_batches=8 --cls_dropout=0.3 --loss_weight_con=0.9 --loss_weight_gen_text=0.2 --loss_weight_text=0.6 --lr=0.00001 --weight_decay=0.0 --validate_every=1.0 --exp_name=best_1_m16

# need to run best_2_m16
sbatch ada.sh python train.py --encoder_type=sen --model_name=jpwahle/longformer-base-plagiarism-detection --batch_size=4 --max_sen_len=4096 --accumulate_grad_batches=16 --cls_dropout=0.6 --loss_weight_con=0.7 --loss_weight_gen_text=0.1 --loss_weight_text=0.8 --lr=0.00001 --weight_decay=0.0 --validate_every=1.0 --exp_name=best_2_m16