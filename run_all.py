import os


# dropouts = [0.0, 0.2, 0.4, 0.9]
# sen_lens = [128, 256, 512, 1024, 2048]
# accu_grads = [1, 2, 4, 8]
accu_grads = [4, 32, 64, 128]

def call(cmd):
    print(cmd)
    os.system(cmd)

# for dropout in dropouts:
#     exp_name = f"final_dropout-{dropout}"
#     cmd = f"sbatch ada.sh python src/train.py --exp_name {exp_name} --cls_dropout {dropout}"
#     call(cmd)

# for sen_len in sen_lens:
#     exp_name = f"final_sen-len-{sen_len}"
#     cmd = f"sbatch ada.sh python src/train.py --exp_name {exp_name} --max_sen_len {sen_len}"
#     call(cmd)

# for accu_grad in accu_grads:
#     exp_name = f"final_accu_grad-{accu_grad}"
#     cmd = f"sbatch ada.sh python src/train.py --exp_name {exp_name} --accumulate_grad_batches {accu_grad}"
#     call(cmd)

# cmd = f"sbatch ada.sh python src/train.py --exp_name final"
# call(cmd)