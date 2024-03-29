import argparse

def add_program_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction, required=False, default=False, help='debug?')
    parser.add_argument("--seed", type=int, default=42, help="value for reproducibility") 
    parser.add_argument("--cuda", type=bool, default=True, action=argparse.BooleanOptionalAction, help="Use CUDA?")
    return parser

def add_data_args(parent_parser):
    parser = parent_parser.add_argument_group("Data Config")
    parser.add_argument("--data_dir", type=str, default="./data/", help="Data directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size?")
    parser.add_argument("--max_doc_len", type=int, default=64, help="Max doc length?")
    parser.add_argument("--max_sen_len", type=int, default=4096, help="Max sen length?")
    return parent_parser

def add_model_args(parent_parser):
    parser = parent_parser.add_argument_group("Model Config")
    parser.add_argument("--exp_name", type=str, default="sem8", help="Experiement name?", required=True)
    parser.add_argument("--model_name", type=str, default='jpwahle/longformer-base-plagiarism-detection', help="Model name?")
    parser.add_argument("--loss_weight_con", type=float, default=0.7, help="Contrastive loss weight?")
    parser.add_argument("--loss_weight_text", type=float, default=0.8, help="Text loss weight?")
    parser.add_argument("--loss_weight_gen_text", type=float, default=0.1, help="Gen text loss weight?")
    parser.add_argument("--weight_decay", type=float, default=0.00, help="Weight decay?")
    parser.add_argument("--encoder_type", type=str, default="sen", help="Encoder type? [sen, doc]")
    parser.add_argument("--cls_dropout", type=float, default=0.6, help="CLS dropout?")
    return parent_parser

def add_trainer_args(parent_parser):
    parser = parent_parser.add_argument_group("Trainer Config")
    parser.add_argument("--max_epochs", type=int, default=-1, help="Number of maximum epochs", )
    parser.add_argument("--validate_every", type=float, default=0.04, help="Number of maximum epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--accumulate_grad_batches", type=int, default=16, help="Number of accumulation of grad batches")
    parser.add_argument("--overfit", type=int, default=0, help="Overfit batches")
    return parent_parser

def get_config():
    parser = add_program_args()
    parser = add_data_args(parser)
    parser = add_model_args(parser)
    parser = add_trainer_args(parser)
    cfg    = parser.parse_args()
    return cfg