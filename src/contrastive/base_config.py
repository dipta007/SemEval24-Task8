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
    return parent_parser

def add_model_args(parent_parser):
    parser = parent_parser.add_argument_group("Model Config")
    parser.add_argument("--exp_name", type=str, default="sem8", help="Experiement name?", required=True)
    parser.add_argument("--model_name", type=str, default='sentence-transformers/all-mpnet-base-v2', help="Model name?")
    return parent_parser

def add_trainer_args(parent_parser):
    parser = parent_parser.add_argument_group("Trainer Config")
    parser.add_argument("--max_epochs", type=int, default=-1, help="Number of maximum epochs", )
    parser.add_argument("--validate_every", type=int, default=0.125, help="Number of maximum epochs")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--accumulate_grad_batches", type=int, default=128, help="Number of accumulation of grad batches")
    return parent_parser

def get_config():
    parser = add_program_args()
    parser = add_data_args(parser)
    parser = add_model_args(parser)
    parser = add_trainer_args(parser)
    cfg    = parser.parse_args()
    return cfg