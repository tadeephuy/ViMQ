import argparse

from trainer import Trainer

from utils import MODEL_CLASSES, MODEL_PATH_MAP, init_logger, set_seed

import torch



def main(args):

    init_logger()
    set_seed(args)
    trainer = Trainer(args)

    if args.do_train:
        trainer.train()
    
    if args.do_eval:
        trainer.load_model()
        trainer.evaluate('dev')
        trainer.evaluate('test')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # DATA
    parser.add_argument('--data_dir', required=True, type=str, help='The input data direction')
    parser.add_argument('--file_name_entity_set', default='entity_set.txt', type=str, help='The file name of entity set')
    parser.add_argument('--file_name_char2index', default='char2index.json', type=str, help='The file name of character')


    # HYPER PARAMETER
    parser.add_argument('--iternoise', required=True, type=int)
    parser.add_argument('--omega', required=True, type=int)
    parser.add_argument('--lamda', required=True, type=int)
    parser.add_argument('--threshold_iou', required=True, type=float)
    
    
    parser.add_argument(
        "--max_seq_len", default=120, type=int, help="The maximum total input sequence length after tokenization."
    )
    parser.add_argument(
        "--max_char_len", default=20, type=int, help="The maximum total input character length of word"
    )
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument('--use_char', type=bool, default=False, help='Use char embedding')
    parser.add_argument(
        "--char_hidden_dim", type=int, default=128, help="hidden size of attention output vector"
    )
    parser.add_argument(
        "--char_embedding_dim", type=int, default=100, help="Embedding size of Character embedding"
    )
    parser.add_argument(
        "--char_vocab_size", type=int, default=108, help="Vocab size of Character embedding"
    )

    parser.add_argument(
        '--num_layer_bert', type=int, default=4, help='Number of last - layers of transformers is used contexted representation'
    )
    parser.add_argument(
        '--hidden_dim', default=728, type=int
    )
    parser.add_argument(
        '--hidden_dim_ffw', default=300, type=int
    )
    parser.add_argument(
        '--num_labels', default=5, type=int
    )



    # HYPER-PARAMETER-TRAINING
    parser.add_argument("--seed", type=int, default=1, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=64, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument(
        "--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--num_iteration", default=2, type=int, help="Total number of iteration to perform."
    )
    # HYPER-PARAMETER-OPTIMIZER
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")



    # TOKENIZER 
    parser.add_argument('--pad_char', type=str, default='PAD', help='PAD character for character - level (to be ignore when calculate loss)')


    # LOGGING
    parser.add_argument("--logging_steps", type=int, default=200, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every X updates steps.")

    # USEAGE
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument(
        "--model_type",
        default="vimq_model",
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")

    parser.add_argument("--tuning_metric", default="f1_score", type=str, help="Metrics to tune when training")
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=50,
        help="Number of unincreased validation step to wait for early stopping",
    )

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--gpu_id", type=int, default=0, help="Select gpu id")

    # init pretrained
    parser.add_argument("--pretrained", action="store_true", help="Whether to init model from pretrained base model")
    parser.add_argument("--pretrained_path", default="./ViMQ", type=str, help="The pretrained model path")
 
    args = parser.parse_args()

    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]

    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    main(args)
