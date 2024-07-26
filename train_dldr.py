# main.py
from trainer.train_twa_ddp import get_parser, set_seed, main
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.randomseed)
    main(args)