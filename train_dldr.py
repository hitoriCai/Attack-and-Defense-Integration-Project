# main.py
from trainer.train_twa_ddp import get_parser,  main
if __name__ == '__main__':
    parser = get_parser()
    main(parser)