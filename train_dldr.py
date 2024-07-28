# main.py
from trainer.train_vanilla_and_fgsm_twa import get_parser,  main
if __name__ == '__main__':
    parser = get_parser()
    main(parser)