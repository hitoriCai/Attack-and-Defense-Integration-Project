from trainer.fgsm_at import get_parser_fgsm, start_ddp_fgsm
if __name__ == '__main__':
    parser = get_parser_fgsm()
    start_ddp_fgsm(parser)