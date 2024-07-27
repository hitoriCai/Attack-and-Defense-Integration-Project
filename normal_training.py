from trainer.training_vanilla_and_fgsm import get_parser_normal, start_ddp


if __name__ == "__main__":
    parser = get_parser_normal()
    start_ddp(parser)