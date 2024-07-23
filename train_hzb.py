from trainer import get_parser, start_ddp
if __name__ == '__main__':
    parser = get_parser()
    start_ddp(parser)