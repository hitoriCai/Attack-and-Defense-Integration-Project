from query_attack import *
from query.victim import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--model', default='resnext101_32x8d', type=str,
                        help='[inception_v3, mnasnet1_0, resnext101_32x8d] for ImageNet')
    parser.add_argument('--l2_attack', action='store_true', help='perform l2 attack')
    parser.add_argument('--eps', type=float, default=16, help='the attack bound')
    parser.add_argument('--num_iter', type=int, default=10000, help='maximum query times.')

    parser.add_argument('--num_x', type=int, default=10000, help='number of samples for evaluation.')
    parser.add_argument('--num_srg', type=int, default=0, help='number of surrogates.')
    parser.add_argument('--use_nas', action='store_true', help='use NAS to train the surrogate.')
    parser.add_argument('--use_square_plus', action='store_true', help='use Square+.')
    
    parser.add_argument('--p_init', type=float, default=0.05, help='hyperparameter of Square, the probability of changing a coordinate.')
    parser.add_argument('--gpu', type=str, default='1', help='GPU number(s).')
    parser.add_argument('--run_times', type=int, default=1, help='repeated running time.')
    parser.add_argument('--seed', type=int, default=-1, help='random seed')

    args = parser.parse_args()
    if args.use_nas: assert args.num_srg > 0
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    log = Logger('')
    
    for model_name in args.model.split(','):
        if model_name in ['inception_v3', 'mnasnet1_0', 'resnext101_32x8d']:         dataset = 'imagenet'
        else: raise ValueError('Invalid Victim Name!')

        # imagenet
        assert (not args.use_nas), 'NAS is not supported for ImageNet for resource concerns'
        if not ((args.l2_attack and args.eps == 5) or (not args.l2_attack and args.eps == 12.75)):
            print('Warning: not using default eps in the paper, which is l2=5 or linfty=12.75 for ImageNet.')
        batch_size = 100 if model_name != 'resnext101_32x8d' else 32
        model = VictimImagenet(model_name, batch_size=batch_size) if model_name != 'easydlmnist' else VictimEasydl(arch='easydlmnist')
        x_test, y_test = load_imagenet(args.num_x, model)

        logits_clean = model(x_test)
        corr_classified = logits_clean.argmax(1) == y_test.argmax(1)
        print('Clean accuracy: {:.2%}'.format(np.mean(corr_classified)) + ' ' * 40)
        y_test = dense_to_onehot(y_test.argmax(1), n_cls=10 if dataset != 'imagenet' else 1000)
        for run_time in range(args.run_times):
            attack(model, x_test[corr_classified], y_test[corr_classified], logits_clean[corr_classified], dataset, batch_size, run_time, args, log)
