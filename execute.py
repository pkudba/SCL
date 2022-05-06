import yaml

from runners.segmentation_run import SegmentationRunner
from runners.classification_run import ClassifierRunner
from runners.train_siamese import SiameseNetRunner
from runners.train_backbone import Trainer
from tools.config import get_parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(args).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)
    args = parser.parse_args()

    # test script
    if args.mode == 'segmentation':
        runner = SegmentationRunner(args)
    elif args.mode == 'classification':
        runner = ClassifierRunner(args)
    elif args.mode == 'train':
        runner = Trainer(args)
    elif args.mode == 'train-siamese':
        runner = SiameseNetRunner(args)
    else:
        raise ValueError('Unknown phase.')

    runner.run(args)


if __name__ == '__main__':
    main()
