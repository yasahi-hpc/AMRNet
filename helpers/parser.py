import argparse

def parse():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-data_dir', \
                        action='store', \
                        nargs=None, \
                        const=None, \
                        default=None, \
                        type=str, \
                        choices=None, \
                        help='directory of datasets', \
                        metavar=None)
    parser.add_argument('--device', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default='cuda', \
                        type=str, \
                        choices=None, \
                        help='Device to run the model', \
                        metavar=None)
    parser.add_argument('--dim', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default=2, \
                        type=int, \
                        choices=None, \
                        help='Dimension of the model', \
                        metavar=None)
    parser.add_argument('--batch_size', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default=25, \
                        type=int, \
                        choices=None, \
                        help='Batch size', \
                        metavar=None)
    parser.add_argument('--n_epochs', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default=3, \
                        type=int, \
                        choices=None, \
                        help='Number of epochs', \
                        metavar=None)
    parser.add_argument('--padding_mode', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default='replicate', \
                        type=str, \
                        choices=None, \
                        help='Padding type (default: replicate)', \
                        metavar=None)
    parser.add_argument('--preprocess_type', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default='normalization', \
                        type=str, \
                        choices=None, \
                        help='Preprocess type (default: normalization)', \
                        metavar=None)
    parser.add_argument('--model_name', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default='AMR_Net', \
                        type=str, \
                        choices=None, \
                        help='Name of the model (default: AMR_Net)', \
                        metavar=None)

    args = parser.parse_args()

    return args
