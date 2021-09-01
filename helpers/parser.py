import argparse

def parse():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-data_dir', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default='./dataset', \
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
    parser.add_argument('--run_number', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default=0, \
                        type=int, \
                        choices=None, \
                        help='Run number', \
                        metavar=None)
    parser.add_argument('--padding_mode', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default='reflect', \
                        type=str, \
                        choices=None, \
                        help='Padding type (default: reflect)', \
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
    parser.add_argument('--lr', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default=0.0002, \
                        type=float, \
                        choices=None, \
                        help='Learning rate', \
                        metavar=None)
    parser.add_argument('--beta_1', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default=0.9, \
                        type=float, \
                        choices=None, \
                        help='beta_1 for Adam', \
                        metavar=None)
    parser.add_argument('--beta_2', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default=0.999, \
                        type=float, \
                        choices=None, \
                        help='beta_2 for Adam', \
                        metavar=None)
    # Used for inference
    parser.add_argument('--inference_mode', \
                        action='store_true', \
                        default=False, \
                        help='train or inference')
    parser.add_argument('-state_file_dir', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default='./', \
                        type=str, \
                        choices=None, \
                        help='directory storing torch state files', \
                        metavar=None)
    parser.add_argument('--load_nth_state_file', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default=0, \
                        type=int, \
                        choices=None, \
                        help='nth state file to load', \
                        metavar=None)

    args = parser.parse_args()

    return args
