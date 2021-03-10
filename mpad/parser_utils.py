import argparse
import torch

def get_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', default='test_experiment',
                        help='Path to the dataset.')
    parser.add_argument('--model-type', default='MPAD',
                        help='Path to the dataset.')
    parser.add_argument('--path-to-dataset', default='datasets/subjectivity.txt',
                        help='Path to the dataset.')
    parser.add_argument('--path-to-test-set', default='datasets/subjectivity.txt',
                        help='Path to the dataset.')
    parser.add_argument('--path-to-embeddings', default='GoogleNews-vectors-negative300.bin',
                        help='Path to the to the word2vec binary file.')
    parser.add_argument('--path-to-word2idx', default='word2idx.json',
                        help='Path to the to the word2vec binary file.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--do-train', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--do-evaluate', action='store_true', default=True,
                        help='Disables CUDA training.')
    parser.add_argument('--pretrained-model', default=None,
                        help='Path to the to a model checkpoint to use for inference or continue training from')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--percentage_dev', type=float, default=0.1,
                        help='Initial learning rate.')
    parser.add_argument('--min-freq-word', type=int, default=1,
                        help='Minimum frequency of a word for it to get its own word embedding.')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--penultimate', type=int, default=64,
                        help='Size of penultimate layer.')
    parser.add_argument('--message-passing-layers', type=int, default=2,
                        help='Number of message passing layers.')
    parser.add_argument('--window-size', type=int, default=2,
                        help='Size of window.')
    parser.add_argument('--directed', action='store_true', default=True,
                        help='Create directed graph of words.')
    parser.add_argument('--use-master-node', action='store_true', default=True,
                        help='Include master node in graph of words.')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize adjacency matrices.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--patience', type=int, default=20,
                        help='Number of epochs to wait if no improvement during training.')
    parser.add_argument('--eval-every', default='epoch',
                        help='Number of update steps after which to start evaluating, if "epoch" is chosen, the full train set will be iterated over.')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    device = 'cuda' if args.cuda else 'cpu'

    return args, device