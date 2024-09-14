import argparse
from utils import set_seed

parser = argparse.ArgumentParser(description='Data Pruning')
parser.add_argument('--seed', default=3407, type=int, help='random seed')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
parser.add_argument('--ckpt-dir', default='./checkpoints', type=str, help='checkpoints root directory')
parser.add_argument('--data-dir', default='./data', type=str, help='data root directory')
parser.add_argument('--pruning-rate', default=0.3, type=float, help='pruning rate')
parser.add_argument('--pruning-method', default='prune', type=str, help='pruning method')
parser.add_argument('--stop-epoch', default=100, type=int, help='stop epoch')
parser.add_argument('--topK', default=10, type=int, help='top K epochs')
parser.add_argument('--num-stage', default=3, type=int, help='number of training stages')

parser.add_argument('--lr', default=0.2, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                    help='SGD weight decay (default: 5e-4)')
parser.add_argument('--optimizer',type=str,default='lars',
                    help='different optimizers')
parser.add_argument('--label-smoothing',type=float,default=0.1)
# onecycle scheduling arguments
parser.add_argument('--max-lr',default=0.1,type=float)
parser.add_argument('--div-factor',default=25,type=float)
parser.add_argument('--final-div',default=10000,type=float)
parser.add_argument('--num-epoch',default=300,type=int)
parser.add_argument('--pct-start',default=0.3,type=float)
parser.add_argument('--shuffle', default=True, action='store_true')
parser.add_argument('--ratio',default=0.3,type=float)
parser.add_argument('--delta',default=0.875,type=float)
parser.add_argument('--model',default='r18',type=str)
args = parser.parse_args()

set_seed(args.seed)