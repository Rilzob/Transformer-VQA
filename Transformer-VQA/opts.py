import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--rnn_size', type=int, default=1280,
                        help='size of the rnn in number of hidden nodes in question gru')
    parser.add_argument('--num_hid', type=int, default=1280,
                        help='size of the rnn in number of hidden nodes in question gru')
    parser.add_argument('--num_layers', type=int, default=2,
                    help='number of GCN layers')
    parser.add_argument('--rnn_type', type=str, default='gru',
                    help='rnn, gru, or lstm')
    parser.add_argument('--v_dim', type=int, default=2048,
                    help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--ans_dim', type=int, default=2274,
                    help='3219 for VQA-CP v2, 2185 for VQA-CP v1')
    parser.add_argument('--logit_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--activation', type=str, default='ReLU',
                        help='number of layers in the RNN')
    parser.add_argument('--norm', type=str, default='weight',
                        help='number of layers in the RNN')
    parser.add_argument('--initializer', type=str, default='kaiming_normal',
                        help='number of layers in the RNN')

    # Optimization: General
    parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs')

    parser.add_argument('--s_epoch', type=int, default=0,
                    help='training from s epochs')
    parser.add_argument('--pretrain_epoches', type=int, default=10,
                    help='number of epochs for normal training')
    parser.add_argument('--ratio', type=float, default=1,
                    help='ratio of training set used')
    parser.add_argument('--ml_loss', dest='ml_loss', action='store_true')
    parser.add_argument('--ce_loss', dest='ce_loss', action='store_true')

    parser.add_argument('--batch_size', type=int, default=64,
                    help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=0.25,
                    help='clip gradients at this value')
    parser.add_argument('--dropC', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--dropG', type=float, default=0.2,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--dropL', type=float, default=0.1,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--dropW', type=float, default=0.4,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--dropout', type=float, default=0.2,
                    help='strength of dropout in the Language Model RNN')

    #Optimization: for the Language Model
    parser.add_argument('--optimizer', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='learning rate')
#     parser.add_argument('--self_loss_weight', type=float, default=3,
#                     help='self-supervised loss weight')
    parser.add_argument('--self_loss_weight', type=float, default=6,
                    help='self-supervised loss weight')


    parser.add_argument('--optim_alpha', type=float, default=0.9,
                    help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0.2,
                    help='weight_decay')
    parser.add_argument('--seed', type=int, default=1024,
                    help='seed')
    parser.add_argument('--ntokens', type=int, default=777,
                    help='ntokens')
    
    parser.add_argument('--dataroot', type=str, default='./data/vqacp2/',help='dataroot')
    parser.add_argument('--img_root', type=str, default='./data/coco/',help='image_root')
                    
    parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='directory to store checkpointed models')


    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default='saved_models_cp2/')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--logits', action='store_true')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--label', type=str, default='best')

    parser.add_argument('--HIDDEN_SIZE', type=int, default=128)
    parser.add_argument('--MULTI_HEAD', type=int, default=8)
    parser.add_argument('--FLAT_MLP_SIZE', type=int, default=128)
    parser.add_argument('--FLAT_GLIMPSES', type=int, default=1)
    parser.add_argument('--FLAT_OUT_SIZE', type=int, default=256)
    # FeedForwardNet size in every MCA layer
    parser.add_argument('--FF_SIZE', type=int, default=512)
    parser.add_argument('--WORD_EMBED_SIZE', type=int, default=300)
    parser.add_argument('--IMG_FEAT_SIZE', type=int, default=2048)
    parser.add_argument('--LAYER', type=int, default=1)
    parser.add_argument('--HIDDEN_SIZE_HEAD', type=int, default=16, help='HIDDEN_SIZE_HEAD = HIDDEN_SIZE / MULTI_HEAD')
    # Adam optimizer betas and eps
    parser.add_argument('--OPT_BETAS', default=(0.9, 0.98))
    parser.add_argument('--OPT_EPS', type=int, default=1e-9)

    parser.add_argument('--LD_DECAY_LIST', type=list, default=[10, 12, 16, 20, 24, 28])
    parser.add_argument('--LD_DECAY_R', type=float, default=0.2)

    parser.add_argument('--WEIGHT_DECAY', type=float, default=0.001)

    args = parser.parse_args()

    return args
