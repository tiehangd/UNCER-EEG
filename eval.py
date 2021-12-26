'''Train EEGNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import argparse
import os
import time

from contrib import adf

from models.eegnet import EEGNet
from utils import progress_bar
from models.eegnet_dropout import EEGNet_Dropout
from models_adf.eegnet_adf import EEGNet_adf
from models_adf.eegnet_dropout_adf import EEGNet_dropout_adf
import os
#from dataloader.dataset_loader_BCI_IV_c import DatasetLoader_BCI_IV_subjects as Dataset
from dataloader.dataset_loader_BCI_IV_i import DatasetLoader_BCI_IV_subjects as Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score
from scipy.io import savemat

os.environ['CUDA_VISIBLE_DEVICES']='2,3'
# Model flags
parser = argparse.ArgumentParser(description='PyTorch EEGNet Training')
parser.add_argument('--p', default=0.2, type=float, help='dropout rate')
parser.add_argument('--num_samples', default=10, type=int, help='number of samples to collect with Monte Carlo dropout')
parser.add_argument('--noise_variance', default=1e-4, type=float, 
                    help='noise variance')
parser.add_argument('--min_variance', default=1e-4, type=float, 
                    help='min variance')
parser.add_argument('--tau', default=1e-4, type=float, 
                    help='constant data variance for Monte Carlo dropout.')

# Testing flags
parser.add_argument('--load_model_name', default='eegnet', type=str,  
                    help='model to load')
parser.add_argument('--test_model_name', default='eegnet_dropout', type=str,  
                    help='model to test')
parser.add_argument('--resume', '-r', action='store_true', default=True, 
                    help='resume from checkpoint')
parser.add_argument('--show_bar', '-b', action='store_true', default=True, 
                    help='show bar or not')
parser.add_argument('--batch_size', default=12, type=int,
                    help='size of batch size')
parser.add_argument('--verbose', '-v', action='store_true', default=True, 
                    help='regulate output verbosity')
parser.add_argument('--use_mcdo', '-m', action='store_true', default=False,  
                    help='use Monte Carlo dropout to compute predictions and'
                    'model uncertainty estimates.')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

trainset = Dataset('train', args)
trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

testset = Dataset('test', args)
testloader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)


# Model
if args.verbose: print('==> Building model...')

def model_loader():
    model = {'eegnet': EEGNet,
             'eegnet_dropout': EEGNet_Dropout,
             'eegnet_adf':EEGNet_adf,
             'eegnet_dropout_adf': EEGNet_dropout_adf
             }

    params = {'eegnet': [args.p],
             'eegnet_dropout': [args.p],
             'eegnet_adf': [args.noise_variance, args.min_variance],
             'eegnet_dropout_adf': [args.p, args.noise_variance, args.min_variance]
             }   
 
    return model[args.test_model_name.lower()](*params[args.test_model_name.lower()])

net = model_loader().to(device)
criterion = nn.CrossEntropyLoss()

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    if args.verbose: print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    
    model_to_load = args.load_model_name.lower()
    # if model_to_load.endswith('adf'):
    #     model_to_load = model_to_load[0:-4]
    ckpt_path = './checkpoint/ckpt_{}.pth'.format(model_to_load)
    checkpoint = torch.load(ckpt_path)
    if args.verbose: print('Loaded checkpoint at location {}'.format(ckpt_path))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

def set_training_mode_for_dropout(net, training=True):
    """Set Dropout mode to train or eval."""

    for m in net.modules():
#        print(m.__class__.__name__)
        if m.__class__.__name__.startswith('Dropout'):
            if training==True:
                m.train()
            else:
                m.eval()
    return net        

def one_hot_pred_from_label(y_pred, labels):
    y_true = torch.zeros_like(y_pred)
    ones = torch.ones_like(y_pred)
    indexes = [l for l in labels]
    y_true[torch.arange(labels.size(0)), indexes] = ones[torch.arange(labels.size(0)), indexes]
    
    return y_true

def compute_log_likelihood(y_pred, y_true, sigma):
    dist = torch.distributions.normal.Normal(loc=y_pred, scale=sigma)
    log_likelihood = dist.log_prob(y_true)
    log_likelihood = torch.mean(log_likelihood, dim=1)
    return log_likelihood

def compute_brier_score(y_pred, y_true):
    """Brier score implementation follows 
    https://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf.
    The lower the Brier score is for a set of predictions, the better the predictions are calibrated."""        
        
    brier_score = torch.mean((y_true-y_pred)**2, 1)
    return brier_score

def compute_preds(net, inputs, use_adf=False, use_mcdo=False):
    
    model_variance = None
    data_variance = None
    
    def keep_variance(x, min_variance):
        return x + min_variance

    keep_variance_fn = lambda x: keep_variance(x, min_variance=args.min_variance)
    softmax = nn.Softmax(dim=1)
    adf_softmax = adf.Softmax(dim=1, keep_variance_fn=keep_variance_fn)
    
    net.eval()
    if use_mcdo:
        net = set_training_mode_for_dropout(net, True)
        outputs = [net(inputs) for i in range(args.num_samples)]
        
        if use_adf:
            outputs = [adf_softmax(*outs) for outs in outputs]
            outputs_mean = [mean for (mean, var) in outputs]
            data_variance = [var for (mean, var) in outputs]
            data_variance = torch.stack(data_variance)
            data_variance = torch.mean(data_variance, dim=0)
        else:
            outputs_mean = [softmax(outs) for outs in outputs]
            
        outputs_mean = torch.stack(outputs_mean)
        model_variance = torch.var(outputs_mean, dim=0)
        # Compute MCDO prediction
        outputs_mean = torch.mean(outputs_mean, dim=0)
    else:
        outputs = net(inputs)
        if adf:
            outputs_mean, data_variance = adf_softmax(*outputs)
        else:
            outputs_mean = outputs
        
    net = set_training_mode_for_dropout(net, False)
    
    return outputs_mean, data_variance, model_variance


def evaluate(net, use_adf=False, use_mcdo=False):

    def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return roc_auc_score(y_test, y_pred, average=average)


    def ece_score(logits, labels, n_bins=30):
        confidences, predictions = torch.max(logits, 1)
        ece = torch.zeros(1, device=logits.device)
        accuracies = predictions.eq(labels)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece 


    net.eval()
    test_loss = 0
    correct = 0
    brier_score = 0
    neg_log_likelihood = 0
    auc=0
    ece=0
    total = 0
    outputs_variance = None
    print("use adf")
    print(use_adf)

    pred=[]
    label=[] 
    ece_l=[]
    mean_l=[]
    acc_l=[]
    nll_l=[]
    brier_l=[]


    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):

            mean_l.append(torch.mean(inputs)) 
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs_mean, data_variance, model_variance = compute_preds(net, inputs, use_adf, use_mcdo)
            #print("outputs_mean")
            #print(outputs_mean.shape)
            if data_variance is not None and model_variance is not None:
                outputs_variance = data_variance + model_variance
#                print("outputs_variance")
#                print(outputs_variance)
            elif data_variance is not None:
                outputs_variance = data_variance
            elif model_variance is not None:
                outputs_variance = model_variance + args.tau
                #print("model variance")
                #print(model_variance.shape)

            one_hot_targets = one_hot_pred_from_label(outputs_mean, targets)
            
            # Compute negative log-likelihood (if variance estimate available)
            if outputs_variance is not None:
                batch_log_likelihood = compute_log_likelihood(outputs_mean, one_hot_targets, outputs_variance)
                batch_neg_log_likelihood = -batch_log_likelihood
                # Sum along batch dimension
                neg_log_likelihood += torch.sum(batch_neg_log_likelihood, 0).cpu().numpy().item()
                nll_l.append(torch.sum(batch_neg_log_likelihood, 0).cpu().numpy().item()/targets.size(0))
            
            # Compute brier score
            batch_brier_score = compute_brier_score(outputs_mean, one_hot_targets)
            # Sum along batch dimension
            brier_score += torch.sum(batch_brier_score, 0).cpu().numpy().item()
            brier_l.append(torch.sum(batch_brier_score, 0).cpu().numpy().item()/targets.size(0))
            # Compute loss
            loss = criterion(outputs_mean, targets)
            test_loss += loss.item()


            ece_l.append(ece_score(outputs_mean, targets).item())

            
            # Compute predictions and numer of correct predictions
            _, predicted = outputs_mean.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc_l.append(predicted.eq(targets).sum().item()/targets.size(0))


            pred.append(predicted)
            label.append(targets)
            
            if args.show_bar and args.verbose:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    mat={'nll':nll_l, 'acc':acc_l, 'brier':brier_l, 'ece':ece_l}
    savemat('data.mat', mat) 



    accuracy = 100.*correct/total
    brier_score = brier_score/total
    neg_log_likelihood = neg_log_likelihood/total
    auc=auc/total
    pred=torch.cat(pred, dim=0)
    label=torch.cat(label, dim=0)
    auc=multiclass_roc_auc_score(pred, label)
    ece=sum(ece_l)/len(ece_l)
    print("input mean")
    print(sum(mean_l)/len(mean_l))
    return accuracy, brier_score, neg_log_likelihood, auc, ece


# Testing adf model
print('==> Loaded model statistics:')
print('        test_model_name = {}'.format(args.test_model_name))
print('        load_model_name = {}'.format(args.load_model_name))
print('        @epoch = {}'.format(start_epoch))
print('        best_acc    = {}'.format(best_acc))
print('==> Selected parameters:')
print('        use_mcdo       = {}'.format(args.use_mcdo))
print('        num_samples    = {}'.format(args.num_samples))
print('        p              = {}'.format(args.p))
print('        min_variance   = {}'.format(args.min_variance))
print('        noise_variance = {}'.format(args.noise_variance))
print('        tau = {}'.format(args.tau))
print('==> Starting evaluation...')

eval_time = time.time()

accuracy, brier_score, neg_log_likelihood, auc, ece = evaluate(
        net,
        use_adf=args.test_model_name.lower().endswith('adf'), 
        use_mcdo=args.use_mcdo)

eval_time = time.time() - eval_time

print('Accuracy                = {}'.format(accuracy))
print('Brier Score             = {}'.format(brier_score))
print('Negative log-likelihood = {}'.format(neg_log_likelihood))
print('auc roc              ={}'.format(auc))
print("ece                ={}".format(ece))
print('Time                    = {}'.format(eval_time))
    
