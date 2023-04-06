import json
import os
import re
import time
import argparse
import numpy as np
from encoder_block import Encoder
import torch
import torch.nn as nn
import torch.optim as optim

from model import DTI_Graph
from dataloader import load_info_data, load_pre_process
from utils import accuracy, precision, recall, specificity, mcc, auc, aupr


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

###############################################################
# Training settings


parser = argparse.ArgumentParser(description='DTI-GRAPH')

parser.add_argument('--dropout', type=float, default=0.3,  #default=0.3
                 help='Dropout rate (1 - keep probability).')

parser.add_argument('--gat_nheads', type=int, default=6, #default=8
                    help='GAT layers')


parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=223, help='Random seed.')
parser.add_argument('--epochs', type=int, default=4000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-3,
                    help='Weight decay (L2 loss on parameters).')


parser.add_argument('--if_save', type=bool, default=True,
                    help='Weight decay (L2 loss on parameters).')

###############################################################
# Model hyper setting
# Protein_NN
parser.add_argument('--protein_ninput', type=int, default=220,
                    help='protein vector size')
parser.add_argument('--pnn_nlayers', type=int, default=1,
                    help='Protein_nn layers num')
parser.add_argument('--pnn_nhid', type=str, default='[]',
                    help='pnn hidden layer dim, like [200,100] for tow hidden layers')
# Drug_NN
parser.add_argument('--drug_ninput', type=int, default=881,
                    help='Drug fingerprint dimension')
parser.add_argument('--dnn_nlayers', type=int, default=1, #default=1
                    help='dnn_nlayers num')
parser.add_argument('--dnn_nhid', type=str, default='[]',
                    help='dnn hidden layer dim, like [200,100] for tow hidden layers')
# GAT
parser.add_argument('--gat_type', type=str, default='PyG',
                    help="two different type, 'PyG Sparse GAT'(PyG) and 'Dense GAT Self'(Dense-Self)")
parser.add_argument('--gat_ninput', type=int, default=256,
                    help='GAT node feature length, is also the pnn  outpu size and dnn output size')
parser.add_argument('--gat_nhid', type=int, default=256,
                    help='hidden dim of gat')
parser.add_argument('--gat_noutput', type=int, default=256,
                    help='GAT output feature dim and the input dim of Decoder')




parser.add_argument('--gat_negative_slope', type=float, default=0.2,
                    help='GAT LeakyReLU angle of the negative slope.')
# Decoder
parser.add_argument('--DTI_nn_nlayers', type=int, default=3,
                    help='Protein_nn layers num')
parser.add_argument('--DTI_nn_nhid', type=str, default='[256,256,256]',   #default='[256,256,256]'
                    help='DTI_nn hidden layer dim, like [200,100] for tow hidden layers')
###############################################################
# data

# parser.add_argument('--model_dir', type=str, default='./save_nuc_receptor_model_com3',#default='./save_enzyme_model_com3'
#                     help='model save path')
parser.add_argument('--crossvalidation', type=int, default=1,
                    help='whether use crossvalidation or not')

parser.add_argument('--dataset', type=str, default='cross_reverse', 
                    help='dataset name')
parser.add_argument('--common_neighbor', type=int, default=3, #default=1
                    help='common neighbor of adj transform, this will determine what preprocessed matrix you use')
parser.add_argument('--sample_num', type=int, default=300,
                    help='different epoch use different sample, the sample num')
parser.add_argument('--data_path', type=str, default='./data',
                    help='dataset root path')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)



args.model_dir='./saved_model/save_{}_model_com3'.format(args.dataset)
    
# nn layers
p1 = re.compile(r'[[](.*?)[]]', re.S)
if args.dnn_nhid == '[]':
    args.dnn_nhid = []
else:
    args.dnn_nhid = [int(i) for i in re.findall(p1, args.dnn_nhid)[0].replace(' ', '').split(',')]
if args.pnn_nhid == '[]':
    args.pnn_nhid = []
else:
    args.pnn_nhid = [int(i) for i in re.findall(p1, args.pnn_nhid)[0].replace(' ', '').split(',')]
args.DTI_nn_nhid = [int(i) for i in re.findall(p1, args.DTI_nn_nhid)[0].replace(' ', '').split(',')]
# load data
data_Path = os.path.join(args.data_path, 'data_'+args.dataset+'.npz')
preprocess_path = os.path.join(args.data_path, 'preprocess', args.dataset+'_com_'+str(args.common_neighbor))
# save dir
if not os.path.exists(args.model_dir):
    os.mkdir(args.model_dir)
protein_tensor, drug_tensor, node_num, protein_num = load_info_data(data_Path)

# Hyper Setting
pnn_hyper = [args.protein_ninput, args.pnn_nhid, args.gat_ninput, args.pnn_nlayers]
dnn_hyper = [args.drug_ninput, args.dnn_nhid, args.gat_ninput, args.dnn_nlayers]
GAT_hyper = [args.gat_ninput, args.gat_nhid, args.gat_noutput, args.gat_negative_slope, args.gat_nheads]
Deco_hyper = [args.gat_noutput, args.DTI_nn_nhid, args.DTI_nn_nlayers]

def train(epoch, link_dti_id_train, edge_index, edge_weight, train_dti_inter_mat):
    # if use PyG's sparse gcn, you will need the edge_weight
    t = time.time()
    model.train()
    optimizer.zero_grad()
    row_dti_id = link_dti_id_train.permute(1, 0)[0]
    col_dti_id = link_dti_id_train.permute(1, 0)[1]
    protein_index = row_dti_id
    drug_index = col_dti_id + train_dti_inter_mat.shape[0]
    output = model(protein_tensor, drug_tensor, edge_index, protein_index, drug_index)
    Loss = nn.BCELoss()
   # ---------------------------！！！------------------------------------------
    loss_train = Loss(output, train_dti_inter_mat[row_dti_id, col_dti_id])
    acc_dti_train = accuracy(output, train_dti_inter_mat[row_dti_id, col_dti_id])
    loss_train.backward()
    optimizer.step()
    print('Epoch {:04d} Train '.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_dti_train: {:.4f}'.format(acc_dti_train),
          'time: {:.4f}s'.format(time.time() - t))

def test(link_dti_id_test, edge_index, edge_weight, test_dti_inter_mat):
    # if use PyG's sparse gcn, you will need the edge_weight
    model.eval()
    row_dti_id = link_dti_id_test.permute(1, 0)[0]
    col_dti_id = link_dti_id_test.permute(1, 0)[1]
    protein_index = row_dti_id
    drug_index = col_dti_id + test_dti_inter_mat.shape[0]
    output = model(protein_tensor, drug_tensor, edge_index, protein_index, drug_index)
    Loss = nn.BCELoss()
    predicts = output
    targets = test_dti_inter_mat[row_dti_id, col_dti_id]
    loss_test = Loss(predicts, targets)
    acc_dti_test = accuracy(output, test_dti_inter_mat[row_dti_id, col_dti_id])
    return acc_dti_test, loss_test, predicts, targets

# Train model
t_total = time.time()
acc_score = np.zeros(5)
precision_score = np.zeros(5)
recall_score = np.zeros(5)
specificity_score = np.zeros(5)
mcc_score = np.zeros(5)
auc_score = np.zeros(5)
aupr_score = np.zeros(5)
# fold_num = 5 if args.crossvalidation else 1
fold_num = 5 if args.crossvalidation else 1

for train_times in range(fold_num):
    model = DTI_Graph(GAT_hyper=GAT_hyper, PNN_hyper=pnn_hyper, DNN_hyper=dnn_hyper, DECO_hyper=Deco_hyper,
                      Protein_num=protein_tensor.shape[0], Drug_num=drug_tensor.shape[0], dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_test = 0
    preprocess_oripath = os.path.join(preprocess_path, '0_'+str(train_times)+'.json')  # ori fold we use test
    adj, ori_dti_inter_mat, ori_train_interact_pos, ori_val_interact_pos = load_pre_process(preprocess_oripath)
    edge_index = torch.nonzero(adj > 0).permute(1, 0)
    edge_weight = adj[np.array(edge_index)]
    if args.cuda:
#         model = model.cuda()
        protein_tensor = protein_tensor.to('cuda:0')
        drug_tensor = drug_tensor.to('cuda:0')
        edge_index = edge_index.to('cuda:0')
        # edge_weight = edge_weight.cuda() # if you want to use gcn and so on
        ori_dti_inter_mat = ori_dti_inter_mat.to('cuda:0')
        ori_train_interact_pos = ori_train_interact_pos.to('cuda:0')
        ori_val_interact_pos = ori_val_interact_pos.to('cuda:0')
    save_time_fold = os.path.join(args.model_dir, str(train_times))
    if not os.path.exists(save_time_fold):
        os.mkdir(save_time_fold)
    
    for epoch in range(args.epochs):
        data_id = epoch % args.sample_num
        print("use sample", data_id)
        preprocess_generate_path = os.path.join(preprocess_path, str(data_id)+'_'+str(train_times)+'.json')
        adj, dti_inter_mat, train_interact_pos, val_interact_pos = load_pre_process(preprocess_generate_path)
        if args.cuda:
            dti_inter_mat = dti_inter_mat.cuda()
            train_interact_pos = train_interact_pos.cuda()
        print('Epoch: {:04d}'.format(epoch + 1), 'Train_times:', train_times)

        train(epoch, train_interact_pos, edge_index, edge_weight, dti_inter_mat)

        test_score, test_loss, predicts, targets = test(ori_val_interact_pos, edge_index, edge_weight, ori_dti_inter_mat)
        if test_score > best_test:
            best_test = test_score
            acc_score[train_times] = round(best_test, 4)
            
            model_dict=model.state_dict()
            
            predict_target = torch.cat((predicts, targets), dim=0).detach().cpu().numpy()


            precision_score[train_times] = round(precision(predicts, targets), 4)
            recall_score[train_times] = round(recall(predicts, targets), 4)
            specificity_score[train_times] = round(specificity(predicts, targets), 4)
            mcc_score[train_times] = round(mcc(predicts, targets), 4)
            auc_score = round(auc(predicts, targets), 4)
            aupr_score = round(aupr(predicts, targets), 4)
            

        print('Epoch: {:04d}'.format(epoch + 1), 'Train_times:', train_times)
        print("*****************test_score {:.4f} best_socre {:.4f}****************".format(test_score, best_test))
        print("All Test Score:", acc_score)
    save_model_path = os.path.join(args.model_dir,str(train_times),args.dataset+'_com_3_times_'+str(train_times)+'_'+str(round(best_test, 4))+'.pth.tar')

    #保存模型和数据
#   torch.save(model_dict, save_model_path)
#    np.savetxt(save_predict_target_path, predict_target)

with open(os.path.join(args.model_dir,str(train_times),'saved_train_result.txt'),'w') as f: 
    f.write( 
            'dropout:{}'.format(args.dropout) + '\n' +
            'n_heads:{}'.format(args.gat_nheads) + '\n' + 
            "Total time elapsed: {:.4f}s".format(time.time() - t_total)+'\n' +
            "acc Score:{}".format(acc_score) + '\n' +
            "precision Score:{}".format(precision_score) + '\n' +
            "recall score{}".format(recall_score) + '\n' +
            "specificity score{}".format(specificity_score) + '\n' +
            "mcc score{}".format(mcc_score) + '\n' +
            'auc socre{}'.format(auc_score) + '\n' +
            'aupr score{}'.format(aupr_score) + '\n' +
            'Best Ave Test: {:.4f}'.format(np.mean(acc_score)) + '\n'+'\n'
            
           )

print(args.dataset, " Optimization Finished!")
print('dropout:{}'.format(args.dropout))
print('n_heads:{}'.format(args.gat_nheads))
print("data:{}".format(args.dataset))
print("epochs:{}".format(args.epochs))
print("lr:{}".format(args.lr))
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print("acc Score:", acc_score)
print("precision Score:", precision_score)
print("recall score", recall_score)
print("specificity score", specificity_score)
print("mcc score", mcc_score)
print("auc socre", auc_score)
print("aupr score", aupr_score)
print("Best Ave Test: {:.4f}".format(np.mean(acc_score)))
