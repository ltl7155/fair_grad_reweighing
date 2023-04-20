import torch
import numpy as np
from numpy.random import beta
from sklearn.metrics import average_precision_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, confusion_matrix

device=  torch.device("cuda" if torch.cuda.is_available() else "cpu")

from utils_van import sample_batch_sen_idx, sample_batch

from utils_van import sample_batch_sen_idx_y 


def train_eo(writer, epochs, model, criterion, optimizer, X_train, A_train, y_train, 
             method, lam, lam2, batch_size=500, niter=100,
             pruning_engine=None, group_wd_optimizer=None, args=None, pretest_call=lambda x:x,
            domain_network0=None, domain_optimizer0=None, domain_network1=None, domain_optimizer1=None):
    model.train()
    
    print ("args.pruning,",args.pruning)
    for it in range(niter):

        optimizer.zero_grad()

        # Gender Split
        batch_x_0, batch_y_0 = sample_batch_sen_idx_y(X_train, A_train, y_train, batch_size, 0)
        batch_x_1, batch_y_1 = sample_batch_sen_idx_y(X_train, A_train, y_train, batch_size, 1)

        inputs = [batch_x_0, batch_x_1]
        target = [batch_y_0, batch_y_1]

        # separate class
        batch_x_0_ = [batch_x_0[:batch_size], batch_x_0[batch_size:]]
        batch_x_1_ = [batch_x_1[:batch_size], batch_x_1[batch_size:]]

        batch_x_ = [batch_x_0[:batch_size], batch_x_0[batch_size:], batch_x_1[:batch_size], batch_x_1[batch_size:]]
        batch_y_ = [batch_y_0[:batch_size], batch_y_0[batch_size:], batch_y_1[:batch_size], batch_y_1[batch_size:]]

        # ERM loss
        batch_x = torch.cat((batch_x_0, batch_x_1), 0)
        batch_y = torch.cat((batch_y_0, batch_y_1), 0)
        batch_g = torch.cat((torch.zeros_like(batch_y_0), torch.ones_like(batch_y_1)), 0)
        
#         batch_x_0_dp, batch_y_0_dp = sample_batch_sen_idx(X_train, A_train, y_train, batch_size, 0)
#         batch_x_1_dp, batch_y_1_dp = sample_batch_sen_idx(X_train, A_train, y_train, batch_size, 1)
#         batch_x_dp = [batch_x_0_dp, batch_x_1_dp]
#         batch_y_dp = [batch_y_0_dp, batch_y_1_dp]
        batch_x_0_dp, batch_y_0_dp = batch_x_0, batch_y_0
        batch_x_1_dp, batch_y_1_dp = batch_x_1, batch_y_1
        batch_x_dp = [batch_x_0_dp, batch_x_1_dp]
        batch_y_dp = [batch_y_0_dp, batch_y_1_dp]


        output = model(batch_x)
        loss_sup = criterion(output, batch_y)

        
                
        if method == "MNIGA":

            losses = []
            grads = []
            output_mean = []
            for g in range(4):
                output = model(batch_x_[g])
                output_mean.append(output.mean().data)
                loss_c = criterion(output, batch_y_[g])
                losses.append(loss_c)
                env_grad = torch.autograd.grad(loss_c, model.parameters(), create_graph=True)
                grads.append(env_grad)

            # compute trace penalty
            loss_reg = 0
            g_abs_0 = 0
            g_abs_1 = 0

            weight00 = (1-output_mean[0]).pow(args.T).data
            weight01 = output_mean[1].pow(args.T).data
            weight10 = (1-output_mean[2]).pow(args.T).data
            weight11 = output_mean[3].pow(args.T).data
#             weight00 = (1-output_mean[0]).pow(3)
#             weight01 = output_mean[1].pow(3)
#             weight10 = (1-output_mean[2]).pow(3)
#             weight11 = output_mean[3].pow(3)
#           The loss item is identical to the item design in the paper and can reduce the calculation.
            for k, (g0, g1) in enumerate(zip(grads[0], grads[2])):
                if k >= args.sl:
                    # print("v1:", (1-output_mean[2]))
                    # print("v2:",(1 - output_mean[1]))
                    loss_reg += ((g0.pow(2))*weight00 - (g1.pow(2))*weight10).abs().sum()
                    # loss_reg += ((g0.pow(2))*5 - (g1.pow(2))).abs().sum()

            for k, (g0, g1) in enumerate(zip(grads[1], grads[3])):
                if k >= args.sl:
                    loss_reg += ((g0.pow(2))*weight01 - (g1.pow(2))*weight11).abs().sum()

            loss = loss_sup + lam * loss_reg

#             mean_grad = torch.autograd.grad(loss_sup, model.parameters(), create_graph=True)
            writer.add_scalar('Loss/loss_sup', loss_sup, niter * epochs + it)
            writer.add_scalar('Loss/loss_reg', loss_reg, niter * epochs + it)
        
#             writer.add_scalar('Dis/dis1', penalty_value[0], niter * epochs + it)
#             writer.add_scalar('Dis/dis2', penalty_value[1], niter * epochs + it)
#             writer.add_scalar('Dis/dis3', penalty_value[2], niter * epochs + it)
#             writer.add_scalar('Dis/dis4', penalty_value[3], niter * epochs + it)

            if (it % 100) == 0:
                print(f"loss_sup:{loss_sup}, loss_reg: {loss_reg}")
            
        elif method == "van":
            batch_x, batch_y, batch_a, idx_00, idx_01, idx_10, idx_11  = sample_batch(X_train, A_train, y_train, batch_size)

            output = model(batch_x)
            loss_sup = criterion(output, batch_y)

            batch_x_ = [batch_x[idx_00], batch_x[idx_01], batch_x[idx_10], batch_x[idx_11]]
            batch_y_ = [batch_y[idx_00], batch_y[idx_01], batch_y[idx_10], batch_y[idx_11]]
            
            output_mean = []
            losses = []
            grads = []
            
            for g in range(4):
                output = model(batch_x_[g])
                output_mean.append(output.mean().data)
                loss_c = criterion(output, batch_y_[g])
                losses.append(loss_c)
                env_grad = torch.autograd.grad(loss_c, model.parameters(), create_graph=True)
                grads.append(env_grad)
            
            counter = 0
            with torch.no_grad():
                penalty_value = [0, 0, 0, 0]
                for g0, g1 in zip(grads[0], grads[2]):
                    counter += 1
                    penalty_value[0] += max((g1.pow(2).sum() / g0.pow(2).sum()), (g0.pow(2).sum() / g1.pow(2).sum()) )
                for g0, g1 in zip(grads[1], grads[3]):
                    penalty_value[1] += max((g1.pow(2).sum() / g0.pow(2).sum()), (g0.pow(2).sum() / g1.pow(2).sum()) )
#                 for g0, g1 in zip(grads[0], grads[2]):
#                     penalty_value[2] += (g1.pow(2) / g0.pow(2)).sum()
#                 for g0, g1 in zip(grads[1], grads[3]):
#                     penalty_value[3] += (g1.pow(2) / g0.pow(2)).sum()
                    
#             print(penalty_value[0], penalty_value[1], penalty_value[2], penalty_value[3])
            loss = loss_sup
            
            
            writer.add_scalar('Dis/dis1', penalty_value[0]/counter, niter * epochs + it)
            writer.add_scalar('Dis/dis2', penalty_value[1]/counter, niter * epochs + it)
#             writer.add_scalar('Dis/dis3', penalty_value[2], niter * epochs + it)
#             writer.add_scalar('Dis/dis4', penalty_value[3], niter * epochs + it)
            writer.add_scalar('Dis/sum12', (penalty_value[0]+penalty_value[1])/counter, niter * epochs + it)
#             writer.add_scalar('Dis/sum34', penalty_value[2]+penalty_value[3], niter * epochs + it)

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.0001)
        # step_after will calculate flops and number of parameters left
        # needs to be launched before the main optimizer,
        # otherwise weight decay will make numbers not correct

        optimizer.step()


