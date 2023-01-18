import os ,sys 
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from dataset import preprocess_adult_data
from model import  NetRaw as Net

from model import  count_parameters , freezen_gate 


from utils_van import   evaluate_eo_eop
from utils_van_prune import train_eo



from torch.utils.tensorboard import SummaryWriter

# from pruning_engine import pytorch_pruning, PruningConfigReader, prepare_pruning_list
# from pruning_utils import save_checkpoint, adjust_learning_rate, AverageMeter, accuracy, load_model_pytorch, dynamic_network_change_local, get_conv_sizes, connect_gates_with_parameters_for_flops
# from pruning_utils import group_lasso_decay


device=  torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_experiments(method='mixup', mode='dp', lam=0.5, lam2=0.5, exp=5, args=None):
    '''
    Retrain each model for 10 times and report the mean ap and dp.
    '''

    ap = []
    gap_logit = []
    gap_05 = []
#     print("Exp:", exp)
    final_list= []
    for i in [exp]:
        print('On experiment', i)

#         if args.pruning:
#             pruning_settings = dict()
#             if not (args.pruning_config is None):
#                 pruning_settings_reader = PruningConfigReader()
#                 pruning_settings_reader.read_config(args.pruning_config)
#                 pruning_settings = pruning_settings_reader.get_parameters()
#             for k, v in vars(args).items():
#                 pruning_settings[k] = v

#             if args.pruning:
#                 print (pruning_settings, "pruning_settings")

#             for k, v in pruning_settings.items():
#                 setattr(args, k, v)

                
#         logs_root = "logs_gap_test"
        logs_root = "logs_analysis_100_50"
        
        if  args.fair_method == "IGA":
            log_folder = f"{logs_root}/{args.fair_method}/{args.mode}/Exp{i}_{args.mode}_{args.name}_{args.fair_method}_lam-{args.lam}"
            pth_folder = f"save_gap_test/{args.fair_method}/{args.mode}/Exp{i}_{args.mode}_{args.name}_{args.fair_method}_lam-{args.lam}"
        elif args.fair_method == "MNIGA":
            log_folder = f"{logs_root}/{args.fair_method}/{args.mode}/Exp{i}_{args.mode}_{args.name}_{args.fair_method}-lam_{args.lam}_T-{args.T}_sl-{args.sl}"
            pth_folder = f"save_gap_test/{args.fair_method}/{args.mode}/Exp{i}_{args.mode}_{args.name}_{args.fair_method}-lam_{args.lam}_T-{args.T}_sl-{args.sl}"
        elif args.fair_method == "van" or args.fair_method == "os":
            log_folder = f"{logs_root}/{args.fair_method}/{args.mode}/Exp{i}_{args.mode}_{args.name}_{args.fair_method}"
            pth_folder = f"save_gap_test/{args.fair_method}/{args.mode}/Exp{i}_{args.mode}_{args.name}_{args.fair_method}"
        else:
            log_folder = f"{logs_root}/{args.fair_method}/{args.mode}/Exp{i}_{args.mode}_{args.name}_{args.fair_method}_lam-{args.lam}"
            pth_folder = f"save_gap_test/{args.fair_method}/{args.mode}/Exp{i}_{args.mode}_{args.name}_{args.fair_method}_lam-{args.lam}"
            
        
        os.makedirs(pth_folder, exist_ok=True)
        writer = SummaryWriter(log_folder)

        model_name = args.model_name
        x_train_size = 120
        torch.random.manual_seed(i)
        np.random.seed(i)
        model = Net(input_size=x_train_size).to(device)
        output_sizes = get_conv_sizes(args, model)
        
                # get train/test data
        X_train, X_val, X_test, y_train, y_val, y_test, A_train, A_val, A_test = preprocess_adult_data(seed=i)
        
        if args.finetune:
            model = torch.load(args.finetune_path)
            model = model.cuda()
            print("*"*50)
            print("Info of Base model:")
            if mode == 'eo' or mode == "eop":
                ap_test, gap_logit_test, gap_05_test, eop_05_test = evaluate_eo_eop(model, X_test, y_test, A_test)
            print("*"*50)

        parameters_for_update = []
        parameters_for_update_named = []
        for name, m in model.named_parameters():
            if "gate" not in name:
                parameters_for_update.append(m)
                parameters_for_update_named.append((name, m))

        group_wd_optimizer = None
        pruning_engine = None

#         if args.pruning:
#             has_attribute = lambda x: any([x in a for a in sys.argv])
#             if has_attribute('pruning-momentum'):
#                 pruning_settings['pruning_momentum'] = vars(args)['pruning_momentum']
#             if has_attribute('pruning-method'):
#                 pruning_settings['pruning_method'] = vars(args)['pruning_method']

#             pruning_parameters_list = prepare_pruning_list(pruning_settings, model, model_name=model_name,
#                                                        pruning_mask_from=args.pruning_mask_from, name=args.name)
#             print("Total pruning layers:", len(pruning_parameters_list))

#         # print ("pruning_parameters_list" , pruning_parameters_list ,"pruning_settings", pruning_settings )
#             pruning_engine = pytorch_pruning(pruning_parameters_list, pruning_settings=pruning_settings,
#                                              log_folder=log_folder, writer=writer)

#             pruning_engine.model = model_name  # args.model
#             pruning_engine.pruning_mask_from = args.pruning_mask_from
#             pruning_engine.load_mask()
#             gates_to_params = connect_gates_with_parameters_for_flops(model_name, parameters_for_update_named)
#             pruning_engine.gates_to_params = gates_to_params


#         # print("gates_to_params", gates_to_params )
#         if args.pruning:
#             group_wd_optimizer = group_lasso_decay(parameters_for_update, group_lasso_weight=args.group_wd_coeff,
                                                   named_parameters=parameters_for_update_named,
                                                   output_sizes=output_sizes)


    
        total_size_params = sum([np.prod(par.shape) for par in parameters_for_update])
        print("Total number of parameters, w/o usage of bn consts: ", total_size_params)

        # optimizer = optim.Adam(model.parameters(), lr=1e-3)
        optimizer = optim.Adam(parameters_for_update, lr=1e-3)#, momentum=args.momentum, weight_decay=weight_decay)

        criterion = nn.BCELoss()

        # run experiments
        ap_val_epoch = []
        gap_logit_val_epoch = []
        gap_05_val_epoch = []
        eop_05_val_epoch = []
        
        ap_test_epoch = []
        gap_logit_test_epoch = []
        gap_05_test_epoch = []
        eop_05_test_epoch = []
        
        domain_network0 = nn.Sequential(nn.Linear(2, 200), nn.ReLU(), nn.Linear(200, 1),nn.Sigmoid()).cuda()
        domain_network1 = nn.Sequential(nn.Linear(2, 10), nn.Linear(10, 1),nn.Sigmoid()).cuda()
        domain_optimizer0 = optim.Adam(domain_network0.parameters(), lr=1e-3)
        domain_optimizer1 = optim.Adam(domain_network1.parameters(), lr=1e-2)
        
        for j in tqdm(range(args.epochs)):
                
            print ("epoch:",j , "total:",args.epochs, "--->", "args.pruning", args.pruning)

            def pretest_call():
                if not args.pruning:
                    return 0
                '''
                do test before prunning
                '''
                if mode == 'dp':
                    ap_test, gap_logit_test, gap_05_test= evaluate_dp(model, X_test, y_test, A_test)
                if mode == 'eo' or mode == "eop":
                    # ap_valal, gap_logit_val, gap_05_v = evaluate_eo(model, X_val, y_val, A_val)
                    ap_test, gap_logit_test, gap_05_test, eop_05_test = evaluate_eo_eop(model, X_test, y_test, A_test)

                writer.add_scalar('Test/ap', ap_test, j)
                writer.add_scalar('Test/gap_logit_test', gap_logit_test, j)
                writer.add_scalar('Test/gap_05_test', gap_05_test, j)

#             if mode == 'eop':
#                 train_eop(writer, j, model, criterion, optimizer, X_train, A_train, y_train, method, lam, lam2,
#                          group_wd_optimizer=group_wd_optimizer,
#                          pruning_engine=pruning_engine, args=args, pretest_call=pretest_call)
            if mode == 'eo' or mode == "eop":
                train_eo(writer, j, model, criterion, optimizer, X_train, A_train, y_train, method, lam, lam2,
                         group_wd_optimizer=group_wd_optimizer, 
                         pruning_engine=pruning_engine, args=args, pretest_call=pretest_call,
                         domain_network0=domain_network0, domain_optimizer0=domain_optimizer0,
                         domain_network1=domain_network1, domain_optimizer1=domain_optimizer1)
                # ap_test, gap_test = evaluate_eo(model, X_test, y_test, A_test)
            if args.finetune:
                torch.save(model, pth_folder + f"/finetune-{j}.pth")
            else:
                torch.save(model, pth_folder + f"/{j}.pth")
#             for _, p in enumerate(model.named_parameters()):
#                 print(p[1])
#                 break
                
#             if mode == 'eop':
#                 print("Train:")
#                 ap_train, gap_logit_train, gap_05_train = evaluate_eo_eop(model, X_train, y_train, A_train)
#                 print("Val:")
#                 ap_val, gap_logit_val, gap_05_val = evaluate_eo_eop(model, X_val, y_val, A_val)
#                 print("Test:")
#                 ap_test, gap_logit_test, gap_05_test = evaluate_eo_eop(model, X_test, y_test, A_test)
            if mode == 'eo' or mode == "eop":
                print("Train:")
                ap_train, gap_logit_train, gap_05_train, eop_05_train = evaluate_eo_eop(model, X_train, y_train, A_train)
                print("Val:")
                ap_val, gap_logit_val, gap_05_val, eop_05_val = evaluate_eo_eop(model, X_val, y_val, A_val)
                print("Test:")
                ap_test, gap_logit_test, gap_05_test, eop_05_test = evaluate_eo_eop(model, X_test, y_test, A_test)
                
            if args.mode == "eo" and "IGA" in args.fair_method:    
                s_epoch = 2 
            else:
                s_epoch = 0
                
            if j >= s_epoch:
                ap_val_epoch.append(ap_val)
                ap_test_epoch.append(ap_test)
                gap_logit_val_epoch.append(gap_logit_val)
                gap_logit_test_epoch.append(gap_logit_test)
                gap_05_val_epoch.append(gap_05_val)
                gap_05_test_epoch.append(gap_05_test)
                eop_05_val_epoch.append(eop_05_val)
                eop_05_test_epoch.append(eop_05_test)

            writer.add_scalar('Val/ap', ap_val, j)
            writer.add_scalar('Val/gap_logit_val', gap_logit_val, j)
            writer.add_scalar('Val/gap_05_val', gap_05_val, j)
            writer.add_scalar('Val/eop_05_val', eop_05_val, j)

            writer.add_scalar('Test/ap', ap_test, j)
            writer.add_scalar('Test/gap_logit_test', gap_logit_test, j)
            writer.add_scalar('Test/gap_05_test', gap_05_test, j)
            writer.add_scalar('Test/eop_05_test', eop_05_test, j)

        idx = gap_05_val_epoch.index(min(gap_05_val_epoch))
#         gap_logit.append(gap_logit_test_epoch[idx])
#         gap_05.append(gap_05_test_epoch[idx])
#         ap.append(ap_test_epoch[idx])
        print("Epochs of Best EO:", idx+s_epoch)
        
        idx_eop = eop_05_val_epoch.index(max(eop_05_val_epoch))
#         idx_eop = 0
#         eop_05.append(eop_05_test_epoch[idx])
#         ap_besteop.append(ap_test_epoch[idx_idx])
        print("Epochs of Best EOP:", idx_eop+s_epoch)

        print('--------AVG---------')
        print(f'AP of best eo: {ap_test_epoch[idx]}')
        print(f'eo_gap_logit: {gap_logit_test_epoch[idx]}')
        print(f'eo_gap_05: {gap_05_test_epoch[idx]}')
        
        print(f'AP of best eop: {ap_test_epoch[idx_eop]}',)
        print(f'eop: {eop_05_test_epoch[idx_eop]}')
        
        with open(pth_folder + f"/{idx+s_epoch}.txt", "w") as f:
            f.write(f'\n')
            
         
        os.makedirs( f"results/Normal/eo/{args.fair_method}", exist_ok=True)
        os.makedirs( f"results/Normal/eop/{args.fair_method}", exist_ok=True)
        
#         if args.mode == "eo":
        fname = f"results/Normal/eo/{args.fair_method}/Exp{i}_{args.name}_{args.fair_method}_lam-{args.lam}_T-{args.T}.txt"
        if  args.fair_method == "MNIGA": 
            fname = f"results/Normal/eo/{args.fair_method}/Exp{i}_{args.name}_{args.fair_method}_lam-{args.lam}_T-{args.T}_sl-{args.sl}.txt"
        with open(fname, "w") as f:
            f.write(f'{ap_test_epoch[idx]}\n')
            f.write(f'{gap_logit_test_epoch[idx]}\n')
            f.write(f'{gap_05_test_epoch[idx]}\n')

#         if args.mode == "eop":
        fname = f"results/Normal/eop/{args.fair_method}/Exp{i}_{args.name}_{args.fair_method}_lam-{args.lam}_T-{args.T}.txt"
        if  args.fair_method == "MNIGA":    
            fname = f"results/Normal/eop/{args.fair_method}/Exp{i}_{args.name}_{args.fair_method}_lam-{args.lam}_T-{args.T}_sl-{args.sl}.txt"

        with open(fname, "w") as f:
            f.write(f'{ap_test_epoch[idx_eop]}\n')
            f.write(f'0.0\n')
            f.write(f'{eop_05_test_epoch[idx_eop]}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adult Experiment')
    parser.add_argument('--fair_method',"-fair_method", default='IGA', type=str, help='mixup/GapReg/erm')
    parser.add_argument('--mode', default='eo', type=str, help='dp/eo')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--exp', default=0, type=int)
    parser.add_argument('--lam',"-lam", default=0.5, type=float, help='Lambda for regularization')
    parser.add_argument('--lam2',"-lam2", default=0.5, type=float, help='Lambda for regularization')
    parser.add_argument('--name', default="adult", type=str, )
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--finetune_path', default='save_gap_test/van/eo/Exp0_eo_adult_van/0.pth', type=str,)
    parser.add_argument('--T', default=0, type=int, help='Temperature')
    parser.add_argument('--sl', default=0, type=int, help='start layer')
    parser.add_argument('--l', default=0, type=int, help='layer')
    
    parser.add_argument('--warm',"-warm", default="scratch", type=str, )
    parser.add_argument('--frequency',"-freq", default=10, type=int, )
    parser.add_argument('--prune_per_iteration',"-ppi", default=100, type=int, )
    parser.add_argument('--maximum_pruning_iterations',"-mpi", default=120, type=int, )
    parser.add_argument('--prune_neurons_max',"-pnm", default=400, type=int, )
    parser.add_argument('--pruning_threshold',"-pt", default=10, type=float, )
    

    #pruning 
    # ============================PRUNING added
    parser.add_argument('--log_interval', default=5, type=int, help='Lambda for regularization')
    parser.add_argument('--group_wd_coeff', type=float, default=0.0,
                        help='group weight decay')
    parser.add_argument('--pruning_config', default="./config/prune_config.json", type=str,
                        help='path to pruning configuration file, will overwrite all pruning parameters in arguments')
    parser.add_argument('--pruning_mask_from', default='', type=str,
                        help='path to mask file precomputed')
    parser.add_argument('--dataset', default='adult', type=str,
                        help='dataset name')
    parser.add_argument('--compute_flops', action='store_false',
                        help='if True, will run dummy inference of batch 1 before training to get conv sizes')
    parser.add_argument('--pruning', action='store_true',
                        help='enable or not pruning, def False')
    # ============================END pruning added
    parser.add_argument('--model_name', default="fc", type=str, )


    args = parser.parse_args()

    if args.fair_method == "post":
        evaluate_eo_eop = evaluate_eo_eop_post
    ## validate 
    assert  "scratch" in args.warm
    # warmup_epoch=0
    # if "frz" in args.warm :
    #     args.pruning = False
    #     args_warm= args.warm
    #     warmup_epoch  = args_warm.replace("frz","")
    #     expect_total = warmup_epoch.split("_")[-1]
    #     assert args.epochs==int(expect_total)
    #
    #     warmup_epoch = int( warmup_epoch.split("_")[0] )
    #     assert warmup_epoch <= args.epoch

    # ============================PRUNING end

    
    run_experiments(args.fair_method, args.mode, args.lam, args.lam2, args.exp, args)

