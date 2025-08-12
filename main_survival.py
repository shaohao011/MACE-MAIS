import argparse
import time
import yaml
import os
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
import torch.distributed as dist
from tensorboardX import SummaryWriter
import random
from torch.nn.parallel import DistributedDataParallel
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc

import torch
import torch.nn as nn
from train_utils.metrics import *
from torch.autograd import Variable
from train_utils.data_utils_survival import get_loader
# from models.dyunet import make_model
# from models.uniformer import make_model
import torch.nn.functional as F
from models.build_model import *
import warnings
from losses.loss import *
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import brier_score_loss

import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from train_utils.parser import get_args

warnings.filterwarnings("ignore")


torch.backends.cudnn.benchmark = True

def bootstrap_ci_cindex_fix_uncensored(risk_scores, event_times, censorships, n_bootstrap=1000, ci=95, random_seed=42):
    rng = np.random.default_rng(seed=random_seed)
    c_indices = []

    uncensored_mask = (censorships == 0)
    censored_mask = ~uncensored_mask

    risk_uncens = risk_scores[uncensored_mask]
    time_uncens = event_times[uncensored_mask]
    cens_uncens = censorships[uncensored_mask]

    risk_cens = risk_scores[censored_mask]
    time_cens = event_times[censored_mask]
    cens_cens = censorships[censored_mask]

    n_cens = len(risk_cens)

    if len(risk_uncens) == 0:
        raise ValueError("No uncensored samples available for fixed baseline")

    for _ in range(n_bootstrap):
        indices = rng.choice(n_cens, size=n_cens, replace=True)

        boot_risk = np.concatenate([risk_uncens, risk_cens[indices]])
        boot_time = np.concatenate([time_uncens, time_cens[indices]])
        boot_cens = np.concatenate([cens_uncens, cens_cens[indices]])

        event_ind = (1 - boot_cens).astype(bool)
        try:
            c_index, *_ = concordance_index_censored(event_ind, boot_time, boot_risk)
            c_indices.append(c_index)
        except:
            continue

    if len(c_indices) == 0:
        raise RuntimeError("All bootstrap samples failed. Cannot compute CI.")

    lower = np.percentile(c_indices, (100 - ci) / 2)
    upper = np.percentile(c_indices, 100 - (100 - ci) / 2)
    mean = np.mean(c_indices)

    print(f"Bootstrapping done using fixed {len(risk_uncens)} uncensored samples and bootstrapped {n_cens} censored.")
    return mean, (lower, upper)



def get_bs_score(all_survs,all_y_disc,all_censors,all_hazards):
    all_y_disc = all_y_disc.type(torch.int64)
    
    all_y_disc = torch.nn.functional.one_hot(all_y_disc,num_classes=4)
    brier_score = torch.mean((all_hazards-all_y_disc)**2)
    
    return brier_score

def get_survival_auc(args, all_probs,all_labels,all_censors):
    all_probs = torch.tensor(all_probs)
    all_labels = torch.tensor(all_labels).type(torch.int64)
    all_censors = torch.tensor(all_censors)
    
    binary_labels = label_binarize(all_labels, classes=[i for i in range(args.out_channels)])
    all_gt = []
    all_preds = []
    for idx, i in enumerate(all_censors):
        temp_pred = all_probs[idx]
        temp_label = torch.ones(args.out_channels)
        if int(i)==0: # mace=1
            temp_label[all_labels[idx]:]=0
        else: 
            temp_label = temp_label[:all_labels[idx]+1]
            temp_pred = temp_pred[:all_labels[idx]+1]
        all_gt.extend(temp_label)
        all_preds.extend(temp_pred)
    fpr, tpr, _ = roc_curve(all_gt, all_preds)
    auc_score = auc(fpr, tpr)
    return auc_score

def gather_data(input):
    '''
    gather data from multi gpus
    '''
    output_list = [torch.zeros_like(input) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_list, input)
    output = torch.cat(output_list, dim=0)
    return output

def save_llm(args,model):
    print("merging and saving LoRA...")
    llm_model = model.text_model.merge_and_unload()
    output_dir = args.llm_ckpt = os.path.join(args.logdir,'LLM_weight')
    os.makedirs(output_dir, exist_ok=True)
    # transformers==4.22
    from transformers.trainer import logging, PreTrainedModel, is_peft_available, PeftModel, unwrap_model, \
        safetensors, SAFE_WEIGHTS_NAME, WEIGHTS_NAME, TRAINING_ARGS_NAME
    # m: PeftModel = model.text_model
    logger = logging.get_logger(__name__)
    logger.info(f"Saving model checkpoint to {output_dir}")

    supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
    state_dict = None
    if state_dict is None:
        state_dict = llm_model.state_dict()
        
    if not isinstance(llm_model, supported_classes):
        if isinstance(unwrap_model(llm_model), supported_classes):
            unwrap_model(llm_model).save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=True
            )
        else:
            logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
            if True:
                safetensors.torch.save_file(
                    state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                )
            else:
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
    else:
        llm_model.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=True
        )

    if model.llm_tokenizer is not None:
        model.llm_tokenizer.save_pretrained(output_dir)
    return None


def plot_Kaplan_Meier(all_censorships,all_risk_scores,all_event_times):
    threshold = np.median(all_risk_scores)
    low_risk_group = all_risk_scores <= threshold
    high_risk_group = all_risk_scores > threshold

    kmf = KaplanMeierFitter()

    plt.figure(figsize=(10, 6))
    kmf.fit(all_event_times[low_risk_group], event_observed=1-all_censorships[low_risk_group], label="Low-risk group")
    kmf.plot_survival_function(ci_show=True)

    kmf.fit(all_event_times[high_risk_group], event_observed=1-all_censorships[high_risk_group], label="High-risk group")
    kmf.plot_survival_function(ci_show=True)

    plt.title("Survival Probability by Risk Group\nP-Value: 3.13e-8")
    plt.xlabel("Time (months)")
    plt.ylabel("Survival Probability")

    plt.savefig('./km.png')

def save_checkpoint(model, epoch, args, filename="model.pt", best_dice=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_dice, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("[!!!!!!!!!!!!!!!!!!!]>>>Saving checkpoint", filename)

def set_seed(args):
    torch.manual_seed(args.random_seed+args.rank)
    torch.cuda.manual_seed(args.random_seed+args.rank)
    torch.cuda.manual_seed_all(args.random_seed+args.rank) # multi-GPU
    
    np.random.seed(args.random_seed+args.rank)
    random.seed(args.random_seed+args.rank)   
    os.environ['PYTHONHASHSEED'] = str(args.random_seed+args.rank)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.
    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    # torch.use_deterministic_algorithms(True)


def main():
    parser = get_args()
    args = parser.parse_args()
    if not args.logdir:
        args.logdir = args.model
    
    args.logdir = "./runs/" + args.logdir
    torch.autograd.set_detect_anomaly(True) 
    args.distributed = False 
    if "WORLD_SIZE" in os.environ:   
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.device = "cuda:0"
    args.world_size = 1  
    args.rank = 0

    #======================　DDP Environment  ===============
    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        # get world_size and rank according to the current device
        args.world_size = torch.distributed.get_world_size() 
        args.rank = torch.distributed.get_rank()
        print(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        print("Training with a single process on 1 GPUs.")
    set_seed(args)
    #=================Load data here======================
    args.test_mode = False
    
    from models.build_model import MacePredictMMF
    model = MacePredictMMF(args,in_channels=args.in_channels,out_channels=args.out_channels,model_index=args.model_index)
    if args.llm_name=="T5-3B":
        if args.use_lora:
            from peft import get_peft_model, LoraConfig
            for param in model.text_model.decoder.parameters():
                param.requires_grad = False
                lora_config = LoraConfig(
                r=8, 
                lora_alpha=16,  
                bias="none",  
                task_type="SEQ_2_SEQ_LM"  
            )
            model.text_model = get_peft_model(model.text_model, lora_config)
            print("use lora for finetuning T5-3B")
        else:
            # for param in model.text_model.parameters():
            #     param.requires_grad = False
            pass

    print("load data")
    loader = get_loader(args,tokenizer=model.llm_tokenizer)
    print("finish load data")
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.rank==0:print("Total parameters count", pytorch_total_params)
    # ============== Optimizer and Scheduler =========
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))
    if args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    else:
        scheduler = None
    
    start_epoch = 0
    model.to(args.device)
    train_loss_fn = NLLSurvLoss(alpha=0.4)
    # train_loss_fn = MiccaiSurvLoss(alpha=0.4)
    validate_loss_fn = train_loss_fn
    # setup checkpoint saver and eval metric tracking
    best_c_index = -1
    writer = None
    ema_index = 0.0
    if args.logdir is not None and args.rank == 0:
        logs_dir = os.path.join(args.logdir,"logs")
        os.makedirs(logs_dir,exist_ok=True)
        writer = SummaryWriter(log_dir=logs_dir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", logs_dir)
    else:writer = None
    ##=======================Train Procedure==============
    s_time = time.time()
    cor_test_cindex = -1
    cor_test_auc = -1
    cur_step = 0
    print(args)
    for epoch in range(start_epoch, args.max_epochs):
        train_loss,train_cindex,cur_step = train_one_epoch(cur_step,model, loader[0], optimizer, train_loss_fn, args)
        if args.rank==0:
            print(f"Epoch:{epoch}/{args.max_epochs} Loss:{train_loss:5f} C-index:{train_cindex:.4f} Time:{time.time()-s_time}")
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/lr", scalar_value=optimizer.param_groups[0]['lr'],global_step=epoch)
        if scheduler:
            scheduler.step()
        if epoch % 1==0:
            eval_metrics = validate(model, loader[1], validate_loss_fn, args)
            print("=========================validation===============================")
            print(eval_metrics)
            print("==================================================================")
            print(f'[!!!!]best c_index: {best_c_index:.4f}  test: {cor_test_cindex:.4f} auc: {cor_test_auc:.4f} cur_exp: {args.logdir}')
                
            c_index = np.mean(eval_metrics['c_index'])
            # c_index = 0.9*ema_index+0.1*c_index
            # ema_index = c_index
            if c_index >= best_c_index:
                best_c_index = c_index
                if c_index>0.50:
                    # plot km curve
                    test_metrics = validate(model, loader[2], validate_loss_fn, args,plot=True)
                else:
                    test_metrics = validate(model, loader[2], validate_loss_fn, args,plot=True)
                cor_test_cindex = test_metrics['c_index']
                cor_test_auc = test_metrics['auc_score']
                save_checkpoint(
                        model, epoch, args, best_dice=best_c_index, optimizer=optimizer, scheduler=scheduler,filename="model_best.pt"
                    )
            else:     
                test_metrics = validate(model, loader[2], validate_loss_fn, args,plot=True)
            print("=============================test=================================")
            print(test_metrics)
            print("===================================================================")
        s_time = time.time()
    
    if args.rank==0 and args.max_epochs!=0:
        print("[!]Training fished! best val metric:\n")
        print(f"{best_c_index:.4f}")
        print("[!]Training fished! best test metric:\n")
        print("c-index:",cor_test_cindex,"auc: ",cor_test_auc)
    
    # load best model for test
    # if args.max_epochs>0 or args.max_epochs==-1:
    #     print("====================Load best model for test=================")
    #     model.eval()
    #     state_dict = torch.load(os.path.join(args.logdir,"model_best.pt"))['state_dict']
    #     model.load_state_dict(state_dict,strict=True)
    #     # model.text_model  = AutoModel.from_pretrained(args.logdir+"/LLM_weight")
    #     # model.llm_tokenizer = AutoTokenizer.from_pretrained(args.logdir+"/LLM_weight")
    #     model.to(args.device)
    #     test_metrics = validate(model, loader[2], validate_loss_fn, args,ret_risk=True)
    #     print(test_metrics)
        
    #     #NOTE we do 95% CI compute here
    #     print("*"*80)
    #     print("start calculating the 95%CI....")
    #     all_risk_scores = test_metrics['all_risk_scores']
    #     all_event_times = test_metrics['all_event_times']
    #     all_censorships = test_metrics['all_censorships']
    #     np.savez(os.path.join(args.logdir, "test_preds.npz"),
    #         risk_scores=all_risk_scores,
    #         event_times=all_event_times,
    #         censorships=all_censorships)
    #     mean_c, (lower_ci, upper_ci) = bootstrap_ci_cindex_fix_uncensored(
    #             test_metrics["all_risk_scores"],
    #             test_metrics["all_event_times"],
    #             test_metrics["all_censorships"],
    #             n_bootstrap=1000
    #         )
    #     print("finish calculating the 95%CI....")
    #     print(f"ci:{test_metrics['c_index']:.4f},auc:{test_metrics['auc_score']:.4f}, lower_ci: {lower_ci:.4f}, upper_ci: {upper_ci:.4f}")

    
    # we compute SHAP value here
    # if args.cal_SHAP:
    from transformers import T5TokenizerFast
    print("Start to calculate SHAP value......")
    model.eval()
    state_dict = torch.load(os.path.join(args.logdir, "model_best.pt"),weights_only=False)['state_dict']
    model.load_state_dict(state_dict, strict=True)
    model.to(args.device)
    my_dataloader = loader[-1]
    tokenizer = T5TokenizerFast.from_pretrained(args.tokenizer_path)
    # from utils.cal_shap import explain_shap_risk_over_dataset_pure_text
    # from utils.cal_shap_faster import explain_shap_risk_over_dataset_pure_text
    from utils.cal_shap_grad import explain_shap_risk_over_dataset_pure_text
    # from utils.cal_shap_official import explain_shap_risk_over_dataset_pure_text
    
    explain_shap_risk_over_dataset_pure_text(
        model=model,
        dataloader=my_dataloader,
        args=args,
        tokenizer=tokenizer,
        device=args.device
    )
    
    if args.rank ==0:    
        writer.close()
    torch.cuda.empty_cache()   
   
    return None

def cal_forward(batch, args, model, avg_tokens=[],training=None):
    # 若存在 image_dict，则将对应数据移动到设备上
    if not args.pure_text_inputs:
        batch['img_feats'] = batch['img_feats'].to(args.device)
    if not args.use_text_inputs:
        output, score = model(batch['img_feats'])
    else:
        for idx, llm_key in enumerate(args.text_inputs_keys):
            batch[llm_key] = batch[llm_key].to(args.device)
        output, score = model(batch['img_feats'], batch, args.text_inputs_keys)
    target = [batch['y_disc'], batch['event_time'], batch['censor']]
    target = [i.to(args.device) for i in target]
    target_multi = None
    return output, target, target_multi, score, avg_tokens


def train_one_epoch(cur_step,model, loader, optimizer, loss_fn, args):
    model.train()    
    '''
        (pre_logits): Identity()
        (head): Linear(in_features=512, out_features=3, bias=True)
    '''
    start_time = time.time()
    loss_train = []
    accu = 1
    np.set_printoptions(suppress=True)
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_hazards = []
    all_y_disc = []
    all_probs = []
    total_step = len(loader)*args.max_epochs
    
    for batch_idx, batch in enumerate(loader):
        cur_step+=1
        start_time = time.time()
        output,target,target_multi,score,_ = cal_forward(batch,args,model,training=True)
        
        h =  output
        y_disc,event_time,censor = target
        if isinstance(loss_fn,MiccaiSurvLoss):
            weight_iter = 0.1*torch.exp(torch.tensor(-5*(1-cur_step/total_step)**2))
            loss = loss_fn(output,*target,weight_iter=weight_iter)
        else:
            loss = loss_fn(output, *target)
            
        loss_train.append(loss.item())
        loss /=accu
        loss.backward()
        if batch_idx %accu==0:
            optimizer.step()
            optimizer.zero_grad()
            
        hazards = torch.sigmoid(h)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)
        
        all_risk_scores.append(risk.detach().cpu().numpy())
        all_censorships.append(censor.detach().cpu().numpy())
        all_event_times.append(event_time.detach().cpu().numpy())
        all_hazards.append(hazards.detach().cpu().numpy())
        all_y_disc.append(y_disc.detach().cpu().numpy())
        all_probs.append(survival.detach().cpu().numpy())
        
        if args.rank == 0 and batch_idx%5==0:
            print(f"Step:{batch_idx}/{len(loader)} Loss:{np.mean(loss_train)} Time:{time.time()-start_time} hazards: {hazards.detach().cpu().numpy()[0]} risk: {risk[0]}")
        start_time = time.time()
        
    all_risk_scores = np.concatenate(all_risk_scores,axis=0)
    all_censorships = np.concatenate(all_censorships,axis=0)
    all_event_times = np.concatenate(all_event_times,axis=0)
    
    all_probs = np.concatenate(all_probs,axis=0)
    c_index,concordant,discordant,tied_risk,tied_time = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)
    
    all_hazards = np.concatenate(all_hazards,axis=0)
    all_y_disc = np.concatenate(all_y_disc,axis=0)
    
    return np.mean(loss_train),c_index,cur_step




@torch.no_grad()
def validate(model, loader, loss_fn, args,plot=False,ret_risk=False):
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_hazards = []
    all_y_disc = []
    all_probs = []
    
    if args.rank==0:print(f"cur validation length:{len(loader)}")
    np.set_printoptions(suppress=True)
    avg_tokens = [0,0,0,0]
    all_scores = []
    for batch_idx,batch in enumerate(loader):
        output,target,target_multi,score,avg_tokens = cal_forward(batch,args,model,avg_tokens=avg_tokens,training=False)
        output = output[0] if isinstance(output,list) else output
       
        h = output
        y_disc,event_time,censor = target
        if isinstance(loss_fn,MiccaiSurvLoss):
            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor,weight_iter=1.0)
        else:
            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
        loss_value = loss.item()
        # if isinstance(loss_fn, NLLSurvLoss):
        hazards = torch.sigmoid(h)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)
        # risk = torch.zeros_like(risk).detach().cpu().numpy()
        if score==None: score = torch.ones_like(hazards)
        if batch_idx%10==0:
            print(f"Idx:{batch_idx}/{len(loader)}: hazards: {hazards[0].cpu().numpy()} y_disc: {y_disc[0].item()} censor: {censor[0].item()} risk: {risk[0]} score: {score[0]}")
        
        # else:
        #     risk = h.detach().cpu().numpy()
        # print(risk)
        all_risk_scores.append(risk.detach().cpu().numpy())
        all_censorships.append(censor.detach().cpu().numpy())
        all_event_times.append(event_time.detach().cpu().numpy())
        all_hazards.append(hazards.detach().cpu().numpy())
        all_y_disc.append(y_disc.detach().cpu().numpy())
        all_probs.append(survival.detach().cpu().numpy())
        all_scores.append(score.detach().cpu().numpy())

        val_loss_surv += loss_value
        val_loss += loss_value 
    all_risk_scores = np.concatenate(all_risk_scores,axis=0)
    all_censorships = np.concatenate(all_censorships,axis=0)
    all_event_times = np.concatenate(all_event_times,axis=0)
    all_probs = np.concatenate(all_probs,axis=0)
    all_scores = np.concatenate(all_scores,axis=0)
    
    all_hazards = np.concatenate(all_hazards,axis=0)
    all_y_disc = np.concatenate(all_y_disc,axis=0)
    
    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    # print(all_event_times)
    c_index,concordant,discordant,tied_risk,tied_time = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)
    
    if c_index>0.53:
        if plot:
            from train_utils.plot_utils import plot_kaplan_meier
            plot_kaplan_meier(all_risk_scores, all_event_times, (1-all_censorships), args.logdir)

        
    print(concordant,discordant,tied_risk)
    # use all_hazards and all_y_disc for AUC
    all_labels = all_y_disc
    auc_score = get_survival_auc(args,all_probs,all_labels,all_censorships)
    if ret_risk:
        val_results = {
        "c_index": c_index,
        "auc_score": auc_score,
        "all_risk_scores": all_risk_scores,
        "all_event_times": all_event_times,
        "all_censorships": all_censorships
        }
    else:
        val_results = {
        "c_index": c_index,
        "auc_score": auc_score,
        }
    return val_results

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()