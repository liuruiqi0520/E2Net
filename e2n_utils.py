from ofa.ofa_resnets import OFAResNets
import torch
from torch.nn import functional as F
import math

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
def get_net(args):
    if args.dataset == 'seq-cifar10' :
        backbone = OFAResNets(n_classes=10, depth_list=[1],
                         expand_ratio_list=[1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                         width_mult_list=[1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    elif args.dataset == 'seq-cifar100':
        backbone = OFAResNets(n_classes=100, depth_list=[1], expand_ratio_list=[1, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05],
                         width_mult_list=[1, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05])
    else:
        backbone = OFAResNets(n_classes=200, depth_list=[1], expand_ratio_list=[1, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05],
                         width_mult_list=[1, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05])
    return backbone

def self_search(t,model,sub_net_dict,last_config,finconfig_list,total_config_dict):
    CNS = 1
    z = torch.randn((32, 3, 32, 32))
    z = z.to(model.device)
    max_output = model.net(z)
    sub_loss_list = []
    sub_config_list = []
    super_parameters_list = []
    super_loss_list = []
    super_config_list = []
    super_net_list = []
    parameters_occupy = []
    model.net.re_organize_middle_weights()
    net_paramerters = get_parameter_number(model.net)["Trainable"]
    total_paramerters = 0
    limit_paramerters = net_paramerters * 0.95
    sub_limit_paramerters = limit_paramerters
    sub_paramerters = dict()
    if CNS:
        for i in range(500):
            sub_paramerters["Total"] = net_paramerters
            sub_config = model.net.sample_active_subnet2(t=t, config_list=last_config)
            subnet = model.net.get_active_subnet()
            sub_paramerters = get_parameter_number(subnet)
            occu = round(sub_paramerters["Total"] / net_paramerters, 3)
            sub_output = model.net(z)
            sub_loss = F.mse_loss(sub_output, max_output) * math.exp(-occu)
            # sub_loss = F.mse_loss(sub_output, max_output)
            sub_config_list.append(sub_config)
            sub_loss_list.append(sub_loss.item())

        sub_loss_list = torch.tensor(sub_loss_list)
        sorted_importance, sorted_idx = torch.sort(sub_loss_list, dim=0, descending=False)
        for i in range(30):
            model.net.set_active_subnet(**sub_config_list[sorted_idx[i]])
            subnet = model.net.get_active_subnet()
            sub_paramerters = get_parameter_number(subnet)
            total_paramerters += sub_paramerters["Total"]
            if total_paramerters < limit_paramerters and sub_paramerters["Total"] < sub_limit_paramerters:
                super_net_list.append(subnet)
                super_parameters_list.append(sub_paramerters["Total"])
                parameters_occupy.append(round(sub_paramerters["Total"] / net_paramerters, 3))
                super_loss_list.append(sub_loss_list[sorted_idx[i]].tolist())
                super_config_list.append(sub_config_list[sorted_idx[i]])
                if len(super_config_list) == 1: break
            else:
                total_paramerters -= sub_paramerters["Total"]
    else:
        model.net.set_active_subnet(maxnet=True)
        subnet = model.net.get_active_subnet()
        sub_output = model.net(z)
        sub_loss = F.mse_loss(sub_output, max_output)
        sub_paramerters = get_parameter_number(subnet)
        super_parameters_list = [sub_paramerters["Total"]]
        super_net_list = [subnet]
        super_config_list = [model.net.get_sub_config()]
        super_loss_list = [sub_loss.item()]
        parameters_occupy = [round(sub_paramerters["Total"] / net_paramerters, 3)]
    finconfig_list["config"].extend(super_config_list)
    finconfig_list["loss"].extend(super_loss_list)
    last_config = super_config_list
    sub_net_dict["net"] = super_net_list
    sub_net_dict["paramerters"] = super_parameters_list
    sub_net_dict["config"] = super_config_list
    sub_net_dict["loss"] = super_loss_list
    # if t ==1:
    #     parameters_occupy = [0.7+num for num in parameters_occupy]
    sub_net_dict["occupy"] = parameters_occupy
    total_config_dict["config"] = super_config_list
    total_config_dict["paramerters"] = super_parameters_list
    total_config_dict["loss"] = super_loss_list
    total_config_dict["occupy"] = parameters_occupy
    print(sub_net_dict["config"])
    print(sub_net_dict["paramerters"])
    print(sub_net_dict["occupy"])
    print(sub_net_dict["loss"])
    model.net.set_max_net()
    return last_config

def set_stage(model,t,dataset):
    model.net.set_stage(t, r=4, task=dataset.N_TASKS)
    # model.net.set_stage(t, r=1, task=dataset.N_TASKS)
    # model.net.set_stage(t, r=3.5, task=dataset.N_TASKS)