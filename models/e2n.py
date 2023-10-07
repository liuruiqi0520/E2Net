# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch
import os
import random
import math
import numpy as np
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    return parser


class E2N(ContinualModel):
    NAME = 'e2n'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(E2N, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.current_task = 0
        self.supernet_index = 0
        self.sub_buffer = [0]*5


    def observe(self, inputs, labels, not_aug_inputs, sub_net_dict=None, config_list=None):
        # lo=dict()
        self.net.set_max_net()
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels.long())
        occupy = 0
        if sub_net_dict:
            # self.net.re_organize_middle_weights()
            index = random.choice(list(range(0, len(sub_net_dict["config"]))))
            sub_config = sub_net_dict["config"][index]
            subnet = sub_net_dict["net"][index]
            occupy = sub_net_dict["occupy"][index]
            self.net.set_active_subnet(**sub_config)
            sub_logits=subnet(inputs)
            sub_outputs=self.net(inputs)
            sub_loss = F.mse_loss(sub_outputs, sub_logits)
            loss += 0.05*sub_loss

        #####################################################################


        ran2 = random.random()
        if ran2 < 0.25:
            if not self.buffer.is_empty():
                self.net.set_max_net()
                buf_inputs, _, buf_logits = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform)
                buf_outputs = self.net(buf_inputs)
                loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

                buf_inputs, buf_labels, _ = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform)
                buf_outputs = self.net(buf_inputs)
                loss += self.args.beta * self.loss(buf_outputs, buf_labels)

        self.net.set_max_net()
        # config = self.net.sample_active_subnet2(t=1,config_list=config_list)
        # print(config)
        # self.net.set_active_subnet(d=config["d"],e=config["e"],w=config["w"])
        # self.net.set_active_subnet(maxnet=True)
        # output_s = self.net(inputs)
        loss.backward()
        self.opt.step()
        # self.net.set_max_net()

        ran = random.random()
        if ran<(math.exp(-occupy*0.75)):
            self.buffer.add_data(examples=not_aug_inputs,
                                 labels=labels,
                                 logits=outputs.data)
        # print(lo)

        return loss.item()

    def observe3(self, inputs, labels, not_aug_inputs, sub_net_dict=None, config_list=None):
        self.net.set_max_net()
        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels.long())
        if sub_net_dict:
            # self.net.re_organize_middle_weights()
            index = random.choice(list(range(0, len(sub_net_dict["config"]))))
            sub_config = sub_net_dict["config"][index]
            subnet = sub_net_dict["net"][index]
            self.net.set_active_subnet(**sub_config)
            sub_logits=subnet(inputs)
            sub_outputs=self.net(inputs)
            sub_loss = F.mse_loss(sub_outputs, sub_logits)
            loss += 0.1*sub_loss
        if not self.buffer.is_empty():
            buf_inputs, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
        # config = self.net.sample_active_subnet2(t=1,config_list=config_list)
        # # print(config)
        # self.net.set_active_subnet(d=config["d"],e=config["e"],w=config["w"])
        loss.backward()
        self.opt.step()
        self.net.set_max_net()
        self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)

        return loss.item()

    def observe2(self, inputs, labels, not_aug_inputs):

        self.net.set_max_net()
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels.long())
        #

        if not self.buffer.is_empty():
            self.net.set_max_net()
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)
        loss.backward()
        self.opt.step()
        self.net.set_max_net()
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)
        # print(lo)

        return loss.item()

    def end_task(self, dataset) -> None:
        self.current_task += 1
        model_dir = os.path.join(self.args.output_dir, "task_models", dataset.NAME, self.args.experiment_id)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.net, os.path.join(model_dir, f'task_{self.current_task}_model.ph'))



