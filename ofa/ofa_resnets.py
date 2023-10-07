import random
import math
from ofa.modules.dynamic_layers import (
    DynamicConvLayer,
    DynamicLinearLayer,
)
from ofa.modules.dynamic_layers import (
    DynamicResNetBottleneckBlock,
)
from ofa.utils.layers import IdentityLayer, ResidualBlock
from ofa.networks import ResNets
from ofa.utils import make_divisible, val2list, MyNetwork
import torch
from torch.nn import functional as F
from torchkeras import summary
import torch.nn as nn
from torch.nn.functional import relu
__all__ = ["OFAResNets"]
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

class OFAResNets(ResNets):
    def __init__(
        self,
        n_classes=1000,
        bn_param=(0.1, 1e-5),
        dropout_rate=0,
        depth_list=2,
        expand_ratio_list=1,
        width_mult_list=[1,2,3],
    ):

        self.depth_list = val2list(depth_list)
        self.expand_ratio_list = val2list(expand_ratio_list)
        self.width_mult_list = val2list(width_mult_list)
        # sort
        self.depth_list.sort()
        self.expand_ratio_list.sort()
        self.width_mult_list.sort()
        self.stage = 0
        self.sub_config = {"d": None, "e":None , "w":None }

        input_channel = [
            make_divisible(64 * width_mult, MyNetwork.CHANNEL_DIVISIBLE)
            for width_mult in self.width_mult_list
        ]
        mid_input_channel = [
            make_divisible(channel // 2, MyNetwork.CHANNEL_DIVISIBLE)
            for channel in input_channel
        ]

        stage_width_list = ResNets.STAGE_WIDTH_LIST.copy()
        for i, width in enumerate(stage_width_list):
            stage_width_list[i] = [
                make_divisible(width * width_mult, MyNetwork.CHANNEL_DIVISIBLE)
                for width_mult in self.width_mult_list
            ]

        n_block_list = [
            base_depth + max(self.depth_list) for base_depth in ResNets.BASE_DEPTH_LIST
        ]
        stride_list = [1, 2, 2, 2]

        # build input stem
        # input_stem = [
        #     DynamicConvLayer(
        #         val2list(3),
        #         mid_input_channel,
        #         3,
        #         stride=1,
        #         use_bn=True,
        #         act_func="relu",
        #     ),
        #     ResidualBlock(
        #         DynamicConvLayer(
        #             mid_input_channel,
        #             mid_input_channel,
        #             3,
        #             stride=1,
        #             use_bn=True,
        #             act_func="relu",
        #         ),
        #         IdentityLayer(mid_input_channel, mid_input_channel),
        #     ),
        #     DynamicConvLayer(
        #         mid_input_channel,
        #         input_channel,
        #         3,
        #         stride=1,
        #         use_bn=True,
        #         act_func="relu",
        #     ),
        # ]
        input_stem = [
            DynamicConvLayer(
                val2list(3),
                input_channel,
                3,
                stride=1,
                use_bn=True,
                act_func="relu",
            ),
        ]

        # blocks
        blocks = []
        # for d, width, s in zip(n_block_list, stage_width_list, stride_list):
        #     for i in range(d):
        #         stride = s if i == 0 else 1
        #         bottleneck_block = DynamicResNetBottleneckBlock(
        #             input_channel,
        #             width,
        #             expand_ratio_list=self.expand_ratio_list,
        #             kernel_size=3,
        #             stride=stride,
        #             act_func="relu",
        #             downsample_mode="avgpool_conv",
        #         )
        #         blocks.append(bottleneck_block)
        #         input_channel = width
        for d, width, s in zip(n_block_list, stage_width_list, stride_list):
            for i in range(d):
                stride = s if i == 0 else 1
                bottleneck_block = DynamicResNetBottleneckBlock(
                    input_channel,
                    width,
                    expand_ratio_list=self.expand_ratio_list,
                    kernel_size=3,
                    stride=stride,
                    act_func="relu",
                    downsample_mode="conv",
                )
                blocks.append(bottleneck_block)
                input_channel = width
        # classifier
        classifier = DynamicLinearLayer(
            input_channel, n_classes, dropout_rate=dropout_rate
        )

        super(OFAResNets, self).__init__(input_stem, blocks, classifier)

        # set bn param
        self.set_bn_param(*bn_param)

        # runtime_depth
        self.input_stem_skipping = 0
        self.runtime_depth = [0] * len(n_block_list)

    @property
    def ks_list(self):
        return [3]

    @staticmethod
    def name():
        return "OFAResNets"

    def get_sub_config(self):
        return self.sub_config

    def forward(self, x):
        for layer in self.input_stem:
            if (
                self.input_stem_skipping > 0
                and isinstance(layer, ResidualBlock)
                and isinstance(layer.shortcut, IdentityLayer)
            ):
                pass
            else:
                x = layer(x)
        # x = self.max_pooling(x)
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                x = self.blocks[idx](x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = ""
        for layer in self.input_stem:
            if (
                self.input_stem_skipping > 0
                and isinstance(layer, ResidualBlock)
                and isinstance(layer.shortcut, IdentityLayer)
            ):
                pass
            else:
                _str += layer.module_str + "\n"
        _str += "max_pooling(ks=3, stride=2)\n"
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + "\n"
        _str += self.global_avg_pool.__repr__() + "\n"
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            "name": OFAResNets.__name__,
            "bn": self.get_bn_param(),
            "input_stem": [layer.config for layer in self.input_stem],
            "blocks": [block.config for block in self.blocks],
            "classifier": self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        raise ValueError("do not support this function")

    def load_state_dict(self, state_dict, **kwargs):
        model_dict = self.state_dict()
        for key in state_dict:
            new_key = key
            if new_key in model_dict:
                pass
            elif ".linear." in new_key:
                new_key = new_key.replace(".linear.", ".linear.linear.")
            elif "bn." in new_key:
                new_key = new_key.replace("bn.", "bn.bn.")
            elif "conv.weight" in new_key:
                new_key = new_key.replace("conv.weight", "conv.conv.weight")
            else:
                raise ValueError(new_key)
            assert new_key in model_dict, "%s" % new_key
            model_dict[new_key] = state_dict[key]
        super(OFAResNets, self).load_state_dict(model_dict)

    """ set, sample and get active sub-networks """

    def set_max_net(self):
        self.set_active_subnet(
            d=max(self.depth_list),
            e=max(self.expand_ratio_list),
            w=len(self.width_mult_list) - 1,
        )

    def set_active_subnet(self, d=None, e=None, w=None, maxnet=False, **kwargs):
        if maxnet:
            d = max(self.depth_list)
            e = self.expand_ratio_list[self.stage-1]
            w = self.stage-1
            self.sub_config = {"d": d, "e":e , "w":w }
        depth = val2list(d, len(ResNets.BASE_DEPTH_LIST) + 1)
        expand_ratio = val2list(e, len(self.blocks))
        # width_mult = val2list(w, len(ResNets.BASE_DEPTH_LIST) + 2)
        width_mult = val2list(w, len(ResNets.BASE_DEPTH_LIST) + 1)

        for block, e in zip(self.blocks, expand_ratio):
            if e is not None:
                block.active_expand_ratio = e

        # if width_mult[0] is not None:
        #     self.input_stem[1].conv.active_out_channel = self.input_stem[
        #         0
        #     ].active_out_channel = self.input_stem[0].out_channel_list[width_mult[0]]
        # if width_mult[1] is not None:
        #     self.input_stem[2].active_out_channel = self.input_stem[2].out_channel_list[
        #         width_mult[1]
        #     ]
        #
        # if depth[0] is not None:
        #     self.input_stem_skipping = depth[0] != max(self.depth_list)
        # for stage_id, (block_idx, d, w) in enumerate(
        #     zip(self.grouped_block_index, depth[1:], width_mult[2:])
        # ):
        if width_mult[0] is not None:
            self.input_stem[0].active_out_channel = self.input_stem[0].out_channel_list[
                    width_mult[0]
                ]
        for stage_id, (block_idx, d, w) in enumerate(
                zip(self.grouped_block_index, depth, width_mult)
        ):
            if d is not None:
                self.runtime_depth[stage_id] = max(self.depth_list) - d
            if w is not None:
                for idx in block_idx:
                    self.blocks[idx].active_out_channel = self.blocks[
                        idx
                    ].out_channel_list[w]

    def sample_active_subnet(self, t = None, r= True):
        len_max=len(self.width_mult_list)
        if t == None:
            expand_ratio_list = self.expand_ratio_list
            out_channel_len = 0
            depth_list = self.depth_list

        elif t == 0:
            if r:
                expand_ratio_list = val2list(max(self.expand_ratio_list), len(self.blocks))
            else:
                expand_ratio_list = self.expand_ratio_list[:1]
            depth_list = val2list(max(self.depth_list), len(self.blocks))
            out_channel_len = len_max-1
        # elif t < len(self.depth_list):
        #     depth_list = self.depth_list[::-1]
        #     depth_list = depth_list[:t + 1]
        #     expand_ratio_list = val2list(max(self.expand_ratio_list), len(self.blocks))
        #     out_channel_len = len_max-1
        # else:
        #     t = t - len(self.depth_list) + 1
        #     t = min(t,len(self.width_mult_list)-1)
        #     expand_ratio_list = self.expand_ratio_list[::-1]
        #     expand_ratio_list = expand_ratio_list[:t + 1]
        #     depth_list = self.depth_list
        #     out_channel_len = len_max-t-1
        elif t < len(self.expand_ratio_list):
            if r:
                expand_ratio_list = self.expand_ratio_list[::-1]
            else:
                expand_ratio_list = self.expand_ratio_list
            expand_ratio_list = expand_ratio_list[:t + 1]
            depth_list = val2list(max(self.depth_list), len(self.blocks))
            out_channel_len = len_max-1
        else:
            t = t - len(self.expand_ratio_list) + 1
            expand_ratio_list = self.expand_ratio_list
            depth_list = val2list(max(self.depth_list), len(self.blocks))
            out_channel_len = len_max - t - 1
        # sample expand ratio
        expand_setting = []
        for block in self.blocks:
            expand_setting.append(random.choice(expand_ratio_list))

        # sample depth
        # depth_setting = [random.choice([max(self.depth_list), min(self.depth_list)])]
        depth_setting = []
        for stage_id in range(len(ResNets.BASE_DEPTH_LIST)):
            depth_setting.append(random.choice(depth_list))

        # sample width_mult
        # width_mult_setting = [
        #     random.choice(list(range(len(self.input_stem[0].out_channel_list)))),
        #     random.choice(list(range(len(self.input_stem[2].out_channel_list)))),
        # ]
        width_mult_setting = []
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            stage_first_block = self.blocks[block_idx[0]]
            width_mult_setting.append(
                random.choice(list(range(out_channel_len,len_max)))
            )

        arch_config = {"d": depth_setting, "e": expand_setting, "w": width_mult_setting}
        self.set_active_subnet(**arch_config)
        return arch_config

    def sample_active_subnet2(self, t = None, config_list = None):
        len_min=0
        if t == None:
            expand_ratio_list = self.expand_ratio_list
            out_channel_len = len(self.width_mult_list)
            depth_list = self.depth_list
        elif self.stage <= len(self.expand_ratio_list):
            expand_ratio_list = self.expand_ratio_list
            # expand_ratio_list = expand_ratio_list[:t + 1]
            expand_ratio_list = expand_ratio_list[:self.stage]
            depth_list = val2list(max(self.depth_list), len(self.blocks))
            out_channel_len = self.stage
            len_min = 0
        else:
            expand_ratio_list = val2list(max(self.expand_ratio_list), len(self.blocks))
            out_channel_len = len(self.width_mult_list)
            len_min = len(self.width_mult_list) - 1
            depth_list = val2list(max(self.depth_list), len(self.blocks))
        if len(config_list) == 0 or config_list==None:
            # sample expand ratio
            expand_setting = []
            for block in self.blocks:
                expand_setting.append(random.choice(expand_ratio_list))

            # sample depth
            depth_setting = []
            for stage_id in range(len(ResNets.BASE_DEPTH_LIST)):
                depth_setting.append(random.choice(depth_list))

            # sample width_mult

            width_mult_setting = []
            for stage_id, block_idx in enumerate(self.grouped_block_index):
                stage_first_block = self.blocks[block_idx[0]]
                width_mult_setting.append(
                    random.choice(list(range(len_min,out_channel_len)))
                )
        else:
            # sample expand ratio
            e_list=[]
            w_list=[]
            for config_dict in config_list:
                e_list.append(config_dict["e"])
                w_list.append(config_dict["w"])
            e_list = list(map(list, zip(*e_list)))
            w_list = list(map(list, zip(*w_list)))
            expand_setting = []
            for i,block in enumerate(self.blocks):
                e = -1
                while(e < max(e_list[i])):
                    e = random.choice(expand_ratio_list)
                expand_setting.append(e)

            # sample depth
            depth_setting = []
            for stage_id in range(len(ResNets.BASE_DEPTH_LIST)):
                depth_setting.append(random.choice(depth_list))

            # sample width_mult

            width_mult_setting = []
            for stage_id, block_idx in enumerate(self.grouped_block_index):
                len_min = max(w_list[stage_id])
                stage_first_block = self.blocks[block_idx[0]]
                width_mult_setting.append(
                    random.choice(list(range(len_min, out_channel_len)))
                )


        arch_config = {"d": depth_setting, "e": expand_setting, "w": width_mult_setting}
        self.set_active_subnet(**arch_config)
        return arch_config

    def get_active_subnet(self, preserve_weight=True):
        input_stem = [self.input_stem[0].get_active_subnet(3, preserve_weight)]
        # if self.input_stem_skipping <= 0:
        #     input_stem.append(
        #         ResidualBlock(
        #             self.input_stem[1].conv.get_active_subnet(
        #                 self.input_stem[0].active_out_channel, preserve_weight
        #             ),
        #             IdentityLayer(
        #                 self.input_stem[0].active_out_channel,
        #                 self.input_stem[0].active_out_channel,
        #             ),
        #         )
        #     )
        # input_stem.append(
        #     self.input_stem[2].get_active_subnet(
        #         self.input_stem[0].active_out_channel, preserve_weight
        #     )
        # )
        # input_channel = self.input_stem[2].active_out_channel
        input_channel = self.input_stem[0].active_out_channel

        blocks = []
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                blocks.append(
                    self.blocks[idx].get_active_subnet(input_channel, preserve_weight)
                )
                input_channel = self.blocks[idx].active_out_channel
        classifier = self.classifier.get_active_subnet(input_channel, preserve_weight)
        subnet = ResNets(input_stem, blocks, classifier)

        subnet.set_bn_param(**self.get_bn_param())
        return subnet

    def get_active_net_config(self):
        input_stem_config = [self.input_stem[0].get_active_subnet_config(3)]
        if self.input_stem_skipping <= 0:
            input_stem_config.append(
                {
                    "name": ResidualBlock.__name__,
                    "conv": self.input_stem[1].conv.get_active_subnet_config(
                        self.input_stem[0].active_out_channel
                    ),
                    "shortcut": IdentityLayer(
                        self.input_stem[0].active_out_channel,
                        self.input_stem[0].active_out_channel,
                    ),
                }
            )
        input_stem_config.append(
            self.input_stem[2].get_active_subnet_config(
                self.input_stem[0].active_out_channel
            )
        )
        input_channel = self.input_stem[2].active_out_channel

        blocks_config = []
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                blocks_config.append(
                    self.blocks[idx].get_active_subnet_config(input_channel)
                )
                input_channel = self.blocks[idx].active_out_channel
        classifier_config = self.classifier.get_active_subnet_config(input_channel)
        return {
            "name": ResNets.__name__,
            "bn": self.get_bn_param(),
            "input_stem": input_stem_config,
            "blocks": blocks_config,
            "classifier": classifier_config,
        }

    """ Width Related Methods """

    def re_organize_middle_weights_r(self, expand_ratio_stage=0):
        for block in self.blocks:
            block.re_organize_middle_weights_r(expand_ratio_stage)
    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for block in self.blocks:
            block.re_organize_middle_weights(expand_ratio_stage)

    def set_stage(self,t, r, task):
        x = t
        p = round(r/2*(1+math.cos(x*math.pi/(task-1))))
        print(p)
        if(x!=(task-1)):
            p = max(p,1)
        self.stage += p
        print(self.stage)
#测试用的代码
if __name__ == "__main__":
    print(4/2*(1+math.cos(3*math.pi/4)))
    # x=[10,20]
    # y = [10,20,15]
    # z=torch.randn((1,3,224,224))
    # layer1=OFAResNets(n_classes=10)
    # k1=layer1(z)
    # print(k1)
    # # layer1.sample_active_subnet()
    # # layer2=layer1.get_active_subnet()
    # # k2 = layer1(z)
    # # print(k2)
    # # layer1.set_active_subnet(d=2, e=1,w=2)
    # # k3 = layer1(z)
    # # print(k3)
    # # layer1.set_max_net()
    # # k4 = layer1(z)
    # # print(k4)
    # # print(k1 == k3)
    # # print(k1 == k4)
    # # l2=layer1.get_active_subnet()
    # # print(l2)
    # # print(layer1.get_active_subnet())
    # # print(layer1)
    # # g = torchviz.make_dot(k, params=dict(layer1.named_parameters()))
    # # g.view()
    # print(layer1)
    # range_list=list(range(0,91))
    # range_list = [x/10 for x in range_list]
    # range_list.append(1)
    # net = OFAResNets(n_classes=10, depth_list=[1], expand_ratio_list=[1, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05],
    #                  width_mult_list=[1, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05])
    net = OFAResNets(n_classes=10, depth_list=[1], expand_ratio_list=[1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                     width_mult_list=[1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    # net = OFAResNets(n_classes=10, depth_list=[1], expand_ratio_list=[1, 0.8, 0.6, 0.4, 0.2],
    #                  width_mult_list=[1, 0.8, 0.6, 0.4, 0.2])
    total_space = get_parameter_number(net)["Trainable"]
    total_space2 = get_parameter_number(net)["Trainable"]*0.95
    last_config = None
    for t in range(0,5):
        # z = torch.randn((1, 3, 32, 32))
        # max_output = net(z)
        # sub_loss_list = []
        # sub_config_list = []
        # sub_net_dict = dict()
        # super_paramerters_list = []
        # super_loss_list = []
        # super_config_list = []
        # super_net_list = []
        # super_space_list = []
        # model.net.re_organize_middle_weights()
        net.set_stage(t,r=4,task=5)
        net.set_active_subnet(maxnet=True)
        # for i in range(1000):
        #     sub_config = net.sample_active_subnet2(t=t, config_list=last_config)
        #     sub_output = net(z)
        #     sub_loss = F.mse_loss(sub_output, max_output)
        #     sub_config_list.append(sub_config)
        #     sub_loss_list.append(sub_loss.item())
        #
        # sub_loss_list = torch.tensor(sub_loss_list)
        # sorted_importance, sorted_idx = torch.sort(sub_loss_list, dim=0, descending=False)
        # total_paramerters = 0
        # limit_paramerters = 20000000
        # sub_limit_paramerters = limit_paramerters
        # for i in range(30):
        #     net.set_active_subnet(**sub_config_list[sorted_idx[i]])
        #     subnet = net.get_active_subnet()
        #     sub_paramerters = get_parameter_number(subnet)
        #     total_paramerters += sub_paramerters["Total"]
        #     if total_paramerters < limit_paramerters and sub_paramerters["Total"] < sub_limit_paramerters:
        #         super_net_list.append(subnet)
        #         super_paramerters_list.append(sub_paramerters)
        #         super_loss_list.append(sub_loss_list[sorted_idx[i]])
        #         super_config_list.append(sub_config_list[sorted_idx[i]])
        #         super_space_list.append(total_space-sub_paramerters["Trainable"])
        #         if len(super_config_list) == 3: break
        #     else:
        #         total_paramerters -= sub_paramerters["Total"]
        # last_config = super_config_list
        # sub_net_dict["net"] = super_net_list
        # sub_net_dict["paramerters"] = super_paramerters_list
        # sub_net_dict["config"] = super_config_list
        # sub_net_dict["loss"] = super_loss_list
        # sub_net_dict["space"] = super_space_list
        # print(sub_net_dict["config"])
        # print(sub_net_dict["paramerters"])
        # print(sub_net_dict["loss"])
        # print(sub_net_dict["space"])
