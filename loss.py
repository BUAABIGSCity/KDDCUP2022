# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FilterMSELoss", "FilterHuberLoss"]


class FilterMSELoss(nn.Module):
    def __init__(self, **kwargs):
        super(FilterMSELoss, self).__init__()
        print('FilterMSELoss')

    def forward(self, pred, gold, raw, col_names):
        # Remove bad input
        cond1 = raw[:, :, :, col_names["Patv"]] < 0

        cond2 = raw[:, :, :, col_names["Pab1"]] > 89
        cond2 = torch.logical_or(cond2, raw[:, :, :, col_names["Pab2"]] > 89)
        cond2 = torch.logical_or(cond2, raw[:, :, :, col_names["Pab3"]] > 89)

        cond2 = torch.logical_or(cond2,
                                  raw[:, :, :, col_names["Wdir"]] < -180)
        cond2 = torch.logical_or(cond2, raw[:, :, :, col_names["Wdir"]] > 180)
        cond2 = torch.logical_or(cond2,
                                  raw[:, :, :, col_names["Ndir"]] < -720)
        cond2 = torch.logical_or(cond2, raw[:, :, :, col_names["Ndir"]] > 720)
        cond2 = torch.logical_or(cond2, cond1)

        cond3 = raw[:, :, :, col_names["Patv"]] == 0
        cond3 = torch.logical_and(cond3,
                                   raw[:, :, :, col_names["Wspd"]] > 2.5)
        cond3 = torch.logical_or(cond3, cond2)

        cond = torch.logical_not(cond3)
        cond = cond.float()
        # cond = torch.cast(cond, "float32")

        return torch.mean(F.mse_loss(pred, gold, reduction='none') * cond)


class FilterHuberLoss(nn.Module):
    def __init__(self, delta=5, **kwargs):
        super(FilterHuberLoss, self).__init__()
        self.delta = delta
        print('FilterHuberLoss', 'delta = {}'.format(self.delta))

    def forward(self, pred, gold, raw, col_names):
        # Remove bad input
        cond1 = raw[:, :, :, col_names["Patv"]] < 0

        cond2 = raw[:, :, :, col_names["Pab1"]] > 89
        cond2 = torch.logical_or(cond2, raw[:, :, :, col_names["Pab2"]] > 89)
        cond2 = torch.logical_or(cond2, raw[:, :, :, col_names["Pab3"]] > 89)

        cond2 = torch.logical_or(cond2,
                                  raw[:, :, :, col_names["Wdir"]] < -180)
        cond2 = torch.logical_or(cond2, raw[:, :, :, col_names["Wdir"]] > 180)
        cond2 = torch.logical_or(cond2,
                                  raw[:, :, :, col_names["Ndir"]] < -720)
        cond2 = torch.logical_or(cond2, raw[:, :, :, col_names["Ndir"]] > 720)
        cond2 = torch.logical_or(cond2, cond1)

        cond3 = raw[:, :, :, col_names["Patv"]] == 0
        cond3 = torch.logical_and(cond3,
                                   raw[:, :, :, col_names["Wspd"]] > 2.5)
        cond3 = torch.logical_or(cond3, cond2)

        cond = torch.logical_not(cond3)
        cond = cond.float()
        # cond = torch.cast(cond, "float32")

        return torch.mean(F.smooth_l1_loss(pred, gold, reduction='none', beta=self.delta) * cond)
