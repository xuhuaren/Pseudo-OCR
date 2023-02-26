#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import string
import sys
from dataclasses import dataclass
from typing import List

import torch

from tqdm import tqdm

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args





@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--batch_size', type=int, default=1500)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
    parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
    parser.add_argument('--new', action='store_true', default=False, help='Evaluate on new benchmark datasets')
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--device', default='cuda')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    file1 = open('dict_challenge_v2.txt', 'r')
    lines=file1.readlines()
    file1.close()
    lines = [t.strip('\n') for t in lines]
    lines = "".join(lines)


    charset_test = lines

    kwargs.update({'charset_test': charset_test})
    print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    hp = model.hparams
    datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                     hp.charset_test, args.batch_size, args.num_workers, False, rotation=args.rotation, remove_whitespace=False, normalize_unicode=False)


    test_set = ['test_lmdb']


    filenames = []
    preds = []
    max_width = max(map(len, test_set))
    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        total = 0
        correct = 0
        ned = 0
        confidence = 0
        label_length = 0
        for imgs, labels,files in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
            #res = model.test_step((imgs.to(model.device), labels), -1)['output']
            p = model(imgs.to(model.device)).softmax(-1)
            pred, p = model.tokenizer.decode(p)
            filenames += files
            preds += pred
            
    with open(args.checkpoint + '_test_res_v2.txt', 'w') as f:
        N = len(filenames)
        for i in range(N):
            temp = "\t".join([filenames[i], preds[i]])
            print(temp)
            f.write(temp + '\n')



if __name__ == '__main__':
    main()
