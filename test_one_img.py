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



from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint
from strhub.models.parseq.system import PARSeq
import os 
from PIL import Image
import time



class ParseqRecognizer:
    def __init__(self):

        self.checkpoint = os.path.join("weights", "best.ckpt")
        self.model = PARSeq.load_from_checkpoint(self.checkpoint)
        self.img_transform = SceneTextDataModule.get_transform(self.model.hparams.img_size)

    def __call__(self, image):
        img = self.img_transform(image).unsqueeze(0)
        logits = self.model(img)
        pred = logits.softmax(-1)
        label, confidence = self.model.tokenizer.decode(pred)
        return label[0], confidence[0].detach().cpu().numpy().mean()
    
    
    
if __name__ == "__main__":
    from similarity.normalized_levenshtein import NormalizedLevenshtein
    normalized_levenshtein = NormalizedLevenshtein()
    model = ParseqRecognizer()
    file_path = "E:\\BaiduNetdiskDownload\\challenge\\code\\tmp_test"
    preds = []
    gts = []
    ress = []
    t1 = time.time()
    
    for file in os.listdir(file_path):
        image_path = os.path.join(file_path, file)
        gts.append(file.split("-")[-1].split(".")[0])
        res = model(Image.open(image_path).convert('RGB'))
        preds.append(res[0])
        ress.append(normalized_levenshtein.similarity(file.split("-")[-1].split(".")[0], res[0]))
        #print(res)
        
    t2 = time.time()
    print((t2 - t1) / len(os.listdir(file_path)) )
    
    
    
    
    
    
    
    
