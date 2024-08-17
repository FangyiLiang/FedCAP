import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from prompt import Prompt
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.model import metaNet
_tokenizer = _Tokenizer()

def load_clip_to_cpu(args):
    backbone_name = args.model_name
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict(), args)



    return model


class FedFSCIL(nn.Module):
    def __init__(self,args,class_names):
        super().__init__()
        self.args=args
        self.class_names=class_names

        self.prompt_template=args.prompt_template
        self.vision_width=0
        self.build_model()
        self.prompt=Prompt(prompt_length=args.length,embed_dim=768,embedding_key=args.embedding_key, prompt_init=args.prompt_init, prompt_pool=args.prompt_pool, prompt_key=args.prompt_key, pool_size=args.size, top_k=args.top_k, batchwise_prompt=args.batchwise_prompt, prompt_key_init=args.prompt_key_init)

        self.metaNet=metaNet(768,768,512)

        self.text_tokens=[]

    def build_model(self):

        print(f"Loading CLIP (backbone: {self.args.model_name})")
        self.clip_model=load_clip_to_cpu(self.args)
        self.vision_width=self.clip_model.vision_width
        if self.args.trainer_prec =="fp32" or self.args.trainer_prec=="amp":
            # CLIP's default precision is fp16
            self.clip_model.float()

        self.dtype=self.clip_model.dtype


        print("Building custom CLIP")


    def set_parameters_grad(self):
        print("Turning off gradients in both the image and the text encoder")
        name_to_update="prompt"

        for name, param in self.clip_model.named_parameters():
            if name_to_update == name:

                param.requires_grad_(True)
            else:

                param.requires_grad_(False)


        self.prompt.key.requires_grad_(False)


    def adaptation(self,target,device):
        self.current_classnames=[]
        for i in target:
            self.current_classnames.append(self.class_names[i])
        self.text_tokens=[]
        self.text_tokens = clip.tokenize([self.prompt_template.format(c) for c in self.current_classnames]).to(device)

    #压根没用到
    def forward(self,input,text_tokens,train=False):

        return self.clip_model(input,text_tokens,self.prompt,self.metaNet,train=train)



