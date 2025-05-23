import os
import torch
from diffusers.utils import export_to_video
from diffusers import  WanPipeline
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
import numpy as np
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
from PIL import Image
import json
from tqdm import tqdm
from diffusers import CogVideoXPipeline
from lib.video_generation_model import Wan_GEN, CogVideo_GEN
import argparse

def gen_tan_num(tna_file_dir,tna_file_item,pipe,save_dir0  ):
    json_file = os.path.join(tna_file_dir, tna_file_item )
    tna_num_index = tna_file_item[:-5]
    with open(json_file, 'r', encoding='utf-8') as file:
        tna_infor = json.load(file)
    
    save_dir = f"{save_dir0}/{tna_num_index}"
    
    os.makedirs(save_dir, exist_ok=True)
    for item in tna_infor:
        item_infor_list = tna_infor[item]
        for item_index,item_infor in enumerate(item_infor_list):
            prompt = item_infor["prompt_gen"]
            if prompt.startswith("```\n") and prompt.endswith("\n```"):
                prompt =  prompt[4:]
            if  prompt.endswith("\n```"):
                prompt =  prompt[:-4]
        
            save_path = f"{save_dir}/{item}_{item_index}_{prompt[:20]}.mp4"
            if os.path.isfile(save_path):
                continue
            output = pipe.generate(prompt,save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate videos by evaluation prompts.')
    
    parser.add_argument('--model_name', type=str, required=False, default="CogVideoX1.5-5B",
                        help='The model name, e.g., CogVideoX1.5-5B')
    parser.add_argument('--model_path', type=str, required=False, default="./resource/pretrained_model/CogVideoX1.5-5B",
                        help='The model path')
                     
    parser.add_argument('--save_path', type=str, required=False, default="./resource/generated_videos",
                        help='the path to save the generated videos')
    
    args = parser.parse_args()
            
    tna_file_dir = "./resource/prompt_suite"
    eval_model = CogVideo_GEN(args.model_path)
    
    tna_file_list = os.listdir(tna_file_dir) 
    for tna_file_item in tna_file_list:
        if ".json" in tna_file_item:
            gen_tan_num(tna_file_dir,tna_file_item,eval_model,args.save_path)

    


