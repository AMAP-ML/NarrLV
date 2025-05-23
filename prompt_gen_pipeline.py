import cv2
import os
import numpy as np
import re
import json
import time
import argparse
from datetime import datetime
import csv
import tqdm
import random
import ast

import pandas as pd
import glob
import openai
import base64
from typing import Union, Optional



class MLLM_by_GPT4o:
    def __init__(self, api_key: str):   # 'YOUR_API_KEY_HERE' 
        openai.api_key = api_key

    def forward(self, image_path: Optional[str] = None, text: Optional[str] = None) -> str:
        if image_path:
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            image_input = {"image": encoded_image}
        else:
            image_input = {}

        messages = []
        if text:
            messages.append({"role": "user", "content": text})
        if image_input:
            messages.append({"role": "user", "content": image_input})

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages
        )

        return response['choices'][0]['message']['content']

def load_text(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def prompt_suit_gen_by_sample_set(vlm,tna_change_reason, tna_num ,prompt_sample_num, json_save_path):
    #  tna_change_reason: "scene_attribute", "target_action",   "target_attribute" 
    tna_change_selected = "Target attribute change"

    json_scene_load_path = "./resource/scene_object_set_infor.json"
    with open(json_scene_load_path,"r") as fp:  
        scene_set_ds = json.load(fp)

    scene_set_list = list(scene_set_ds.keys())


    instrust_prompt = load_text("./resource/instruct/instruct_prompt_suite_gen.txt")
    example_text = load_text(f"./resource/instruct/instruct_prompt_suite_gen_example/{tna_change_reason}_tna_{tna_num}.txt")

    
    save_res = []
    for _ in range(prompt_sample_num):
        scene_set_item = random.sample(scene_set_list, 1)[0]
        scene_item_list = list(scene_set_ds[scene_set_item].keys())
        scene_item = random.sample(scene_item_list, 1)[0]
        scene_item_infor = scene_set_ds[scene_set_item][scene_item]
        try:
            # get the object list
            object_list = []
            for object_item in scene_item_infor:
                object_list.append(object_item)
            
            
            object_selected_list = random.sample(object_list, 1)  # or 2

            object_selected_list = ", ".join(object_selected_list)

            prompt_detail = f"- Scene: {scene_info}\n- Targets: {object_selected_list}\n- TNA count: {tna_num}\n- Reason for TNA change: {tna_change_reason}"

            gpt_prompt = instrust_prompt.format(tna_num,tna_change_reason,tna_change_reason,tna_num,tna_change_reason,tna_num,example_text,prompt_detail)

            re = vlm.forward(text= gpt_prompt)
            time.sleep(2) 
            save_item_info = {}
            save_item_info["scene_category"] = scene_info
            save_item_info["object_selected_list"] = object_selected_list
            save_item_info["tna_change_reason"] = tna_change_reason
            save_item_info["prompt_gen"] = re
        except:
            continue

        save_res.append(save_item_info)
        
    with open(json_save_path, 'w', encoding='utf-8') as json_file:
        json.dump(save_res, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate evaluation prompts based on TNA factors and TNA nums.')
    
    parser.add_argument('--tna_factor', type=str, required=True, 
                        help='The TNA change factor, e.g., scene_attribute, target_attribute, target_action')
    parser.add_argument('--tna_num', type=int, required=True, 
                        help='The number of TNAs, e.g., 1,2,3,4,...')

    parser.add_argument('--prompt_sample_num', type=int, required=False, default=1,
                        help='The number of sampled prompts, e.g., 1,2,3,4,...')
    parser.add_argument('--save_path', type=str, required=False, default="./resource/prompt_suite_demo.json",
                        help='the path to save the generated prompts')
    
    args = parser.parse_args()
    
    api_key = 'YOUR_API_KEY_HERE' 
    my_mllm = MLLM_by_GPT4o(api_key)  # 'YOUR_API_KEY_HERE' 
    prompt_suit_gen_by_sample_set(my_mllm,args.tna_factor, args.tna_num ,args.prompt_sample_num, args.save_path)

