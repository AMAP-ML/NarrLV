from PIL import Image
import requests
import copy
import torch
import sys
import warnings
import os
# from decord import VideoReader, cpu
import numpy as np
import json
import argparse
from tqdm import tqdm
import cv2
import time


import re
import time
from datetime import datetime
import csv
import random
import ast
from transformers import AutoModelForCausalLM



def benchmark_gen_results(model_name,video_gen_dir, model,processor,answer_save_path):
    
    save_res_dir = answer_save_path
    os.makedirs(save_res_dir,exist_ok=True)
    json_load_dir = "./resource/prompt_suit/"

    img_save_temp_dir = f"./resource/img_temp_save_{model_name}"
    os.makedirs(img_save_temp_dir,exist_ok=True)


    video_gen_sub_list_0 = os.listdir(video_gen_dir ) 

    for video_gen_sub_item in video_gen_sub_list_0:

        json_load_name = video_gen_sub_item  +".json"  # .replace("v3","v4") +".json"
        json_load_path = os.path.join(json_load_dir,json_load_name)
        with open(json_load_path,"r") as fp:  
            item_info = json.load(fp)

        json_save_path = os.path.join(save_res_dir, video_gen_sub_item  +".json")
        video_gen_sub_dir = os.path.join(video_gen_dir,video_gen_sub_item)
        video_gen_sub_list = os.listdir( video_gen_sub_dir )

        if os.path.exists(json_save_path):
            with open(json_save_path,"r") as fp:  
                save_res = json.load(fp)
        else:
            save_res = {}
        save_index = 0
        for gen_item in video_gen_sub_list:
            print(f"{video_gen_sub_dir},{gen_item}")
            if gen_item[:-4] in save_res:
                continue

            gen_item_name_infor = gen_item.split("_")
            scene_item = gen_item_name_infor[0]
            scene_item_index = gen_item_name_infor[1]
            try:
                gen_prompt_infor = item_info[scene_item][int(scene_item_index)]

                # load first frame
                file_list = os.listdir( img_save_temp_dir )
                for file_item in file_list:
                    os.remove(os.path.join(img_save_temp_dir, file_item))
                video_path = os.path.join(video_gen_sub_dir, gen_item)
                cap = cv2.VideoCapture(video_path)
                ret, frame = cap.read()
                video_first_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_save_path = os.path.join(img_save_temp_dir, "qalign_first_frame.jpg" )
                video_first_frame.save(img_save_path )

                cap.release()

                save_qa_item = {}
                save_qa_item["prompt_gen"] = gen_prompt_infor["prompt_gen"]
                save_qa_item["element_extract_1"] = gen_prompt_infor["element_extract_1"]
                save_qa_item["auto_gen_question_1"] = gen_prompt_infor["auto_gen_question_1"]
                event_list = gen_prompt_infor["prompt_gen_list_extract"]

                qa_item = {}
                
                img_input = img_save_path

                qalign_score = model.score([Image.open(img_input).convert('RGB')], task_="quality", input_="image") # task_ : quality | aesthetics; # input_: image | video
                qa_item["answer"] = qalign_score.detach().cpu().tolist()

                save_qa_item["qalign_res"] = qa_item
                
                prompt_gen = save_qa_item["prompt_gen"]
                print(f"{save_index}.\nQuestion:{prompt_gen}.\nAnswer:{qa_item}.")
                save_index += 1

                save_res[gen_item[:-4]] = save_qa_item
                
                with open(json_save_path, 'w', encoding='utf-8') as json_file:
                    json.dump(save_res, json_file, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"{video_gen_sub_dir},{gen_item}, error:{e}")
            
        with open(json_save_path, 'w', encoding='utf-8') as json_file:
            json.dump(save_res, json_file, ensure_ascii=False, indent=4)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate answers by evaluation prompts.')
    
    parser.add_argument('--evaluated_model_name', type=str, required=False, default="CogVideoX1.5-5B",
                        help='The model name, e.g., CogVideoX1.5-5B')
    parser.add_argument('--evaluated_model_videos_path', type=str, required=False, default="./resource/generated_videos",
                        help='The path saves the generated videos')
    parser.add_argument('--mllm_model_path', type=str, required=False, default="./resource/pretrained_model/one-align",
                        help='The model path of Q-align')

    parser.add_argument('--answer_save_path', type=str, required=False, default="./resource/generated_answers_aes",
                        help='the path to save the generated answers')
    
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.mllm_model_path, trust_remote_code=True, 
                                                torch_dtype=torch.float16, device_map="auto")

    processor = None

    
    benchmark_gen_results(args.evaluated_model_name, args.evaluated_model_videos_path, model,  processor ,  args.answer_save_path)              

