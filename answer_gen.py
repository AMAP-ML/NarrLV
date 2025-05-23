from PIL import Image
import requests
import copy
import torch
import sys
import warnings
import os
from decord import VideoReader, cpu
import numpy as np
import json
import argparse
from tqdm import tqdm
import cv2
import time


import re
from datetime import datetime
import csv
import tqdm
import random
import ast
import pandas as pd
import glob
from vllm import LLM, SamplingParams
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))


    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx ]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        fps = sample_fps
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()    # frames, 480, 832, 3
    return spare_frames,frame_time,video_time,fps



def benchmark_gen_results(model_name,video_gen_dir, model,processor,answer_save_path):
    save_res_dir = answer_save_path
    os.makedirs(save_res_dir,exist_ok=True)

    json_load_dir = "./resource/prompt_suit/"

    img_save_temp_dir = f"./resource/img_temp_save_{model_name}"
    os.makedirs(img_save_temp_dir,exist_ok=True)


    generation_config = {
                    "temperature": 0.7,    
                    "top_p": 0.8,           
                    "repetition_penalty": 1.05, 
                    "max_new_tokens": 32,   
                }
    def inference(video_input, prompt,fps=1, max_new_tokens=32, total_pixels=20480 * 28 * 28, min_pixels=16 * 28 * 28):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"video": video_input},
                ]
            },
        ]
       
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info([messages])     #  image_inputs, video_inputs [60, 3, 532, 952] , video_kwargs  
       

        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps, padding=True, return_tensors="pt")
        inputs = inputs.to('cuda')

        output_ids = model.generate(**inputs, **generation_config) # max_new_tokens=max_new_tokens)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text[0]

    video_gen_sub_list_0 = os.listdir(video_gen_dir ) 

    for video_gen_sub_item in video_gen_sub_list_0:

        json_load_name = video_gen_sub_item  +".json" 
        json_load_path = os.path.join(json_load_dir,json_load_name)
        with open(json_load_path,"r") as fp:  
            item_info = json.load(fp)


        video_gen_sub_dir = os.path.join(video_gen_dir,video_gen_sub_item)
        video_gen_sub_list = os.listdir( video_gen_sub_dir )
        json_save_path = os.path.join(save_res_dir, video_gen_sub_item  +".json")
        

        if os.path.exists(json_save_path):
            with open(json_save_path,"r") as fp:  
                save_res = json.load(fp)
        else:
            save_res = {}
        save_index = 0
        for gen_item in video_gen_sub_list:
            if gen_item[:-4] in save_res:
                continue

            gen_item_name_infor = gen_item.split("_")
            scene_item = gen_item_name_infor[0]
            scene_item_index = gen_item_name_infor[1]
            try:
                gen_prompt_infor = item_info[scene_item][int(scene_item_index)]

                #### load video
                video_path = os.path.join(video_gen_sub_dir, gen_item)
                
                video,frame_time,video_time,fps = load_video(video_path, 10, 1, force_sample=True)

                file_list = os.listdir( img_save_temp_dir )
                for file_item in file_list:
                    os.remove(os.path.join(img_save_temp_dir, file_item))

                img_path_list = []
                for frame_index in range(video.shape[0]):
                    frame_item =  video[frame_index]
                    image_bgr = cv2.cvtColor(frame_item, cv2.COLOR_RGB2BGR)
                    img_save_path = os.path.join(img_save_temp_dir, f"{frame_index}.jpg" )
                    img_path_list.append( img_save_path  )
                    cv2.imwrite(img_save_path, image_bgr)


                save_qa_item = {}
                save_qa_item["prompt_gen"] = gen_prompt_infor["prompt_gen"]
                save_qa_item["element_extract_1"] = gen_prompt_infor["element_extract_1"]
                save_qa_item["auto_gen_question_1"] = gen_prompt_infor["auto_gen_question_1"]

                question_item_infor = gen_prompt_infor["auto_gen_question_1"]
                qa_item = {}
                for question_key in question_item_infor:
                    question_list = question_item_infor[question_key]
                    if question_list is None:
                        continue
                    if question_key in ["Scene Type","Main Target Category","Initial Scene Attributes","Main Target Layout"]:
                        img_input = [img_path_list[0]]
                    else:
                        img_input = img_path_list

                    qa_item_list = []
                    for question_item in question_list:
                        qa_item_item = {}
                        answer_list = []
                        for _ in range(5):
                            re = inference(
                                            img_input,
                                            question_item + " Only answer with yes or no, no additional explanation needed.",
                                            fps
                                            )

                            answer_list.append(re)
                        qa_item_item["question"] = question_item
                        qa_item_item["answer"] = answer_list
                        qa_item_list.append(qa_item_item)
                        print(f"{qa_item_item}")
                    qa_item[question_key] = qa_item_list
                save_qa_item["qa_infor"] = qa_item
                
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
    parser.add_argument('--mllm_model_path', type=str, required=False, default="./resource/pretrained_model/Qwen2.5-VL-72B-Instruct",
                        help='The mllm model path')

    parser.add_argument('--answer_save_path', type=str, required=False, default="./resource/generated_answers",
                        help='the path to save the generated answers')
    
    args = parser.parse_args()

    
    ### initial vlm
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.mllm_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.mllm_model_path)


    benchmark_gen_results( args.evaluated_model_name, args.evaluated_model_videos_path, vlm,processor,  args.answer_save_path)         





