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
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

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

import seaborn as sns
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy import interpolate
from learning_curve_main_results_smac import smooth
from scipy import stats

def cal_acc_item(answer_list,sample_num=5):
    acc = 0
    item_num = 0
    answer_list = answer_list[:sample_num]
    for answer_item in answer_list:
        if answer_item is None:
            continue
        answer_item = answer_item.lower()
        if "no" in answer_item or "no." in answer_item or "no," in answer_item:
            item_num += 1
            continue
        elif "yes" in answer_item or "yes." in answer_item or "yes," in answer_item:
            item_num += 1
            acc += 1

    return acc/item_num,item_num

def get_evo_infor(qa_infor,threshold=0.5):
        evo_infor = {}
        for k in qa_infor:
            if "Evolution_0" in k:
                evo_infor = qa_infor[k]
        evo_list = []
        evo_num = 0
        # import pdb;pdb.set_trace()
        for item in evo_infor:
            evo_list.append(item["acc"])
            if item["acc"] > threshold:
                evo_num += 1
        return evo_num,evo_list

def benchmark_cal_acc(model_name,answer_save_path,answer_aes_save_path,video_gen_dir,metric_save_path):

    model_gen_name = model_name

    save_res_dir = answer_save_path

    ## qalign
    qalign_res_dir = answer_aes_save_path

    save_eval_acc_res_dir = metric_save_path
    os.makedirs(save_eval_acc_res_dir, exist_ok=True)
    json_load_dir = "./resource/prompt_suit/"


    video_gen_sub_list = os.listdir(video_gen_dir ) 

    for video_gen_sub_item in video_gen_sub_list:
        json_load_name = video_gen_sub_item  +".json"
        json_load_path = os.path.join(json_load_dir,json_load_name)
        with open(json_load_path,"r") as fp:  
            item_info = json.load(fp)


        # print("json_load_name:",json_load_path)
        json_eval_res_path = os.path.join(save_res_dir, video_gen_sub_item  +".json")
        json_eval_acc_save_path = os.path.join(save_eval_acc_res_dir, video_gen_sub_item  +".json")
        

        with open(json_eval_res_path,"r") as fp:  
            json_eval_res = json.load(fp)

        video_0_eval_infor = os.path.join(qalign_res_dir, f"{video_gen_sub_item}.json")
        with open(video_0_eval_infor,"r") as fp:  
            qalin_eval_data = json.load(fp)

        eval_res_keys_list = list(qalin_eval_data.keys())

        save_res = {}
        qalign_score_list = []
        for gen_item in eval_res_keys_list:
            gen_item_name_infor = gen_item.split("_")
            scene_item = gen_item_name_infor[0]
            scene_item_index = gen_item_name_infor[1]

            gen_item_infor = json_eval_res[gen_item]
            
            qa_eval_infor = gen_item_infor["qa_infor"]

            save_qa_item = {}
            qa_item = {}
            for question_key in qa_eval_infor:
                question_list = qa_eval_infor[question_key]
                

                qa_item_list = []
                for question_item in question_list:
                    answer_list = question_item["answer"]

                    question_item["acc"],work_num = cal_acc_item(answer_list,sample_num=5)

                    qa_item_list.append(question_item)
                    if work_num <5:
                        print(f"{gen_item},{question_key}, {question_item}, work_num:{work_num}")
                qa_item[question_key] = qa_item_list
            save_qa_item["prompt_gen"] = gen_item_infor["prompt_gen"]
            save_qa_item["qa_infor"] = qa_item

            ##### cal fianl acc infor ########
            final_acc = {}
            for question_key in qa_item:
                qa_item_list = qa_item[question_key]
                temp_list = []
                for item in qa_item_list:
                    temp_list.append(item["acc"])
                final_acc[question_key] = np.mean(temp_list)
            

            # adjust "Main Target Layout"
            if "Main Target Category" in final_acc and "Scene Type" in final_acc:
                if final_acc["Main Target Category"] < 0.3 or final_acc["Scene Type"] < 0.3:
                    final_acc["Main Target Layout"]= 0

            # adjust "Evolution_0"
            evo_throsh = 0.3
            if "tna_1" not in video_gen_sub_item:
                if "Scene Attribute Evolution_0" in final_acc:
                    evo_0_list = []
                    for item in qa_item["Scene Attribute Evolution_0"]:
                        evo_0_list.append(item["acc"])

                    evo_1_list = []
                    for item in qa_item["Scene Attribute Evolution_1"]:
                        evo_1_list.append(item["acc"])
                    evo_1_list_0 = evo_1_list
                    
                    for evo_index in range(len(evo_1_list) ):
                        ev0_acc_0 = evo_0_list[evo_index]
                        ev0_acc_1 = evo_0_list[evo_index+1]
                        if ev0_acc_0 < evo_throsh or ev0_acc_1 < evo_throsh:
                            evo_1_list[evo_index] = 0
                    final_acc["Scene Attribute Evolution_1"]= np.mean(evo_1_list)
                elif "Main Target Action Evolution_0" in final_acc:
                    evo_0_list = []
                    for item in qa_item["Main Target Action Evolution_0"]:
                        evo_0_list.append(item["acc"])

                    evo_1_list = []
                    for item in qa_item["Main Target Action Evolution_1"]:
                        evo_1_list.append(item["acc"])

                    evo_1_list_0 = evo_1_list
                    
                    for evo_index in range(len(evo_1_list) ):
                        ev0_acc_0 = evo_0_list[evo_index]
                        ev0_acc_1 = evo_0_list[evo_index+1]
                        if ev0_acc_0 < evo_throsh or ev0_acc_1 < evo_throsh:
                            evo_1_list[evo_index] = 0
                    final_acc["Main Target Action Evolution_1"]= np.mean(evo_1_list)
                elif "Main Target Attribute Evolution_0" in final_acc:
                    evo_0_list = []
                    for item in qa_item["Main Target Attribute Evolution_0"]:
                        evo_0_list.append(item["acc"])

                    evo_1_list = []
                    for item in qa_item["Main Target Attribute Evolution_1"]:
                        evo_1_list.append(item["acc"])
                    
                    evo_1_list_0 = evo_1_list
                    
                    for evo_index in range(len(evo_1_list) ):
                        ev0_acc_0 = evo_0_list[evo_index]
                        ev0_acc_1 = evo_0_list[evo_index+1]
                        if ev0_acc_0 < evo_throsh or ev0_acc_1 < evo_throsh:
                            evo_1_list[evo_index] = 0
                    final_acc["Main Target Attribute Evolution_1"]= np.mean(evo_1_list)
                else:
                    print(f"wo evo !!!! xxxxxxxxxx json_eval_res_path:{json_eval_res_path}",len(json_eval_res))


            # adjust "Anomaly Detection"
            if "Scene Anomaly Detection" in final_acc:
                final_acc["Scene Anomaly Detection"] = 1- final_acc["Scene Anomaly Detection"]
            if "Main Target Anomaly Detection" in final_acc:
                final_acc["Main Target Anomaly Detection"] = 1- final_acc["Main Target Anomaly Detection"]

            #### get the final acc
            q_len = 0
            temp_list = []
            for question_key in ["Scene Type","Main Target Category","Initial Scene Attributes","Main Target Layout","Scene Anomaly Detection","Main Target Anomaly Detection"]:                
                if question_key in final_acc:
                    temp_list.append(final_acc[question_key])
            final_acc["metric_1_list"] = temp_list
            q_len += len(temp_list)
            final_acc["metric_1"] = np.mean(temp_list )

            temp_list = []
            for question_key in ["Scene Attribute Evolution_0","Main Target Action Evolution_0","Main Target Attribute Evolution_0"]:                
                if question_key in final_acc:
                    temp_list.append(final_acc[question_key])
            final_acc["metric_2_list"] = temp_list
            q_len += len(temp_list)
            final_acc["metric_2"] = np.mean(temp_list )

            temp_list = []
            for question_key in ["Scene Attribute Evolution_1","Main Target Action Evolution_1","Main Target Attribute Evolution_1"]:                
                if question_key in final_acc:
                    temp_list.append(final_acc[question_key])
            final_acc["metric_3_list"] = temp_list
            q_len += len(temp_list)
            final_acc["metric_3"] = np.mean(temp_list )

            #### add tna ratio in metric3
            eval_item0_evo_num,eval_item0_evo_list = get_evo_infor(qa_item,threshold=0.2)

            eval_item0_evo_score  =eval_item0_evo_num/len(eval_item0_evo_list)


            ### add qalign score
            qalign_score = 0.2*qalin_eval_data[gen_item]["qalign_res"]["answer"][0]
            qalign_score_list.append(qalign_score)

            final_acc["metric_1"] = (final_acc["metric_1"]  + qalign_score)/2

            final_acc["metric_2"] = (final_acc["metric_2"]  + qalign_score)/2
            final_acc["metric_3"] = (final_acc["metric_3"]  + qalign_score  + eval_item0_evo_score)/3
            

            
            save_qa_item["final_acc"] = final_acc
            
            save_res[gen_item] = save_qa_item
                
        with open(json_eval_acc_save_path, 'w', encoding='utf-8') as json_file:
            json.dump(save_res, json_file, ensure_ascii=False, indent=4)




def benchmark_cal_acc_final( save_eval_acc_res_dir  ):

    video_gen_sub_list = os.listdir(save_eval_acc_res_dir ) 
    
    acc_res = {}

    for video_gen_sub_item in video_gen_sub_list:
        json_eval_res_path = os.path.join(save_eval_acc_res_dir, video_gen_sub_item)

        
        with open(json_eval_res_path,"r") as fp:  
            json_eval_res = json.load(fp)

        acc_metric_1_list = []
        acc_metric_2_list = []
        acc_metric_3_list = []

        for gen_item in json_eval_res:
            try:
                final_acc_item = json_eval_res[gen_item]["final_acc"]
            except:
                import pdb;pdb.set_trace()
            final_acc_item = json_eval_res[gen_item]["final_acc"]
            if math.isnan( final_acc_item["metric_1"] ) == False:
                acc_metric_1_list.append( final_acc_item["metric_1"] )
            if math.isnan( final_acc_item["metric_2"] ) == False:
                acc_metric_2_list.append( final_acc_item["metric_2"] )
            if math.isnan( final_acc_item["metric_3"] ) == False:
                acc_metric_3_list.append( final_acc_item["metric_3"] )

        print(f"######## {video_gen_sub_item},{len(json_eval_res)} ########")
        print("acc_metric_1:",np.mean(acc_metric_1_list),"acc_metric_2:",np.mean(acc_metric_2_list),"acc_metric_3:",np.mean(acc_metric_3_list))

        sub_name_tna_name = video_gen_sub_item.split("v4_")[1]
        sub_name_tna_name = sub_name_tna_name.split("_qwen")[0]
        sub_name,tna_name = sub_name_tna_name[:-6],sub_name_tna_name[-1]
        print(sub_name,tna_name)
        if sub_name not in acc_res:
            acc_res[sub_name] = [""]*6
            acc_res[sub_name][int(tna_name)-1] = [np.mean(acc_metric_1_list),np.mean(acc_metric_2_list),np.mean(acc_metric_3_list)]
        else:
            acc_res[sub_name][int(tna_name)-1] = [np.mean(acc_metric_1_list),np.mean(acc_metric_2_list),np.mean(acc_metric_3_list)]
            
                
    print(acc_res)
    return  acc_res





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate metric.')
    
    parser.add_argument('--evaluated_model_name', type=str, required=False, default="CogVideoX1.5-5B",
                        help='The model name, e.g., CogVideoX1.5-5B')
    parser.add_argument('--answer_aes_save_path', type=str, required=False, default="./resource/generated_answers_aes",
                        help='the path to save the generated aes answers')
    parser.add_argument('--answer_save_path', type=str, required=False, default="./resource/generated_answers",
                        help='the path to save the generated answers')

    parser.add_argument('--evaluated_model_videos_path', type=str, required=False, default="./resource/generated_videos",
                        help='The path saves the generated videos')

    parser.add_argument('--metric_save_path', type=str, required=False, default="./resource/calculated_metric_res",
                        help='the path to save the calculated metric')
    
    args = parser.parse_args()

    model_name = args.evaluated_model_name
    benchmark_cal_acc(model_name, args.answer_save_path, args.answer_aes_save_path,  args.evaluated_model_videos_path, args.metric_save_path )
    acc_res = benchmark_cal_acc_final(args.metric_save_path)




