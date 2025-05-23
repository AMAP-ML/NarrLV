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

class Wan_GEN:
    def __init__(self, model_id):
        self.model_id = model_id   # Wan2.1-T2V-14B-Diffusers
        self.model_name = "wan14b"
        self.vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        self.negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

        self.flow_shift = 5.0 # 5.0 for 720P, 3.0 for 480P
        self.scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
        self.pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        self.pipe.scheduler = scheduler
        # device = torch.device("cuda:2")
        self.pipe.to("cuda")
    
    def generate(self, prompt, output_path):
        output = self.pipe(
                    prompt=prompt,
                    negative_prompt=self.negative_prompt,
                    height=720,
                    width=1280,
                    num_frames=81,
                    guidance_scale=5.0,
                    ).frames[0]

        export_to_video(output, output_path, fps=16)



class CogVideo_GEN:
    def __init__(self, model_id):
        self.model_id = model_id   # "CogVideoX1.5-5B"
        self.model_name = "CogVideoX1.5-5B"
        self.pipe = pipe = CogVideoXPipeline.from_pretrained(
                                                model_id,
                                                torch_dtype=torch.bfloat16
                                            )
        # device = torch.device("cuda:2")
        self.pipe.to("cuda")
        self.pipe.enable_sequential_cpu_offload()
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()
    
    def generate(self, prompt, output_path):
        output = self.pipe(
                    prompt=prompt,
                    num_videos_per_prompt=1,
                    num_inference_steps=50,
                    num_frames=81,
                    guidance_scale=6,
                    generator=torch.Generator(device="cuda").manual_seed(42),
                ).frames[0]

        export_to_video(output, output_path, fps=16)