import argparse
import os
import io
from pathlib import Path
import uuid
import sys
import yaml
import traceback
from deoldify.visualize import *

with open('./config.yaml', 'r') as fd:
    opts = yaml.safe_load(fd)

sys.path.insert(0, './white_box_cartoonizer/')

import cv2
import flask
from PIL import Image
import numpy as np
import skvideo.io

from cartoonize import WB_Cartoonize
from youtube import download
import Algorithmia

## Init Cartoonizer and load its weights 
wb_cartoonizer = WB_Cartoonize(os.path.abspath("white_box_cartoonizer/saved_models/"), opts['gpu'])

def convert_bytes_to_image(img_bytes):
    """Convert bytes to numpy array

    Args:
        img_bytes (bytes): Image bytes read from flask.

    Returns:
        [numpy array]: Image numpy array
    """
    
    pil_image = Image.open(io.BytesIO(img_bytes))
    if pil_image.mode=="RGBA":
        image = Image.new("RGB", pil_image.size, (255,255,255))
        image.paste(pil_image, mask=pil_image.split()[3])
    else:
        image = pil_image.convert('RGB')
    
    image = np.array(image)
    
    return image

def colorize(colorizer, render_factor, in_path, out_path):
    path = Path(in_path)
    results_dir = Path(out_path)
    result = colorizer.get_transformed_image(
        path, render_factor, post_process=True, watermarked=False
    )
    result_path = colorizer._save_result_image(path, result, results_dir=results_dir)
    result.close()
    return result_path

def cartoonize(convert_type, path, width, batch_size):
        path = os.path.join("source", path)
        try:
            if convert_type == 'image':
                image = cv2.imread(path, 1)

                filename = path.split("/")[-1]
                filename = filename.split(".")[0]
                
                cartoon_image = wb_cartoonizer.infer(image)
                
                cartoonized_img_name = os.path.join(opts["cartoonized-image"], filename + ".jpg")
                cv2.imwrite(cartoonized_img_name, cv2.cvtColor(cartoon_image, cv2.COLOR_RGB2BGR))

                return cartoonized_img_name

            if convert_type == 'video':
                filename = path.split("/")[-1]
                filename = filename.split(".")[0]

                modified_video_path = os.path.join(opts["cartoonized-video"], filename + "_modified.mp4")
                
                ## Fetch Metadata and set frame rate
                file_metadata = skvideo.io.ffprobe(path)
                original_frame_rate = None
                if 'video' in file_metadata:
                    if '@r_frame_rate' in file_metadata['video']:
                        original_frame_rate = file_metadata['video']['@r_frame_rate']

                if opts['original_frame_rate']:
                    output_frame_rate = original_frame_rate
                else:
                    output_frame_rate = opts['output_frame_rate']    

                output_frame_rate_number = int(output_frame_rate.split('/')[0])

                #change the size if you want higher resolution :
                ############################
                # Recommnded width_resize  #
                ############################
                #width_resize = 1920 for 1080p: 1920x1080.
                #width_resize = 1280 for 720p: 1280x720.
                #width_resize = 854 for 480p: 854x480.
                #width_resize = 640 for 360p: 640x360.
                #width_resize = 426 for 240p: 426x240.
                width_resize=width

                if opts['original_resolution']:
                    os.system("ffmpeg -hide_banner -loglevel warning -ss 0 -vsync 2 -c:v h264_cuvid -i '{}' -filter:v scale=-1:-2 -r {} -c:a copy '{}'".format(os.path.abspath(path), output_frame_rate_number, os.path.abspath(modified_video_path)))
                else:
                    os.system("ffmpeg -hide_banner -loglevel warning -ss 0 -vsync 2 -c:v h264_cuvid -i '{}' -filter:v scale={}:-2 -r {} -c:a copy '{}'".format(os.path.abspath(path), width_resize, output_frame_rate_number, os.path.abspath(modified_video_path)))
                
                audio_file_path = os.path.join(opts["cartoonized-video"], filename + "_audio_modified.mp4")
                os.system("ffmpeg -hide_banner -loglevel warning -c:v h264_cuvid -i '{}' -map 0:1 -vn -acodec copy -strict -2  '{}'".format(os.path.abspath(modified_video_path), os.path.abspath(audio_file_path)))

                cartoon_video_path = wb_cartoonizer.process_video(modified_video_path, output_frame_rate, batch_size)
                
                ## Add audio to the cartoonized video
                final_cartoon_video_path = os.path.join(opts["cartoonized-video"], filename.split(".")[0] + "_cartoon_audio.mp4")
                os.system("ffmpeg -hide_banner -loglevel warning -c:v h264_cuvid -i '{}' -i '{}' -codec copy -shortest '{}'".format(os.path.abspath(cartoon_video_path), os.path.abspath(audio_file_path), os.path.abspath(final_cartoon_video_path)))

                return final_cartoon_video_path
        
        except Exception:
            print(traceback.print_exc())
            raise RuntimeError("Error cannot run convert video")

def cartoonize_youtube(code, batch_size=10):
        try:
            video_path, audio_path = download(code)
            filename = video_path.split("/")[-1]
            filename = filename.split(".")[0]

            ## Fetch Metadata and set frame rate
            file_metadata = skvideo.io.ffprobe(video_path)
            original_frame_rate = None
            if 'video' in file_metadata:
                if '@r_frame_rate' in file_metadata['video']:
                    original_frame_rate = file_metadata['video']['@r_frame_rate']

            cartoon_video_path = wb_cartoonizer.process_video(video_path, original_frame_rate, batch_size)
            
            ## Add audio to the cartoonized video
            final_cartoon_video_path = os.path.join(opts["cartoonized-video"], filename.split(".")[0] + "_cartoonized.mp4")
            os.system("ffmpeg -hide_banner -loglevel warning -c:v h264_cuvid -i '{}' -i '{}' -codec copy -shortest '{}'".format(os.path.abspath(cartoon_video_path), os.path.abspath(audio_path), os.path.abspath(final_cartoon_video_path)))

            return final_cartoon_video_path
        
        except Exception:
            print(traceback.print_exc())
            raise RuntimeError("Error cannot run convert video")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default="video", choices=["image", "video"])
    parser.add_argument('--path', type=str)
    parser.add_argument('--width', type=int, default=1920, choices=[426, 640, 854, 1280, 1920])
    parser.add_argument('--size', type=int, default=10)
    args = parser.parse_args()

    output_path = cartoonize(args.type, args.path, args.width, args.size)
    print(f"Output path: {output_path}")
