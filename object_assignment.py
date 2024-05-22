import cv2
import subprocess
import random
import os
import numpy as np

def get_random_image_size(video_path, folder):
    # 获取视频大小
    video = cv2.VideoCapture(video_path)
    _, frame = video.read()
    video_width = frame.shape[1]
    video_height = frame.shape[0]
    video.release()

    max_image_width = int(video_width / 1.5)
    max_image_height = int(video_height / 1.5)
    image_width = random.randint(int(max_image_width / 1.5), max_image_width)
    image_height = int((image_width/video_width) * video_height)        
    return (image_width, image_height)

def get_random_position(video_path, image_size, folder):
    video = cv2.VideoCapture(video_path)
    _, frame = video.read()
    video_width = frame.shape[1]
    video_height = frame.shape[0]
    video.release()
    
    max_x = video_width - int(image_size[0]*0.8)
    max_y = video_height - int(image_size[1]*0.8)

    x = random.randint(int(-0.2*image_size[0]), max_x)
    y = random.randint(int(0.3*max_y), max_y)
    
    return (x, y)

def resize_image(image_path, output_path, target_size):

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    resized_image = cv2.resize(image, target_size)
    
    cv2.imwrite(output_path, resized_image)
    

def insert_image_in_video(video_path, image_path, output_path, start_time, folder):

    target_size = get_random_image_size(video_path, folder)
    resized_image_path = "resized_image.png"
    resize_image(image_path, resized_image_path, target_size)
    
    position = get_random_position(video_path, target_size, folder)
    position_str = f"{position[0]}:{position[1]}"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    command = f'ffmpeg -i {video_path} -i {resized_image_path} -filter_complex "overlay={position_str}" -c:a copy -y -ss {start_time} {output_path}'
    

    subprocess.call(command, shell=True)
    os.remove(resized_image_path)



start_time = '00:00:00' 

video_file = open('./few_shot.txt', 'r')
video_paths = video_file.readlines()
random.shuffle(video_paths)
video_file.close()


image_folder = './partial_mask/object'

subfolders = [os.path.join(image_folder, name) for name in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, name))]

for path in video_paths:
    video_path = path.strip().split('//')[-1]
    video_path = os.path.join('/DATA/PATH', video_path)
    video_name = path.strip().split('/')[-1]
    
    random_subfolder = random.choice(subfolders)
    random_image = random.choice(os.listdir(random_subfolder))
    image_path = os.path.join(random_subfolder, random_image)
    
    folder = os.path.basename(random_subfolder)
    
    destination = os.path.join('./object_based', folder, video_name)
    

    insert_image_in_video(video_path, image_path, destination, start_time, folder)

print('finished')