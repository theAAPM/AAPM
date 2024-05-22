import cv2
import subprocess
import random
import os
import numpy as np

def get_random_image_size(video_path):
    video = cv2.VideoCapture(video_path)
    _, frame = video.read()
    video_width = frame.shape[1]
    video_height = frame.shape[0]
    video.release()

    max_image_width = int(video_width / 2.5)
    max_image_height = int(video_height / 2.5)
    image_width = random.randint(int(max_image_width / 2.5), max_image_width)
    image_height = random.randint(int(max_image_height / 2.5), max_image_height)
    
    return (image_width, image_height)

def get_random_position(video_path, image_size):
    video = cv2.VideoCapture(video_path)
    _, frame = video.read()
    video_width = frame.shape[1]
    video_height = frame.shape[0]
    video.release()

    max_x = video_width - image_size[0]
    max_y = video_height - image_size[1]

    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    
    return (x, y)

def resize_image(image_path, output_path, target_size):

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    resized_image = cv2.resize(image, target_size)

    cv2.imwrite(output_path, resized_image)
    

def insert_image_in_video(video_path, image_path, output_path, start_time):
    target_size = get_random_image_size(video_path)
    resized_image_path = "resized_image.png"
    resize_image(image_path, resized_image_path, target_size)
    
    position = get_random_position(video_path, target_size)
    position_str = f"{position[0]}:{position[1]}"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    command = f'ffmpeg -i {video_path} -i {resized_image_path} -filter_complex "overlay={position_str}" -c:a copy -y -ss {start_time} {output_path}'
    
    subprocess.call(command, shell=True)
    
    os.remove(resized_image_path)



start_time = '00:00:00' 

video_file = open('./few_shot.txt', 'r') # filelist
video_paths = video_file.readlines()
random.shuffle(video_paths)
video_file.close()


image_folder = './partial_mask/graphic'  
image_files = [filename for filename in os.listdir(image_folder) if filename.endswith(('.png', '.jpg', '.jpeg'))]


for path in video_paths:
    video_path = path.strip().split('//') [-1]
    video_path = os.path.join('/DATA/PATH', video_path)
    video_name = path.strip().split('/') [-1]
    random_image = random.choice(image_files)
    image_path = os.path.join(image_folder, random_image)
    folder = random_image.split('.')[0]
    destination = os.path.join('./graphic_based', folder, video_name)
    insert_image_in_video(video_path, image_path, destination, start_time)

print('finished')



    
