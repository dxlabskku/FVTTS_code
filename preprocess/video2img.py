import os
import torch
import cv2, dlib, math
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--video_dir', default='data/trainval', type=str, help= 'video_dir')
parser.add_argument('--img_path', default='data/image', type=str, help= 'img_path')
parser.add_argument('--emb_path', default='data/img_emb', type=str, help= 'emb_path')
parser.add_argument('--landmark_path', default='shape_predictor_68_face_landmarks.dat', type=str, help= 'landmark_path')
args = parser.parse_args()

predictor = dlib.shape_predictor(args.landmakr_path)
detector = dlib.get_frontal_face_detector()

def get_angel(img):
    dets = detector(img, 1)
    thetas = []
    radians = []
    pos = []
    for k, d in enumerate(dets): 
        shape = predictor(img, d)    
        l = d.left()
        t = d.top()
        b = d.bottom()
        r = d.right()       
        num_of_points_out = 17
        num_of_points_in = shape.num_parts - num_of_points_out
        gx_out = 0; gy_out = 0; gx_in = 0; gy_in = 0
        for i in range(shape.num_parts): 
            shape_point = shape.part(i)
            if i < num_of_points_out:
                gx_out = gx_out + shape_point.x/num_of_points_out
                gy_out = gy_out + shape_point.y/num_of_points_out
            else:
                gx_in = gx_in + shape_point.x/num_of_points_in
                gy_in = gy_in + shape_point.y/num_of_points_in
        theta = math.asin(2*(gx_in-gx_out) / (d.right() - d.left()))
        radian = theta*180/math.pi
        thetas.append(theta)
        radians.append(radian)
        pos.append((l, t, b, r))
    return thetas, radians, pos


def train_transform():
    transform_list = [
        transforms.Resize(size=(128, 128)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

img_tf = train_transform()
resnet = InceptionResnetV1(pretrained='vggface2', classify=False).eval()

video_dir = args.video_dir
video_paths = os.listdir(video_dir)

img_path = args.img_path
emb_path = args.emb_path

if not os.path.exists(img_path):
    os.mkdir(img_path)
if not os.path.exists(emb_path):
    os.mkdir(emb_path)

for idx, name in enumerate(video_paths):
    video_list = [f for f in os.listdir(f"{video_dir}/{name}") if f.endswith('mp4')]
    for v in video_list:
        video_path = f"{video_dir}/{name}/{v}"
        video = cv2.VideoCapture(video_path) 
        fps = video.get(cv2.CAP_PROP_FPS)
        if video.isOpened():     
            sv_id = 0
            while sv_id < 5:    
                ret, img = video.read()    
                if ret:
                    if (int(video.get(1)) > 600) and (int(video.get(1)) % 30 == 0):
                        cv2.imwrite(f"{img_path}/{name}_{v[:-4]}_{sv_id}.jpg", img)
                        thetas, radians, pos = get_angel(img)
                        l, t, b, r = pos[0], pos[1], pos[2], pos[3]
                        crop_img = img[max(0, t-10):min(img.shape[0], b+10), max(0, l-10): min(img.shape[1], r+10)]
                        cv2.imwrite(f"{img_path}/{name}_{v[:-4]}_{sv_id}.jpg", crop_img)
                        img = Image.open(f"{img_path}/{name}_{v[:-4]}_{sv_id}.jpg")
                        img_trans = img_tf(img)
                        content = resnet(img_trans.unsqueeze(0)).T
                        torch.save(content, f'{emb_path}/{name}_{v[:-4]}_{sv_id}.pt')
                        sv_id += 1
                else: 
                    break        
        else:
            print("can't open video.")      
        video.release()
