import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue
import subprocess
from model.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")

def transferAudio(sourceVideo, targetVideo):
    import shutil
    import moviepy.editor
    tempAudioFileName = "./temp/audio.mkv"

    if os.path.isdir("temp"):
        shutil.rmtree("temp")
    os.makedirs("temp")
    os.system(f'ffmpeg -y -i "{sourceVideo}" -c:a copy -vn {tempAudioFileName}')

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    os.system(f'ffmpeg -y -i "{targetNoAudio}" -i {tempAudioFileName} -c copy "{targetVideo}"')

    if os.path.getsize(targetVideo) == 0:
        tempAudioFileName = "./temp/audio.m4a"
        os.system(f'ffmpeg -y -i "{sourceVideo}" -c:a aac -b:a 160k -vn {tempAudioFileName}')
        os.system(f'ffmpeg -y -i "{targetNoAudio}" -i {tempAudioFileName} -c copy "{targetVideo}"')
        if os.path.getsize(targetVideo) == 0:
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    shutil.rmtree("temp")

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--video', type=str, default=None)
parser.add_argument('--output', type=str, default=None)
parser.add_argument('--img', type=str, default=None)
parser.add_argument('--montage', action='store_true')
parser.add_argument('--model', type=str, default='train_log')
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--UHD', action='store_true')
parser.add_argument('--scale', type=float, default=1.0)
parser.add_argument('--skip', action='store_true')
parser.add_argument('--fps', type=int, default=None)
parser.add_argument('--png', action='store_true')
parser.add_argument('--ext', type=str, default='mp4')
parser.add_argument('--exp', type=int, default=1)
parser.add_argument('--multi', type=int, default=2)
parser.add_argument('--qp', type=str, default=18)

args = parser.parse_args()
if args.exp != 1:
    args.multi = 2 ** args.exp
assert args.video or args.img
if args.UHD and args.scale == 1.0:
    args.scale = 0.5
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
if args.img:
    args.png = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if args.fp16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

from train_log.RIFE_HDv3 import Model
model = Model()
if not hasattr(model, 'version'):
    model.version = 0
model.load_model(args.model, -1)
model.eval()
model.device()

if args.video:
    videoCapture = cv2.VideoCapture(args.video)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    videoCapture.release()
    fpsNotAssigned = args.fps is None
    if fpsNotAssigned:
        args.fps = fps * args.multi
    videogen = skvideo.io.vreader(args.video)
    lastframe = next(videogen)
    video_path_wo_ext, ext = os.path.splitext(args.video)
else:
    videogen = sorted([f for f in os.listdir(args.img) if 'png' in f], key=lambda x: int(x[:-4]))
    tot_frame = len(videogen)
    lastframe = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
    videogen = videogen[1:]

h, w, _ = lastframe.shape

vid_out_name = args.output if args.output else f'{video_path_wo_ext}_{args.multi}X_{int(np.round(args.fps))}fps.{args.ext}'

ffmpeg_cmd = [
    'ffmpeg', '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'rgb24',
    '-s', f'{w}x{h}',
    '-r', str(args.fps),
    '-i', '-',
    '-an',
    '-vcodec', 'hevc_nvenc',
    '-qp', args.qp,
    '-pix_fmt', 'yuv420p',
    vid_out_name
]
ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

def clear_write_buffer(write_buffer):
    while True:
        item = write_buffer.get()
        if item is None:
            break
        ffmpeg_proc.stdin.write(item.astype(np.uint8).tobytes())

def build_read_buffer(read_buffer, videogen):
    try:
        for frame in videogen:
            if args.img:
                frame = cv2.imread(os.path.join(args.img, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
            read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)

def make_inference(I0, I1, n):
    if model.version >= 3.9:
        return [model.inference(I0, I1, (i+1)/(n+1), args.scale) for i in range(n)]
    else:
        middle = model.inference(I0, I1, args.scale)
        if n == 1:
            return [middle]
        first_half = make_inference(I0, middle, n=n//2)
        second_half = make_inference(middle, I1, n=n//2)
        return [*first_half, middle, *second_half] if n%2 else [*first_half, *second_half]

def pad_image(img):
    return F.pad(img, padding).half() if args.fp16 else F.pad(img, padding)

tmp = max(128, int(128 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)
pbar = tqdm(total=tot_frame)

write_buffer = Queue(maxsize=500)
read_buffer = Queue(maxsize=500)
_thread.start_new_thread(build_read_buffer, (read_buffer, videogen))
_thread.start_new_thread(clear_write_buffer, (write_buffer,))

I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device).unsqueeze(0).float() / 255.
I1 = pad_image(I1)
temp = None

while True:
    frame = temp if temp is not None else read_buffer.get()
    if frame is None:
        break
    temp = None

    I0 = I1
    I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device).unsqueeze(0).float() / 255.
    I1 = pad_image(I1)
    I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
    I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
    ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

    if ssim > 0.996:
        frame = read_buffer.get()
        if frame is None:
            frame = lastframe
            break
        else:
            temp = frame
        I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device).unsqueeze(0).float() / 255.
        I1 = pad_image(I1)
        I1 = model.inference(I0, I1, scale=args.scale)
        frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

    output = make_inference(I0, I1, args.multi - 1) if ssim >= 0.2 else [I0] * (args.multi - 1)

    write_buffer.put(lastframe)
    for mid in output:
        mid = (mid[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
        write_buffer.put(mid)
    pbar.update(1)
    lastframe = frame

write_buffer.put(lastframe)
write_buffer.put(None)

import time
while not write_buffer.empty():
    time.sleep(0.1)

pbar.close()

ffmpeg_proc.stdin.close()
ffmpeg_proc.wait()

if fpsNotAssigned and args.video:
    try:
        transferAudio(args.video, vid_out_name)
    except:
        print("Audio transfer failed. Interpolated video will have no audio")
