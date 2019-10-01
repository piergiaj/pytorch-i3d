# Written by Boxiao Pan
# This script extracts rgb frames from a video
# It assumes a file hierarchy: video_root -> videos (not clustered by class)
# Sample uage: python extract_rgb_from_video --video_root $video_root --out_root $out_root

import cv2
import argparse
import os

if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser(description='rgb frame extractor')

    parser.add_argument('--video_root', type=str)
    parser.add_argument('--out_root', type=str)

    args = parser.parse_args()--

    all_vid_rel = os.listdir(args.video_root)
    for i in range(len(all_vid_rel)):
        vid_abs = os.path.join(args.video_root, all_vid_rel[i])
        vidcap = cv2.VideoCapture(vid_abs)
        
        success, image = vidcap.read()
        count = 0
        while success:
            if not os.path.exists(os.path.join(args.out_root, all_vid_rel[i])):
                os.makedirs(os.path.join(args.out_root, all_vid_rel[i]))
            cv2.imwrite(os.path.join(args.out_root, all_vid_rel[i], 'rgb_{:04d}.jpg'.format(count)), image)
            success, image = vidcap.read()
            count += 1
        
        print('finished {} / {} videos. This current video has {} frames.'.format(i+1, len(all_vid_rel), count))
