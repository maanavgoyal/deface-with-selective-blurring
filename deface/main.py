#!/usr/bin/env python3

import argparse
import json
import mimetypes
import os
from typing import Dict, Tuple, List, Optional

import tqdm
import skimage.draw
import numpy as np
import imageio
import imageio.v2 as iio
import imageio.plugins.ffmpeg
import cv2
import glob
# import torch
from deepface import DeepFace

from deface import __version__
from deface.centerface import CenterFace

# New imports
from scipy.spatial.distance import cosine

from shapely.geometry import box

# Global variables for face tracking
selected_face = None
face_tracker = None


# Add this function to calculate IoU
def calculate_iou(box1, box2):
    b1 = box(*box1)
    b2 = box(*box2)
    intersection = b1.intersection(b2).area
    union = b1.union(b2).area
    return intersection / union if union > 0 else 0

def select_face(event, x, y, flags, param):
    global selected_face
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_face = (x, y)
        print(f"Selected face at coordinates: ({x}, {y})")

def init_face_selection(frame):
    global selected_face
    cv2.namedWindow("Select Face")
    cv2.setMouseCallback("Select Face", select_face)
    
    while True:
        cv2.imshow("Select Face", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or selected_face is not None:
            break
    
    cv2.destroyAllWindows()
    return selected_face

def init_face_tracker(frame, bbox):
    x, y, w, h = bbox
    center_x = x + w / 2
    center_y = y + h / 2
    new_w = w * 2  # Double the width
    new_h = h * 1.5  # 1.5 times taller
    new_x = center_x - new_w / 2  # Adjust x to keep the same center
    new_y = center_y - new_h / 2  # Adjust y to keep the same center
    
    # Ensure the new bounding box stays within the frame
    new_x = max(0, new_x)
    new_y = max(0, new_y)
    new_w = min(new_w, frame.shape[1] - new_x)
    new_h = min(new_h, frame.shape[0] - new_y)
    
    new_bbox = (int(new_x), int(new_y), int(new_w), int(new_h))
    
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, new_bbox)
    return tracker, new_bbox

def update_face_tracker(frame, tracker, prev_bbox):
    success, bbox = tracker.update(frame)
    if success:
        bbox = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
        new_w, new_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        prev_w, prev_h = prev_bbox[2] - prev_bbox[0], prev_bbox[3] - prev_bbox[1]
        
        max_change = 0.15
        
        # Check if prev_w or prev_h is zero to avoid division by zero
        if prev_w > 0 and prev_h > 0:
            w_change = abs(new_w - prev_w) / prev_w
            h_change = abs(new_h - prev_h) / prev_h
            
            if w_change > max_change or h_change > max_change:
                center_x, center_y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                bbox = (
                    center_x - prev_w / 2,
                    center_y - prev_h / 2,
                    center_x + prev_w / 2,
                    center_y + prev_h / 2
                )
        else:
            # If prev_w or prev_h is zero, use the new bbox as is
            pass

        return bbox
    return None

def get_face_embedding(face_image):
    try:
        result = DeepFace.represent(face_image, model_name="VGG-Face", enforce_detection=False)
        return np.array(result[0]['embedding'])
    except Exception as e:
        print(f"Error getting face embedding: {str(e)}")
        return None

def get_face_embeddings(image_directory):
    embeddings = []
    for image_path in glob.glob(os.path.join(image_directory, '*')):
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            embedding = get_face_embedding(image)
            if embedding is not None:
                embeddings.append(embedding)
    return embeddings

def is_same_person(face, target_embeddings, threshold=0.2):
    face_embedding = get_face_embedding(face)
    
    if face_embedding is None:
        return False
    
    if not target_embeddings:
        return False
    
    total_similarity = 0
    for target_embedding in target_embeddings:
        cosine_similarity = 1 - cosine(face_embedding, target_embedding)
        total_similarity += cosine_similarity
    
    average_similarity = total_similarity / len(target_embeddings)
    
    return average_similarity > threshold

def find_person_in_frame(frame, target_embeddings, centerface, threshold):
    dets, _ = centerface(frame, threshold=threshold)
    
    for det in dets:
        x1, y1, x2, y2 = map(int, det[:4])
        face_image = frame[y1:y2, x1:x2]
        if is_same_person(face_image, target_embeddings):
            return (x1, y1, x2-x1, y2-y1)  # Return as (x, y, w, h)
    
    return None

def detect_scene_change(prev_frame, curr_frame, threshold=0.1):
    if prev_frame is None or curr_frame is None:
        return False
    
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
    
    # Compute histograms
    prev_hist = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
    curr_hist = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])
    
    # Normalize histograms
    prev_hist = cv2.normalize(prev_hist, prev_hist).flatten()
    curr_hist = cv2.normalize(curr_hist, curr_hist).flatten()
    
    # Compare histograms
    hist_diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)
    # print(hist_diff)
    
    return hist_diff > threshold

def scale_bb(x1, y1, x2, y2, mask_scale=1.0):
    s = mask_scale - 1.0
    h, w = y2 - y1, x2 - x1
    y1 -= h * s
    y2 += h * s
    x1 -= w * s
    x2 += w * s
    return np.round([x1, y1, x2, y2]).astype(int)


def draw_det(
        frame, score, det_idx, x1, y1, x2, y2,
        replacewith: str = 'blur',
        ellipse: bool = True,
        draw_scores: bool = False,
        ovcolor: Tuple[int] = (0, 0, 0),
        replaceimg = None,
        mosaicsize: int = 20
):
    if replacewith == 'solid':
        cv2.rectangle(frame, (x1, y1), (x2, y2), ovcolor, -1)
    elif replacewith == 'blur':
        bf = 2  # blur factor (number of pixels in each dimension that the face will be reduced to)
        blurred_box =  cv2.blur(
            frame[y1:y2, x1:x2],
            (abs(x2 - x1) // bf, abs(y2 - y1) // bf)
        )
        if ellipse:
            roibox = frame[y1:y2, x1:x2]
            # Get y and x coordinate lists of the "bounding ellipse"
            ey, ex = skimage.draw.ellipse((y2 - y1) // 2, (x2 - x1) // 2, (y2 - y1) // 2, (x2 - x1) // 2)
            roibox[ey, ex] = blurred_box[ey, ex]
            frame[y1:y2, x1:x2] = roibox
        else:
            frame[y1:y2, x1:x2] = blurred_box
    elif replacewith == 'img':
        target_size = (x2 - x1, y2 - y1)
        resized_replaceimg = cv2.resize(replaceimg, target_size)
        if replaceimg.shape[2] == 3:  # RGB
            frame[y1:y2, x1:x2] = resized_replaceimg
        elif replaceimg.shape[2] == 4:  # RGBA
            frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - resized_replaceimg[:, :, 3:] / 255) + resized_replaceimg[:, :, :3] * (resized_replaceimg[:, :, 3:] / 255)
    elif replacewith == 'mosaic':
        for y in range(y1, y2, mosaicsize):
            for x in range(x1, x2, mosaicsize):
                pt1 = (x, y)
                pt2 = (min(x2, x + mosaicsize - 1), min(y2, y + mosaicsize - 1))
                color = (int(frame[y, x][0]), int(frame[y, x][1]), int(frame[y, x][2]))
                cv2.rectangle(frame, pt1, pt2, color, -1)
    elif replacewith == 'none':
        pass
    if draw_scores:
        cv2.putText(
            frame, f'{score:.2f}', (x1 + 0, y1 - 20),
            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0)
        )


def anonymize_frame(
        dets, frame, mask_scale,
        replacewith, ellipse, draw_scores, replaceimg, mosaicsize
):
    for i, det in enumerate(dets):
        boxes, score = det[:4], det[4]
        x1, y1, x2, y2 = boxes.astype(int)
        x1, y1, x2, y2 = scale_bb(x1, y1, x2, y2, mask_scale)
        # Clip bb coordinates to valid frame region
        y1, y2 = max(0, y1), min(frame.shape[0] - 1, y2)
        x1, x2 = max(0, x1), min(frame.shape[1] - 1, x2)
        draw_det(
            frame, score, i, x1, y1, x2, y2,
            replacewith=replacewith,
            ellipse=ellipse,
            draw_scores=draw_scores,
            replaceimg=replaceimg,
            mosaicsize=mosaicsize
        )


def cam_read_iter(reader):
    while True:
        yield reader.get_next_data()


def video_detect(
        ipath: str,
        opath: str,
        centerface: CenterFace,
        threshold: float,
        enable_preview: bool,
        cam: bool,
        nested: bool,
        replacewith: str,
        mask_scale: float,
        ellipse: bool,
        draw_scores: bool,
        ffmpeg_config: Dict[str, str],
        replaceimg = None,
        keep_audio: bool = False,
        mosaicsize: int = 20,
        target_embeddings = None,
):
    global face_tracker

    try:
        if 'fps' in ffmpeg_config:
            reader: imageio.plugins.ffmpeg.FfmpegFormat.Reader = imageio.get_reader(ipath, fps=ffmpeg_config['fps'])
        else:
            reader: imageio.plugins.ffmpeg.FfmpegFormat.Reader = imageio.get_reader(ipath)

        meta = reader.get_meta_data()
        _ = meta['size']
    except:
        if cam:
            print(f'Could not find video device {ipath}. Please set a valid input.')
        else:
            print(f'Could not open file {ipath} as a video file with imageio. Skipping file...')
        return

    if cam:
        nframes = None
        read_iter = cam_read_iter(reader)
    else:
        read_iter = reader.iter_data()
        nframes = reader.count_frames()

    if nested:
        bar = tqdm.tqdm(dynamic_ncols=True, total=nframes, position=1, leave=True)
    else:
        bar = tqdm.tqdm(dynamic_ncols=True, total=nframes)

    if opath is not None:
        _ffmpeg_config = ffmpeg_config.copy()
        _ffmpeg_config.setdefault('fps', meta['fps'])
        if keep_audio and meta.get('audio_codec'):
            _ffmpeg_config.setdefault('audio_path', ipath)
            _ffmpeg_config.setdefault('audio_codec', 'copy')
        writer: imageio.plugins.ffmpeg.FfmpegFormat.Writer = imageio.get_writer(
            opath, format='FFMPEG', mode='I', **_ffmpeg_config
        )

    face_tracker = None
    target_person_found = False
    iou_threshold = 0.2
    prev_frame = None
    scene_change_threshold = 0.2

    for frame in read_iter:

        # Ensure frame is in the correct format (numpy array)
        if isinstance(frame, np.ndarray):
            current_frame = frame
        else:
            current_frame = np.array(frame)

        if prev_frame is not None:
            if detect_scene_change(prev_frame, current_frame, scene_change_threshold):
                print("Scene change detected. Reinitializing face tracking.")
                face_tracker = None
                target_person_found = False

        if not target_person_found:
            person_bbox = find_person_in_frame(frame, target_embeddings, centerface, threshold)
            if person_bbox is not None:
                # face_tracker = cv2.TrackerCSRT_create()
                # face_tracker.init(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), person_bbox)
                face_tracker, prev_bbox = init_face_tracker(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), person_bbox)
                target_person_found = True
                print("Target person found and tracking started.")

        # Perform network inference, get bb dets but discard landmark predictions
        dets, _ = centerface(frame, threshold=threshold)
        flag = True

        if face_tracker is not None:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            tracked_bbox = update_face_tracker(bgr_frame, face_tracker, prev_bbox)
            if tracked_bbox is not None:
                flag = True
                x1, y1, x2, y2 = map(int, tracked_bbox)
                # x2, y2 = x1 + w, y1 + h
                # tracked_box = (x1, y1, x2, y2)
                
                # Calculate IoU for each detection with the tracked box
                ious = [calculate_iou(tracked_bbox, (det[0], det[1], det[2], det[3])) for det in dets]
                
                # Keep detections that are within the tracked region
                # dets_in_tracked = [det for det in dets if x1 < det[0] < x2 and y1 < det[1] < y2]
                dets_in_tracked = [det for det, iou in zip(dets, ious) if iou > iou_threshold]

                # If there are exactly two detections in the tracked region, blur them
                if len(dets_in_tracked) < 2:
                    # dets = [det for det in dets if not (x1 < det[0] < x2 and y1 < det[1] < y2)]
                   dets = [det for det, iou in zip(dets, ious) if iou < iou_threshold]

                # Draw a green box around the tracked face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw a red rectangle around the whole frame
                height, width = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (255, 0, 0), 2)
                
                prev_bbox = tracked_bbox
            else:
                # Tracker lost the face, try to find it again
                if flag==True:
                    flag=False
                    # print("Face tracking lost, attempting to recover...")
                for det in dets:
                    x1, y1, x2, y2 = map(int, det[:4])
                    face_image = frame[y1:y2, x1:x2]
                    if is_same_person(face_image, target_embeddings):
                        face_tracker, prev_bbox = init_face_tracker(bgr_frame, (x1, y1, x2-x1, y2-y1))
                        dets = [d for d in dets if not np.array_equal(d, det)]
                        break


        anonymize_frame(
            dets, frame, mask_scale=mask_scale,
            replacewith=replacewith, ellipse=ellipse, draw_scores=draw_scores,
            replaceimg=replaceimg, mosaicsize=mosaicsize
        )

        if opath is not None:
            writer.append_data(frame)

        if enable_preview:
            cv2.imshow('Preview of anonymization results (quit by pressing Q or Escape)', frame[:, :, ::-1])  # RGB -> BGR
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:  # 27 is the escape key code
                cv2.destroyAllWindows()
                break

        prev_frame = current_frame.copy()
        bar.update()

    reader.close()
    if opath is not None:
        writer.close()
    bar.close()


def image_detect(
        ipath: str,
        opath: str,
        centerface: CenterFace,
        threshold: float,
        replacewith: str,
        mask_scale: float,
        ellipse: bool,
        draw_scores: bool,
        enable_preview: bool,
        keep_metadata: bool,
        replaceimg = None,
        mosaicsize: int = 20,
):
    frame = iio.imread(ipath)

    if keep_metadata:
        # Source image EXIF metadata retrieval via imageio V3 lib
        metadata = imageio.v3.immeta(ipath)
        exif_dict = metadata.get("exif", None)

    # Perform network inference, get bb dets but discard landmark predictions
    dets, _ = centerface(frame, threshold=threshold)

    anonymize_frame(
        dets, frame, mask_scale=mask_scale,
        replacewith=replacewith, ellipse=ellipse, draw_scores=draw_scores,
        replaceimg=replaceimg, mosaicsize=mosaicsize
    )

    if enable_preview:
        cv2.imshow('Preview of anonymization results (quit by pressing Q or Escape)', frame[:, :, ::-1])  # RGB -> RGB
        if cv2.waitKey(0) & 0xFF in [ord('q'), 27]:  # 27 is the escape key code
            cv2.destroyAllWindows()

    imageio.imsave(opath, frame)

    if keep_metadata:
        # Save image with EXIF metadata
        imageio.imsave(opath, frame, exif=exif_dict)

    # print(f'Output saved to {opath}')


def get_file_type(path):
    if path.startswith('<video'):
        return 'cam'
    if not os.path.isfile(path):
        return 'notfound'
    mime = mimetypes.guess_type(path)[0]
    if mime is None:
        return None
    if mime.startswith('video'):
        return 'video'
    if mime.startswith('image'):
        return 'image'
    return mime


def get_anonymized_image(frame,
                         threshold: float,
                         replacewith: str,
                         mask_scale: float,
                         ellipse: bool,
                         draw_scores: bool,
                         replaceimg = None
                         ):
    """
    Method for getting an anonymized image without CLI
    returns frame
    """

    centerface = CenterFace(in_shape=None, backend='auto')
    dets, _ = centerface(frame, threshold=threshold)

    anonymize_frame(
        dets, frame, mask_scale=mask_scale,
        replacewith=replacewith, ellipse=ellipse, draw_scores=draw_scores,
        replaceimg=replaceimg
    )

    return frame


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Video anonymization by face detection', add_help=False)
    parser.add_argument(
        'input', nargs='*',
        help=f'File path(s) or camera device name. It is possible to pass multiple paths by separating them by spaces or by using shell expansion (e.g. `$ deface vids/*.mp4`). Alternatively, you can pass a directory as an input, in which case all files in the directory will be used as inputs. If a camera is installed, a live webcam demo can be started by running `$ deface cam` (which is a shortcut for `$ deface -p \'<video0>\'`.')
    parser.add_argument(
        '--output', '-o', default=None, metavar='O',
        help='Output file name. Defaults to input path + postfix "_anonymized".')
    # parser.add_argument(
    #     '--target-person', required=True,
    #     help='Path to the image of the target person to be tracked.'
    # )
    parser.add_argument(
        '--target-person-dir', required=True,
        help='Path to the directory containing images of the target person to be tracked.'
    )
    parser.add_argument(
        '--thresh', '-t', default=0.3, type=float, metavar='T',
        help='Detection threshold (tune this to trade off between false positive and false negative rate). Default: 0.2.')
    parser.add_argument(
        '--scale', '-s', default=None, metavar='WxH',
        help='Downscale images for network inference to this size (format: WxH, example: --scale 640x360).')
    parser.add_argument(
        '--preview', '-p', default=False, action='store_true',
        help='Enable live preview GUI (can decrease performance).')
    parser.add_argument(
        '--boxes', default=True, action='store_true',
        help='Use boxes instead of ellipse masks.')
    parser.add_argument(
        '--draw-scores', default=False, action='store_true',
        help='Draw detection scores onto outputs.')
    parser.add_argument(
        '--mask-scale', default=1.3, type=float, metavar='M',
        help='Scale factor for face masks, to make sure that masks cover the complete face. Default: 1.3.')
    parser.add_argument(
        '--replacewith', default='blur', choices=['blur', 'solid', 'none', 'img', 'mosaic'],
        help='Anonymization filter mode for face regions. "blur" applies a strong gaussian blurring, "solid" draws a solid black box, "none" does leaves the input unchanged, "img" replaces the face with a custom image and "mosaic" replaces the face with mosaic. Default: "blur".')
    parser.add_argument(
        '--replaceimg', default='replace_img.png',
        help='Anonymization image for face regions. Requires --replacewith img option.')
    parser.add_argument(
        '--mosaicsize', default=20, type=int, metavar='width',
        help='Setting the mosaic size. Requires --replacewith mosaic option. Default: 20.')
    parser.add_argument(
        '--keep-audio', '-k', default=False, action='store_true',
        help='Keep audio from video source file and copy it over to the output (only applies to videos).')
    parser.add_argument(
        '--ffmpeg-config', default={"codec": "libx264"}, type=json.loads,
        help='FFMPEG config arguments for encoding output videos. This argument is expected in JSON notation. For a list of possible options, refer to the ffmpeg-imageio docs. Default: \'{"codec": "libx264"}\'.'
    )  # See https://imageio.readthedocs.io/en/stable/format_ffmpeg.html#parameters-for-saving
    parser.add_argument(
        '--backend', default='auto', choices=['auto', 'onnxrt', 'opencv'],
        help='Backend for ONNX model execution. Default: "auto" (prefer onnxrt if available).')
    parser.add_argument(
        '--execution-provider', '--ep', default=None, metavar='EP',
        help='Override onnxrt execution provider (see https://onnxruntime.ai/docs/execution-providers/). If not specified, the presumably fastest available one will be automatically selected. Only used if backend is onnxrt.')
    parser.add_argument(
        '--version', action='version', version=__version__,
        help='Print version number and exit.')
    parser.add_argument(
        '--keep-metadata', '-m', default=False, action='store_true',
        help='Keep metadata of the original image. Default : False.')
    parser.add_argument('--help', '-h', action='help', help='Show this help message and exit.')

    args = parser.parse_args()

    if len(args.input) == 0:
        parser.print_help()
        print('\nPlease supply at least one input path.')
        exit(1)

    if args.input == ['cam']:  # Shortcut for webcam demo with live preview
        args.input = ['<video0>']
        args.preview = True

    return args


def main():
    args = parse_cli_args()
    ipaths = []

    # add files in folders
    for path in args.input:
        if os.path.isdir(path):
            for file in os.listdir(path):
                ipaths.append(os.path.join(path,file))
        else:
            # Either a path to a regular file, the special 'cam' shortcut
            # or an invalid path. The latter two cases are handled below.
            ipaths.append(path)

    # target_person_image = cv2.imread(args.target_person)
    # if target_person_image is None:
    #     print(f"Error: Could not load target person image from {args.target_person}")
    #     exit(1)
    # target_person_image = cv2.cvtColor(target_person_image, cv2.COLOR_BGR2RGB)

    target_embeddings = get_face_embeddings(args.target_person_dir)
    if not target_embeddings:
        print(f"Error: Could not load any valid target person images from {args.target_person_dir}")
        exit(1)

    base_opath = args.output
    replacewith = args.replacewith
    enable_preview = args.preview
    draw_scores = args.draw_scores
    threshold = args.thresh
    ellipse = not args.boxes
    mask_scale = args.mask_scale
    keep_audio = args.keep_audio
    ffmpeg_config = args.ffmpeg_config
    backend = args.backend
    in_shape = args.scale
    execution_provider = args.execution_provider
    mosaicsize = args.mosaicsize
    keep_metadata = args.keep_metadata
    replaceimg = None
    if in_shape is not None:
        w, h = in_shape.split('x')
        in_shape = int(w), int(h)
    if replacewith == "img":
        replaceimg = imageio.imread(args.replaceimg)
        print(f'After opening {args.replaceimg} shape: {replaceimg.shape}')


    # TODO: scalar downscaling setting (-> in_shape), preserving aspect ratio
    centerface = CenterFace(in_shape=in_shape, backend=backend, override_execution_provider=execution_provider)

    multi_file = len(ipaths) > 1
    if multi_file:
        ipaths = tqdm.tqdm(ipaths, position=0, dynamic_ncols=True, desc='Batch progress')

    for ipath in ipaths:
        opath = base_opath
        if ipath == 'cam':
            ipath = '<video0>'
            enable_preview = True
        filetype = get_file_type(ipath)
        is_cam = filetype == 'cam'
        if opath is None and not is_cam:
            root, ext = os.path.splitext(ipath)
            opath = f'{root}_anonymized{ext}'
        print(f'Input:  {ipath}\nOutput: {opath}')
        if opath is None and not enable_preview:
            print('No output file is specified and the preview GUI is disabled. No output will be produced.')
        if filetype == 'video' or is_cam:
            video_detect(
                ipath=ipath,
                opath=opath,
                centerface=centerface,
                threshold=threshold,
                cam=is_cam,
                replacewith=replacewith,
                mask_scale=mask_scale,
                ellipse=ellipse,
                draw_scores=draw_scores,
                enable_preview=enable_preview,
                nested=multi_file,
                keep_audio=keep_audio,
                ffmpeg_config=ffmpeg_config,
                replaceimg=replaceimg,
                mosaicsize=mosaicsize,
                # target_person_image=target_person_image,
                target_embeddings=target_embeddings,
            )
        elif filetype == 'image':
            image_detect(
                ipath=ipath,
                opath=opath,
                centerface=centerface,
                threshold=threshold,
                replacewith=replacewith,
                mask_scale=mask_scale,
                ellipse=ellipse,
                draw_scores=draw_scores,
                enable_preview=enable_preview,
                keep_metadata=keep_metadata,
                replaceimg=replaceimg,
                mosaicsize=mosaicsize
            )
        elif filetype is None:
            print(f'Can\'t determine file type of file {ipath}. Skipping...')
        elif filetype == 'notfound':
            print(f'File {ipath} not found. Skipping...')
        else:
            print(f'File {ipath} has an unknown type {filetype}. Skipping...')


if __name__ == '__main__':
    main()
