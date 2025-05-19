from pytube import YouTube
import os
import glob
import os
import cv2
from ultralytics import YOLO
import shutil
import torch
import torchvision
from tqdm import tqdm
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 디바이스: {device}")
if device.type == "cuda":
    print(f"사용 중인 GPU: {torch.cuda.get_device_name(0)}")
#####################################################################################
def func_oneway(video_url, pt_path, frame_interval):
    os.makedirs('./human_in_center/save_video', exist_ok=True)
    video = YouTube(video_url)
    video_stream = video.streams.filter(adaptive=True, file_extension='webm').first()   
    
    saved_video_path = video_stream.download('./human_in_center/save_video')
    video_filename = saved_video_path.split('\\')[-1]
    print(video_filename)
    extract_frames(saved_video_path, frame_interval)
    func_only1_save(pt_path, saved_video_path)
 ##################################################################################### 동영상 저장 및 제목 추출
   
    
def extract_frames(video_path, frame_interval):
    # 동영상 파일 열기
    cap = cv2.VideoCapture(video_path)
    video_name = video_path.split('\\')[-1]
    os.makedirs(f'./human_in_center/extracted_image/{video_name}', exist_ok=True)
    # 동영상 파일 열기에 실패한 경우 종료
    if not cap.isOpened():
        print("동영상 파일을 열 수 없습니다.")
        return

    frame_count = 0
    interval_counter = 0

    # tqdm으로 루프 감싸기
    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        # 프레임 읽기
        ret, frame = cap.read()

        # 프레임 읽기에 실패하거나 동영상의 끝에 도달한 경우 종료
        if not ret:
            break

        frame_count += 1

        # 일정 간격마다 이미지 파일로 저장
        if interval_counter == frame_interval:
            frame_filename = os.path.join(f'./human_in_center/extracted_image/{video_name}',
                                          f"frame_{frame_count:04d}.png")
            cv2.imwrite(frame_filename, frame)
            interval_counter = 0

        interval_counter += 1

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 종료 후 리소스 해제
    cap.release()
    cv2.destroyAllWindows()


def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)

    x_union = min(x1 + w1, x2 + w2)
    y_union = min(y1 + h1, y2 + h2)

    intersection_area = max(0, x_union - x_intersection) * max(0, y_union - y_intersection)
    box1_area = w1 * h1
    box2_area = w2 * h2

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    print(f"iou: {iou}")

    return iou

def calculate_similarity(image1, image2):
    # 이미지를 PyTorch 텐서로 변환합니다.
    image1 = image1.to(device)
    image2 = image2.to(device)

    # 이미지 간의 L2 거리를 계산합니다.
    distance = torch.dist(image1.view(-1), image2.view(-1), p=2)

    # 유사도를 계산합니다. (값이 작을수록 유사함)
    similarity = 1 / (1 + distance)

    return similarity

def similarity_finalpicture(source_folder):
    for root, _, _ in os.walk(source_folder):
        # 현재 서브폴더에 있는 *.png 파일 목록을 가져옵니다.
        image_files = [f for f in os.listdir(root) if f.endswith('.png')]

        # 대상 폴더를 현재 서브폴더에 따라 생성합니다.
        target_root=f"{source_folder}/"
        target_folder = os.path.join(target_root, os.path.basename(root))
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # tqdm을 사용하여 현재 서브폴더의 이미지를 순서대로 비교하고 유사도가 기준 이하인 경우 복사합니다.
        with tqdm(total=len(image_files) - 1, desc="Processing Images in " + root) as pbar:
            for i in range(len(image_files) - 1):
                current_image_path = os.path.join(root, image_files[i])
                next_image_path = os.path.join(root, image_files[i + 1])

                # 이미지를 열어서 PIL Image로 변환합니다.
                current_image = Image.open(current_image_path)
                next_image = Image.open(next_image_path)

                # 이미지를 PyTorch 텐서로 변환합니다.
                current_image = torchvision.transforms.ToTensor()(current_image).to(device)
                next_image = torchvision.transforms.ToTensor()(next_image).to(device)

                # 현재 이미지와 다음 이미지 간의 유사도를 계산합니다.
                similarity = calculate_similarity(current_image, next_image)

                # 유사도가 **% 이하인 경우만 이미지를 대상 폴더로 복사합니다.
                if similarity <= 0.002:
                    target_path = os.path.join(target_folder, image_files[i + 1])
                    shutil.copy(next_image_path, target_path)

                pbar.update(1)

def image_determine(boxes, cls, conf):
    deter_result = 1
    conf_threshold = 0.8
    iou_threshold = 0.6

    if len(boxes) == 1:
        if int(cls[0]) == 0 and float(conf[0]) > conf_threshold:
            deter_result = 0
        else:
            pass
    elif len(boxes) == 2:
        if (int(cls[0]) == 0 and int(cls[1]) == 1) or (int(cls[0]) == 1 and int(cls[1]) == 0):
            box1 = boxes[0]
            box2 = boxes[1]
            iou_boxes = calculate_iou(box1, box2)
            if iou_boxes > iou_threshold:
                deter_result = 0
            else:
                pass
        else:
            pass
    return deter_result
#모델보고 박스위치 크기조건추가?


def func_only1_save(pt_path, video_path):
    model = YOLO(pt_path)
    video_name = video_path.split('\\')[-1]
    image_path_list = glob.glob(os.path.join(f'./human_in_center/extracted_image/{video_name}', '*.png'))
    save_number = 1
    source_folder = f'./human_in_center/all/{video_name}'



    for path in image_path_list:
        results = model.predict(
            path,
            save=False,
            imgsz=640,
            conf=0.5,
            device='cuda'
        )

        for r in results:
            boxes = r.boxes.xyxy
            cls = r.boxes.cls
            conf = r.boxes.conf
            cls_dict = r.names
            image = cv2.imread(path)
            det_result = image_determine(boxes,cls,conf)
            if det_result == 0:
                for box, cls_number, conf in zip(boxes, cls, conf):
                    conf_number = float(conf.item())
                    cls_number_int = int(cls_number.item())
                    cls_name = cls_dict[cls_number_int]
                    x1, y1, x2, y2 = box
                    x1_int = int(x1.item())
                    x2_int = int(x2.item())
                    y1_int = int(y1.item())
                    y2_int = int(y2.item())

                    # print(x1_int, y1_int, x2_int, y2_int, cls_name, conf_number)
                    color_map = {'human': (0, 255, 0), 'body': (255, 0, 0)}  # human 초록 body 파랑
                    bbox_color = color_map.get(cls_name, (255, 255, 255))
                    image = cv2.rectangle(image, (x1_int, y1_int), (x2_int, y2_int), bbox_color, 2)

                os.makedirs(f'{source_folder}', exist_ok=True)
                cv2.imwrite(f'{source_folder}/all_frame_no_{save_number}.png', image)
                save_number += 1
            else:
                pass
    
    similarity_finalpicture(source_folder)
    print('작업이 끝났습니다!')

if __name__ == "__main__":
    video_url = 'https://www.youtube.com/watch?v=swkFs7G6Tlc'    
    pt_path = "C:/Users/123/Downloads/x150/train5/weights/best.pt"
    frame_interval = 15  

    func_oneway(video_url, pt_path, frame_interval)