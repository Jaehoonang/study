import glob
import os
import cv2
from ultralytics import YOLO
import cv2
from PIL import Image
import os
import shutil
import torch
import torchvision
from tqdm import tqdm

###동영상에서 이미지 추출  frame_interval = 15  # 프레임 간격 (0.5초에 해당하는 값)에서 프레임 설정

# GPU가 사용 가능한지 확인합니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 디바이스: {device}")
if device.type == "cuda":
    print(f"사용 중인 GPU: {torch.cuda.get_device_name(0)}")

def extract_frames(video_path, output_folder, frame_interval=15):
    # 동영상 파일 열기
    cap = cv2.VideoCapture(video_path)

    # 동영상 파일 열기에 실패한 경우 종료
    if not cap.isOpened():
        print("동영상 파일을 열 수 없습니다.")
        return

    # 출력 폴더가 없는 경우 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            interval_counter = 0

        interval_counter += 1

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 종료 후 리소스 해제
    cap.release()
    cv2.destroyAllWindows()

def func_only1_save(pt_path, image_path, video_file):
    model = YOLO(pt_path)
    image_path_list = glob.glob(os.path.join(image_path, '*.jpg')) # 다른 형식도 고려해야함
    save_number = 1
    folder_save_name = 1

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
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if len(boxes) == 1: # 추가된 코드 한 줄
                os.makedirs(f'./human_in_center/{video_file}', exist_ok=True)
                cv2.imwrite(f'./human_in_center/{video_file}/best_photo{save_number}.png', image) ################################ 다른 동영상을 넣었을 때 덮었음 
                save_number += 1
                # #################################################################################################### 밑에는 나중에 필요 없음(bbox 확인용)
                for box, cls_number, conf in zip(boxes, cls, conf):
                    conf_number = float(conf.item())
                    cls_number_int = int(cls_number.item())
                    cls_name = cls_dict[cls_number_int]
                    x1, y1, x2, y2 = box
                    x1_int = int(x1.item())
                    x2_int = int(x2.item())
                    y1_int = int(y1.item())
                    y2_int = int(y2.item())
                                    
                    print(x1_int, y1_int, x2_int, y2_int, cls_name, conf_number)
                    image = cv2.rectangle(image, (x1_int, y1_int), (x2_int, y2_int), (0, 255, 0), 2) # rectangle 확인용 나중에 지우면 됨
                    
                
                cv2.imshow('only 1 in shoot', image)
                cv2.waitKey(0)
                


if __name__ == "__main__":
    video_file = "my_girl_85sec.mp4"  # 동영상 파일 경로 설정
    output_directory = f"./play_to_image/{video_file}"  # 이미지를 저장할 폴더 경로 설정
    frame_interval = 15  # 프레임 간격 (0.5초에 해당하는 값)
    extract_frames(video_file, output_directory, frame_interval)

    # pt 파일 주소
    pt_path = "C:/Users/123/Downloads/train_m125/train/weights/best.pt"
    # 넣을 이미지 주소
    # image_path = 'C:/Users/123/Downloads/real_test'
    # image_path = 'C:/Users/123/Desktop/stat/code/teampro/teampro/ultralytics-main/ultralytics/cfg/teamproject/test/images'
    func_only1_save(pt_path, output_directory, video_file)