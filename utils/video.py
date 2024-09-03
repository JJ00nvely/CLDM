import tempfile
import cv2
import os
import shutil

def convert_images_to_video_file(self, images, frame_rate=1):
    if not images:
        raise ValueError("No images to convert to video")

    # PIL 이미지를 numpy 배열로 변환하고 3채널 RGB인지 확인
    processed_images = []
    for img in images:
        img_np = np.array(img)
        if img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]
        processed_images.append(img_np)

    if processed_images[0].ndim != 3 or processed_images[0].shape[2] != 3:
        raise ValueError("Images must have 3 channels (RGB)")

    frame_height, frame_width, _ = processed_images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 비디오를 저장할 임시 파일 생성
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    video_path = temp_file.name
    temp_file.close()  # OpenCV가 파일에 쓸 수 있도록 파일 닫기

    # 임시 파일에 비디오를 저장하는 VideoWriter 생성
    video_writer = cv2.VideoWriter(video_path, fourcc, frame_rate, (frame_width, frame_height))

    for img in processed_images:
        video_writer.write(img)

    video_writer.release()
    return video_path



def video_make(img_bbox, dir,epoch):
    bbox=convert_images_to_video_file(img_bbox,frame_rate=1)
    os.makedirs(dir, exist_ok=True)
    local_video_path_bbox = os.path.join(dir, f'pred_bboxes_video_1_epoch_{epoch}.mp4')
    shutil.copy(bbox, local_video_path_bbox)
    os.remove(bbox)