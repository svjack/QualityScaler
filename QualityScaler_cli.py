import sys
import os
import argparse
import subprocess
from time import sleep
from multiprocessing import Process, Queue
from onnxruntime import InferenceSession
from PIL import Image
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip
from typing import List

# Constants
AI_LIST_SEPARATOR = ["----"]
IRCNN_models_list = ["IRCNN_Mx1", "IRCNN_Lx1"]
SRVGGNetCompact_models_list = ["RealESR_Gx4", "RealSRx4_Anime"]
RRDB_models_list = ["BSRGANx4", "BSRGANx2", "RealESRGANx4"]
AI_models_list = SRVGGNetCompact_models_list + AI_LIST_SEPARATOR + RRDB_models_list + AI_LIST_SEPARATOR + IRCNN_models_list
gpus_list = ["GPU 1", "GPU 2", "GPU 3", "GPU 4"]
image_extension_list = [".png", ".jpg", ".bmp", ".tiff"]
video_extension_list = [".mp4 (x264)", ".mp4 (x265)", ".avi"]
interpolation_list = ["Low", "Medium", "High", "Disabled"]
AI_multithreading_list = ["1 threads", "2 threads", "3 threads", "4 threads", "5 threads", "6 threads"]

# Helper functions
def find_by_relative_path(relative_path: str) -> str:
    base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

def image_read(file_path: str, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    with open(file_path, 'rb') as file:
        return cv2.imdecode(np.frombuffer(file.read(), np.uint8), flags)

def image_write(file_path: str, file_data: np.ndarray) -> None:
    _, file_extension = os.path.splitext(file_path)
    cv2.imencode(file_extension, file_data)[1].tofile(file_path)

def get_image_resolution(image: np.ndarray) -> tuple:
    height = image.shape[0]
    width = image.shape[1]
    return height, width

def get_video_fps(video_path: str) -> float:
    video_capture = cv2.VideoCapture(video_path)
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    video_capture.release()
    return frame_rate

def extract_video_frames_and_audio(video_path: str, target_directory: str, cpu_number: int):
    os.makedirs(target_directory, exist_ok=True)

    # Audio extraction
    with VideoFileClip(video_path) as video_file_clip:
        try:
            audio_path = f"{target_directory}/audio.mp3"
            video_file_clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
        except:
            pass

    # Video frame extraction
    video_capture = cv2.VideoCapture(video_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_to_save = []
    frames_path_to_save = []
    video_frames_list = []

    for frame_number in range(frame_count):
        success, frame = video_capture.read()
        if success:
            frames_to_save.append(frame)
            frame_path = f"{target_directory}/frame_{frame_number:03d}.jpg"
            frames_path_to_save.append(frame_path)
            video_frames_list.append(frame_path)

    video_capture.release()

    return video_frames_list

def video_reconstruction_by_frames(video_path: str, audio_path: str, video_output_path: str, upscaled_frame_list_paths: list[str], cpu_number: int, selected_video_extension: str) -> None:
    frame_rate = get_video_fps(video_path)

    clip = ImageSequenceClip(sequence=upscaled_frame_list_paths, fps=frame_rate)
    clip.write_videofile(
        video_output_path,
        fps=frame_rate,
        audio=audio_path if os.path.exists(audio_path) else None,
        codec='libx264' if selected_video_extension == '.mp4' else 'png',
        bitrate='12M',
        verbose=False,
        logger=None,
        threads=cpu_number,
        preset="ultrafast"
    )

def upscale_image(image_path: str, output_path: str, AI_instance: 'AI', selected_AI_model: str, selected_image_extension: str, resize_factor: float, selected_interpolation_factor: float) -> None:
    starting_image = image_read(image_path)
    upscaled_image = AI_instance.AI_orchestration(starting_image)

    if selected_interpolation_factor > 0:
        interpolate_images_and_save(output_path, starting_image, upscaled_image, selected_interpolation_factor)
    else:
        image_write(output_path, upscaled_image)

def upscale_video(video_path: str, output_path: str, AI_instance: 'AI', selected_AI_model: str, resize_factor: float, cpu_number: int, selected_video_extension: str, selected_interpolation_factor: float) -> None:
    target_directory = os.path.splitext(output_path)[0]
    os.makedirs(target_directory, exist_ok=True)

    frame_list_paths = extract_video_frames_and_audio(video_path, target_directory, cpu_number)
    upscaled_frame_list_paths = [f"{target_directory}/upscaled_{os.path.basename(frame_path)}" for frame_path in frame_list_paths]

    for frame_index, frame_path in enumerate(frame_list_paths):
        upscaled_frame_path = upscaled_frame_list_paths[frame_index]
        if not os.path.exists(upscaled_frame_path):
            starting_frame = image_read(frame_path)
            upscaled_frame = AI_instance.AI_orchestration(starting_frame)
            manage_upscaled_video_frame_save_async(upscaled_frame, starting_frame, upscaled_frame_path, selected_interpolation_factor)

    video_reconstruction_by_frames(video_path, f"{target_directory}/audio.mp3", output_path, upscaled_frame_list_paths, cpu_number, selected_video_extension)

def interpolate_images_and_save(target_path: str, starting_image: np.ndarray, upscaled_image: np.ndarray, starting_image_importance: float) -> None:
    upscaled_image_importance = 1 - starting_image_importance
    starting_height, starting_width = get_image_resolution(starting_image)
    target_height, target_width = get_image_resolution(upscaled_image)

    starting_resolution = starting_height + starting_width
    target_resolution = target_height + target_width

    if starting_resolution > target_resolution:
        starting_image = cv2.resize(starting_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    else:
        starting_image = cv2.resize(starting_image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    interpolated_image = cv2.addWeighted(starting_image, starting_image_importance, upscaled_image, upscaled_image_importance, 0)
    image_write(target_path, interpolated_image)

def manage_upscaled_video_frame_save_async(upscaled_frame: np.ndarray, starting_frame: np.ndarray, upscaled_frame_path: str, selected_interpolation_factor: float) -> None:
    if selected_interpolation_factor > 0:
        interpolate_images_and_save(upscaled_frame_path, starting_frame, upscaled_frame, selected_interpolation_factor)
    else:
        image_write(upscaled_frame_path, upscaled_frame)

class AI:
    def __init__(self, AI_model_name: str, directml_gpu: str, resize_factor: float, max_resolution: int):
        self.AI_model_name = AI_model_name
        self.directml_gpu = directml_gpu
        self.resize_factor = resize_factor
        self.max_resolution = max_resolution
        self.AI_model_path = find_by_relative_path(f"AI-onnx/{self.AI_model_name}_fp16.onnx")
        self.inferenceSession = self._load_inferenceSession()
        self.upscale_factor = self._get_upscale_factor()

    def _get_upscale_factor(self) -> int:
        if "x1" in self.AI_model_name:
            return 1
        elif "x2" in self.AI_model_name:
            return 2
        elif "x4" in self.AI_model_name:
            return 4
        return 1

    def _load_inferenceSession(self) -> InferenceSession:
        #directml_backend = [('DmlExecutionProvider', {"device_id": gpus_list.index(self.directml_gpu)})]
        providers = ['CUDAExecutionProvider','CPUExecutionProvider',]
        return InferenceSession(self.AI_model_path, providers=providers)

    def get_image_mode(self, image: np.ndarray) -> str:
        if len(image.shape) == 2:
            return "Grayscale"
        elif image.shape[2] == 3:
            return "RGB"
        elif image.shape[2] == 4:
            return "RGBA"
        return "RGB"

    def get_image_resolution(self, image: np.ndarray) -> tuple:
        return image.shape[:2]

    def calculate_target_resolution(self, image: np.ndarray) -> tuple:
        height, width = self.get_image_resolution(image)
        target_height = height * self.upscale_factor
        target_width = width * self.upscale_factor
        return target_height, target_width

    def resize_image_with_resize_factor(self, image: np.ndarray) -> np.ndarray:
        old_height, old_width = self.get_image_resolution(image)
        new_width = int(old_width * self.resize_factor)
        new_height = int(old_height * self.resize_factor)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR if self.resize_factor > 1 else cv2.INTER_AREA)

    def resize_image_with_target_resolution(self, image: np.ndarray, t_height: int, t_width: int) -> np.ndarray:
        return cv2.resize(image, (t_width, t_height), interpolation=cv2.INTER_LINEAR if t_height + t_width > image.shape[0] + image.shape[1] else cv2.INTER_AREA)

    def normalize_image(self, image: np.ndarray) -> tuple:
        range = 255 if np.max(image) > 256 else 65535
        normalized_image = image / range
        return normalized_image, range

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image

    def onnxruntime_inference(self, image: np.ndarray) -> np.ndarray:
        onnx_input = {self.inferenceSession.get_inputs()[0].name: image}
        onnx_output = self.inferenceSession.run(None, onnx_input)[0]
        return onnx_output

    def postprocess_output(self, onnx_output: np.ndarray) -> np.ndarray:
        onnx_output = np.squeeze(onnx_output, axis=0)
        onnx_output = np.clip(onnx_output, 0, 1)
        onnx_output = np.transpose(onnx_output, (1, 2, 0))
        return onnx_output.astype(np.float32)

    def de_normalize_image(self, onnx_output: np.ndarray, max_range: int) -> np.ndarray:
        return (onnx_output * max_range).astype(np.uint8) if max_range == 255 else (onnx_output * max_range).round().astype(np.float32)

    def AI_upscale(self, image: np.ndarray) -> np.ndarray:
        image_mode = self.get_image_mode(image)
        image, range = self.normalize_image(image)
        image = image.astype(np.float32)

        if image_mode == "RGB":
            image = self.preprocess_image(image)
            onnx_output = self.onnxruntime_inference(image)
            onnx_output = self.postprocess_output(onnx_output)
            output_image = self.de_normalize_image(onnx_output, range)
            return output_image

        elif image_mode == "RGBA":
            alpha = image[:, :, 3]
            image = image[:, :, :3]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)
            alpha = alpha.astype(np.float32)

            image = self.preprocess_image(image)
            onnx_output_image = self.onnxruntime_inference(image)
            onnx_output_image = self.postprocess_output(onnx_output_image)
            onnx_output_image = cv2.cvtColor(onnx_output_image, cv2.COLOR_BGR2RGBA)

            alpha = np.expand_dims(alpha, axis=-1)
            alpha = np.repeat(alpha, 3, axis=-1)
            alpha = self.preprocess_image(alpha)
            onnx_output_alpha = self.onnxruntime_inference(alpha)
            onnx_output_alpha = self.postprocess_output(onnx_output_alpha)
            onnx_output_alpha = cv2.cvtColor(onnx_output_alpha, cv2.COLOR_RGB2GRAY)

            onnx_output_image[:, :, 3] = onnx_output_alpha
            output_image = self.de_normalize_image(onnx_output_image, range)
            return output_image

        elif image_mode == "Grayscale":
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = self.preprocess_image(image)
            onnx_output = self.onnxruntime_inference(image)
            onnx_output = self.postprocess_output(onnx_output)
            output_image = cv2.cvtColor(onnx_output, cv2.COLOR_RGB2GRAY)
            output_image = self.de_normalize_image(onnx_output, range)
            return output_image

    def AI_orchestration(self, image: np.ndarray) -> np.ndarray:
        resized_image = self.resize_image_with_resize_factor(image)
        return self.AI_upscale(resized_image)

def main():
    parser = argparse.ArgumentParser(description="AI Video and Image Upscaler")
    parser.add_argument("input_file", type=str, help="Path to the input file (image or video)")
    parser.add_argument("output_file", type=str, help="Path to the output file (image or video)")
    parser.add_argument("--ai_model", type=str, choices=AI_models_list, default=AI_models_list[0], help="AI model to use for upscaling")
    parser.add_argument("--gpu", type=str, choices=gpus_list, default=gpus_list[0], help="GPU to use for upscaling")
    parser.add_argument("--resize_factor", type=float, default=1.0, help="Resize factor for input image/video (0.0 to 1.0)")
    parser.add_argument("--interpolation", type=str, choices=interpolation_list, default="Disabled", help="Interpolation mode for upscaled image/video")
    parser.add_argument("--cpu_number", type=int, default=1, help="Number of CPU threads to use")
    parser.add_argument("--video_extension", type=str, choices=video_extension_list, default=video_extension_list[0], help="Output video extension")
    parser.add_argument("--image_extension", type=str, choices=image_extension_list, default=image_extension_list[0], help="Output image extension")
    parser.add_argument("--vram_limit", type=int, default=4, help="VRAM limit in GB")

    args = parser.parse_args()

    selected_interpolation_factor = {
        "Disabled": 0,
        "Low": 0.3,
        "Medium": 0.5,
        "High": 0.7,
    }.get(args.interpolation)

    AI_instance = AI(args.ai_model, args.gpu, args.resize_factor, args.vram_limit * 100)

    if args.input_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        upscale_image(args.input_file, args.output_file, AI_instance, args.ai_model, args.image_extension, args.resize_factor, selected_interpolation_factor)
    elif args.input_file.lower().endswith(('.mp4', '.avi', '.mkv', '.flv', '.gif', '.mov', '.m4v', '.3gp', '.mpg', '.mpeg')):
        upscale_video(args.input_file, args.output_file, AI_instance, args.ai_model, args.resize_factor, args.cpu_number, args.video_extension, selected_interpolation_factor)
    else:
        print("Unsupported file format")

if __name__ == "__main__":
    main()
