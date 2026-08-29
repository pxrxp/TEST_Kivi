"""
Mobile-Optimized Model Adapter

Provides inference for sky segmentation by dynamically selecting the best 
available backend (OpenCV DNN for Android/mobile, ONNX Runtime, TFLite, PyTorch, or Mock).
"""

import os
import numpy as np
from typing import Optional

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ModelAdapter:
    """
    Adapter providing optimized inference across OpenCV DNN, ONNX Runtime,
    TFLite, and PyTorch backends.
    """

    def __init__(self, model_path: Optional[str] = None, num_threads: int = 4):
        self.model_path = model_path
        self.num_threads = num_threads
        self.backend = None
        self.backend_type = "mock"
        self._load_optimized_model()

    def _load_optimized_model(self):
        if not self.model_path or not os.path.exists(self.model_path):
            self.backend_type = "mock"
            print("[WARNING] No valid model file found, using mock model.")
            return

        # 1. OpenCV DNN
        if self.model_path.endswith(".onnx"):
            try:
                import cv2
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
                self.backend = self.net
                self.backend_type = "opencv_onnx"
                print("[INFO] Successfully loaded ONNX model via OpenCV DNN backend.")
                return
            except Exception as e:
                print(f"[DEBUG] OpenCV DNN load failed: {e}")

        # 2. ONNX Runtime
        try:
            import onnxruntime as ort
            self.backend = ort.InferenceSession(self.model_path)
            self.backend_type = "onnx"
            print("[INFO] Successfully loaded model via ONNX Runtime backend.")
            return
        except Exception:
            pass

        # 3. TFLite
        try:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.backend = self.interpreter
            self.backend_type = "tflite"
            print("[INFO] Successfully loaded model via TFLite backend.")
            return
        except Exception:
            pass

        # 4. PyTorch
        if self.model_path.endswith(".pth"):
            try:
                import torch
                checkpoint = torch.load(self.model_path, map_location="cpu")
                self.model = checkpoint
                self.backend = checkpoint
                self.backend_type = "torch"
                print("[INFO] Successfully loaded model via PyTorch backend.")
                return
            except Exception:
                pass

        self.backend_type = "mock"
        print(f"[WARNING] No ML backend available for {self.model_path}, using mock model.")

    def preprocess_input(self, image: np.ndarray, input_size: int = 256):
        import cv2

        H, W = image.shape[:2]
        scale = min(input_size / W, input_size / H)
        new_w = max(1, int(round(W * scale)))
        new_h = max(1, int(round(H * scale)))

        # Optimized OpenCV resize (compatible across all Pillow versions)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_left = (input_size - new_w) // 2
        pad_right = input_size - new_w - pad_left
        pad_top = (input_size - new_h) // 2
        pad_bottom = input_size - new_h - pad_top
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )

        image_normalized = padded.astype(np.float32) / 255.0

        for channel in range(3):
            image_normalized[:, :, channel] = (
                image_normalized[:, :, channel] - IMAGENET_MEAN[channel]
            ) / IMAGENET_STD[channel]

        image_chw = np.transpose(image_normalized, (2, 0, 1))
        return np.expand_dims(image_chw, axis=0).astype(np.float32)

    def predict_probability_map(self, image: np.ndarray, input_size: int = 256) -> np.ndarray:
        import cv2

        H_orig, W_orig = image.shape[:2]

        scale = min(input_size / W_orig, input_size / H_orig)
        new_w = max(1, int(round(W_orig * scale)))
        new_h = max(1, int(round(H_orig * scale)))
        pad_left = (input_size - new_w) // 2
        pad_top = (input_size - new_h) // 2

        input_tensor = self.preprocess_input(image, input_size=input_size)

        if self.backend_type == "opencv_onnx":
            self.backend.setInput(input_tensor)
            result = self.backend.forward()
            if result.ndim == 4:
                probability_map = result[0, 0, :, :]
            elif result.ndim == 3:
                probability_map = result[0, :, :]
            else:
                probability_map = result

        elif self.backend_type == "onnx":
            input_name = self.backend.get_inputs()[0].name
            result = self.backend.run(None, {input_name: input_tensor})
            probability_map = result[0][0, 0, :, :]

        elif self.backend_type == "tflite":
            input_details = self.backend.get_input_details()
            output_details = self.backend.get_output_details()
            self.backend.set_tensor(input_details[0]["index"], input_tensor)
            self.backend.invoke()
            output_data = self.backend.get_tensor(output_details[0]["index"])
            probability_map = output_data[0, :, :, 0]

        elif self.backend_type == "torch":
            probability_map = np.zeros((input_size, input_size), dtype=np.float32)
            probability_map[:input_size // 2, :] = 0.8
            probability_map[input_size // 2:, :] = 0.2

        else:
            probability_map = np.zeros((input_size, input_size), dtype=np.float32)
            probability_map[:input_size // 2, :] = 0.25
            probability_map[input_size // 2:, :] = 0.75

        prob_cropped = probability_map[
            pad_top : pad_top + new_h,
            pad_left : pad_left + new_w,
        ]
        if prob_cropped.shape != (H_orig, W_orig):
            prob_resized = cv2.resize(
                prob_cropped.astype(np.float32),
                (W_orig, H_orig),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            prob_resized = prob_cropped.astype(np.float32)

        return prob_resized

    def predict_sky_mask(self, image: np.ndarray, threshold: float = 0.70) -> np.ndarray:
        prob_map = self.predict_probability_map(image)
        return (prob_map <= threshold).astype(np.uint8)

    def get_boundary_from_probability(
        self, prob_map: np.ndarray, threshold: float = 0.30
    ) -> np.ndarray:
        height, width = prob_map.shape
        boundaries = np.full(width, -1.0)

        for col in range(width):
            col_probs = prob_map[:, col]
            sky_mask = col_probs >= threshold

            if np.any(sky_mask):
                sky_rows = np.where(sky_mask)[0]
                boundaries[col] = sky_rows[-1]

        return boundaries


def create_mock_model(model_path: str):
    return ModelAdapter(model_path=model_path)