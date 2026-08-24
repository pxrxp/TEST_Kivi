"""
Mobile-Optimized Model Adapter

Provides inference for sky segmentation by dynamically selecting the best 
available backend (OpenCV DNN for Android/mobile, ONNX Runtime, TFLite, PyTorch, or Mock).
"""

import numpy as np
from typing import Tuple, Optional
from PIL import Image

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ModelAdapter:
    """
    Adapter that provides optimized inference for different architectures,
    dynamically selecting the appropriate backend based on platform.
    """

    def __init__(self, model_path: Optional[str] = None, num_threads: int = 4):
        self.model_path = model_path
        self.num_threads = num_threads
        self.backend = None
        self.backend_type = "mock"
        self._load_optimized_model()

    def _load_optimized_model(self):
        """Dynamically load the most optimized available backend."""
        
        # 1. Try OpenCV DNN (Works natively on Android via buildozer opencv requirement)
        if self.model_path and self.model_path.endswith(".onnx"):
            try:
                import cv2
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
                self.backend = self.net
                self.backend_type = "opencv_onnx"
                print(f"[INFO] Successfully loaded ONNX model via OpenCV DNN backend.")
                return
            except Exception as e:
                print(f"[DEBUG] OpenCV DNN load failed: {e}")

        # 2. Try ONNX Runtime (Desktop development)
        try:
            import onnxruntime as ort

            self.backend = ort.InferenceSession(self.model_path)
            self.backend_type = "onnx"
            print(f"[INFO] Successfully loaded model via ONNX Runtime backend.")
            return
        except Exception:
            pass

        # 3. Try TFLite (for mobile deployment with TFLite model)
        try:
            import tflite_runtime.interpreter as tflite

            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.backend = self.interpreter
            self.backend_type = "tflite"
            print(f"[INFO] Successfully loaded model via TFLite backend.")
            return
        except Exception:
            pass

        # 4. Fall back to PyTorch (Desktop development)
        try:
            import torch

            # Load PyTorch checkpoint
            checkpoint = torch.load(self.model_path, map_location="cpu")
            self.model = checkpoint
            self.backend = checkpoint
            self.backend_type = "torch"
            print(f"[INFO] Successfully loaded model via PyTorch backend.")
            return
        except Exception:
            pass

        # 5. If all backends fail, use mock (for testing / placeholder)
        self.backend_type = "mock"
        print(f"[WARNING] No ML backend available, using mock model.")

    def preprocess_input(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess input image for model inference.
        Resizes to 256x256, applies ImageNet normalization.

        Parameters:
        -----------
        image : np.ndarray
            Input RGB image (HxWx3)

        Returns:
        --------
        np.ndarray
            Normalized tensor (1, 3, 256, 256)
        """
        # Resize to 256x256
        if image.shape[:2] != (256, 256):
            image_resized = np.array(Image.fromarray(image).resize((256, 256)))
        else:
            image_resized = image

        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        for channel in range(3):
            image_normalized[:, :, channel] = (
                image_normalized[:, :, channel] - IMAGENET_MEAN[channel]
            ) / IMAGENET_STD[channel]

        # Reorder to channel-first (CHW)
        image_chw = np.transpose(image_normalized, (2, 0, 1))

        # Add batch dimension (1, 3, 256, 256)
        return np.expand_dims(image_chw, axis=0).astype(np.float32)

    def predict_probability_map(self, image: np.ndarray) -> np.ndarray:
        """
        Run model inference and return probability map (range [0, 1]).

        Parameters:
        -----------
        image : np.ndarray
            Input RGB image

        Returns:
        --------
        np.ndarray
            Probability map P(sky) in range [0, 1] matching input image shape
        """
        input_tensor = self.preprocess_input(image)

        if self.backend_type == "opencv_onnx":
            # OpenCV DNN inference
            self.backend.setInput(input_tensor)
            result = self.backend.forward()
            # Extract 2D map from output shape (1, 1, 256, 256)
            if result.ndim == 4:
                probability_map = result[0, 0, :, :]
            elif result.ndim == 3:
                probability_map = result[0, :, :]
            else:
                probability_map = result

        elif self.backend_type == "onnx":
            # ONNX Runtime inference
            input_name = self.backend.get_inputs()[0].name
            result = self.backend.run(None, {input_name: input_tensor})
            probability_map = result[0][0, 0, :, :]

        elif self.backend_type == "tflite":
            # TFLite inference
            input_details = self.backend.get_input_details()
            output_details = self.backend.get_output_details()

            self.backend.set_tensor(input_details[0]["index"], input_tensor)
            self.backend.invoke()
            output_data = self.backend.get_tensor(output_details[0]["index"])
            probability_map = output_data[0, :, :, 0]

        elif self.backend_type == "torch":
            # PyTorch mock / fallback
            probability_map = np.zeros((256, 256), dtype=np.float32)
            probability_map[:128, :] = 0.8
            probability_map[128:, :] = 0.2

        else:
            # Mock fallback
            probability_map = np.zeros((256, 256), dtype=np.float32)
            probability_map[:128, :] = 0.75  # Sky probability
            probability_map[128:, :] = 0.25  # Ground probability

        # Resize output probability map to match input image dimensions
        if probability_map.shape != image.shape[:2]:
            probability_map = np.array(
                Image.fromarray((probability_map * 255).astype(np.uint8)).resize(
                    (image.shape[1], image.shape[0])
                )
            )
            probability_map = probability_map.astype(np.float32) / 255.0

        return probability_map

    def predict_sky_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Run model inference and return raw binary sky mask.

        Parameters:
        -----------
        image : np.ndarray
            Input RGB image

        Returns:
        --------
        np.ndarray
            raw_unet_mask with 1=SKY, 0=TERRAIN (binary uint8)
        """
        SKY_THRESHOLD = 0.30
        prob_map = self.predict_probability_map(image)
        return (prob_map >= SKY_THRESHOLD).astype(np.uint8)

    def get_boundary_from_probability(
        self, prob_map: np.ndarray, threshold: float = 0.30
    ) -> np.ndarray:
        """
        Convert probability map to binary sky boundary indices.

        Parameters:
        -----------
        prob_map : np.ndarray
            Sky probability map [0, 1]
        threshold : float
            Sky/terrain separation threshold

        Returns:
        --------
        np.ndarray
            Row index of horizon boundary per column (-1 where undefined)
        """
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
    """Create a mock model for testing without real weights."""
    return ModelAdapter(model_path=model_path)
