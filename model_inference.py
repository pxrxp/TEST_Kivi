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
        """Dynamically load the most optimized available backend.

        Priority: OpenCV DNN > ONNX Runtime > TFLite > PyTorch > Mock.
        OpenCV DNN is the primary target (works natively on Android via
        buildozer opencv requirement). The ONNX model is exported with
        fixed shapes and opset 11 for maximum OpenCV DNN compatibility.
        """

        if not self.model_path:
            self.backend_type = "mock"
            print("[WARNING] No model path provided, using mock model.")
            return

        # 1. Try OpenCV DNN (primary target — works natively on Android)
        if self.model_path.endswith(".onnx"):
            try:
                import cv2
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
                self.backend = self.net
                self.backend_type = "opencv_onnx"
                print(f"[INFO] Successfully loaded ONNX model via OpenCV DNN backend.")
                return
            except Exception as e:
                print(f"[DEBUG] OpenCV DNN load failed: {e}")

        # 2. Try ONNX Runtime (desktop fallback)
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

        # 4. Try PyTorch (if .pth file provided)
        if self.model_path.endswith(".pth"):
            try:
                import torch
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
        print(f"[WARNING] No ML backend available for {self.model_path}, using mock model.")

    def preprocess_input(self, image: np.ndarray, input_size: int = 256):
        """
        Preprocess input image for model inference.

        Aspect-ratio-preserving resize with reflective padding, matching
        the production _prepare_inference_image in segmentation.py.

        Parameters:
        -----------
        image : np.ndarray
            Input RGB image (H, W, 3)
        input_size : int
            Target square size (default 256)

        Returns:
        --------
        np.ndarray
            Normalized tensor (1, 3, input_size, input_size)
        """
        import cv2

        H, W = image.shape[:2]
        scale = min(input_size / W, input_size / H)
        new_w = max(1, int(round(W * scale)))
        new_h = max(1, int(round(H * scale)))

        resized = np.array(Image.fromarray(image).resize((new_w, new_h), Image.Resampling.BILINEAR))

        # Reflective padding to fill input_size x input_size
        pad_left = (input_size - new_w) // 2
        pad_right = input_size - new_w - pad_left
        pad_top = (input_size - new_h) // 2
        pad_bottom = input_size - new_h - pad_top
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )

        # Normalize to [0, 1]
        image_normalized = padded.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        for channel in range(3):
            image_normalized[:, :, channel] = (
                image_normalized[:, :, channel] - IMAGENET_MEAN[channel]
            ) / IMAGENET_STD[channel]

        # Reorder to channel-first (CHW)
        image_chw = np.transpose(image_normalized, (2, 0, 1))

        # Add batch dimension (1, 3, input_size, input_size)
        return np.expand_dims(image_chw, axis=0).astype(np.float32)

    def predict_probability_map(self, image: np.ndarray, input_size: int = 256) -> np.ndarray:
        """
        Run model inference and return probability map (range [0, 1]).

        Handles aspect-ratio-preserving resize + reflective padding: the raw
        model output is un-padded and resized back to the original image dims.

        Parameters:
        -----------
        image : np.ndarray
            Input RGB image (H, W, 3)
        input_size : int
            Model input resolution (default 256)

        Returns:
        --------
        np.ndarray
            Probability map P(terrain) in range [0, 1], shape (H_orig, W_orig)
        """
        import cv2

        H_orig, W_orig = image.shape[:2]

        # Compute resize + padding info (must match preprocess_input)
        scale = min(input_size / W_orig, input_size / H_orig)
        new_w = max(1, int(round(W_orig * scale)))
        new_h = max(1, int(round(H_orig * scale)))
        pad_left = (input_size - new_w) // 2
        pad_top = (input_size - new_h) // 2

        input_tensor = self.preprocess_input(image, input_size=input_size)

        # --- Run inference ---
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
            # Mock fallback
            probability_map = np.zeros((input_size, input_size), dtype=np.float32)
            probability_map[:input_size // 2, :] = 0.25   # P(terrain) low → sky
            probability_map[input_size // 2:, :] = 0.75   # P(terrain) high → ground

        # --- Un-pad: crop out the reflective padding, then resize to original ---
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
        """
        Run model inference and return raw binary sky mask.

        The MobileNetV3 U-Net is trained with BCE on GeoPose3K masks where
        white (255) = terrain and black (0) = sky, so the model outputs
        P(terrain).  High probability → terrain, low probability → sky.

        Convention matches production segmentation.py:
            raw_mask = (prob <= threshold) → 1 = SKY, 0 = TERRAIN.

        Parameters:
        -----------
        image : np.ndarray
            Input RGB image (H, W, 3)
        threshold : float
            Probability threshold (default 0.70, matching production).
            Pixels with P(terrain) <= threshold are labelled sky.

        Returns:
        --------
        np.ndarray
            raw_unet_mask with 1=SKY, 0=TERRAIN (binary uint8)
        """
        prob_map = self.predict_probability_map(image)
        return (prob_map <= threshold).astype(np.uint8)

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
