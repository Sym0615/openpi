from typing import Any, Dict, Optional, Union

import base64
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
import io
import binascii
from turbojpeg import TurboJPEG, TJPF_RGB


_jpeg = TurboJPEG()


def _is_image(arr: np.ndarray) -> bool:
    if arr.dtype != np.uint8:
        return False

    if arr.ndim == 2:
        return True

    if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        return True

    return False


def encode_numpy(data: np.ndarray):
    if _is_image(data):
        jpeg_bytes = _jpeg.encode(data, quality=85, pixel_format=TJPF_RGB)
        metadata = {
            "type": "jpeg_bytes",
            "dtype": str(data.dtype),
            "shape": data.shape,
            "data": jpeg_bytes,
        }
    else:
        metadata = {
            "type": "numpy_array",
            "dtype": str(data.dtype),
            "shape": data.shape,
            "data": base64.b64encode(data.tobytes()).decode("utf-8"),
        }

    return metadata


def encode_tensor(tensor: torch.Tensor) -> dict:
    tensor_dtype = str(tensor.dtype).split(".")[-1]
    tensor_shape = str(tensor.shape)[11:-1]
    tensor_device = str(tensor.device)
    metadata = {
        "type": "tensor",
        "dtype": tensor_dtype,
        "shape": tensor_shape,
        "data": base64.b64encode(tensor.cpu().detach().numpy().tobytes()).decode(
            "utf-8"
        ),
        "device": tensor_device,
    }
    return metadata


def encode_image(image: Image.Image) -> dict:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    metadata = {
        "type": "image",
        "format": "PNG",
        "size": image.size,
        "mode": image.mode,
        "data": base64.b64encode(buffer.getvalue()).decode("utf-8"),
    }
    return metadata


def serialize_data(data):
    if isinstance(data, np.ndarray):
        return encode_numpy(data)
    elif isinstance(data, torch.Tensor):
        return encode_tensor(data)
    elif isinstance(data, Image.Image):
        return encode_image(data)
    elif isinstance(data, (list, tuple)):
        return [serialize_data(item) for item in data]
    elif isinstance(data, dict):
        return {key: serialize_data(value) for key, value in data.items()}
    else:
        return data


def decode_numpy(metadata: dict) -> np.ndarray:
    decoded_bytes = base64.b64decode(metadata["data"])
    numpy_array = np.frombuffer(decoded_bytes, dtype=np.dtype(metadata["dtype"]))
    numpy_array = numpy_array.reshape(metadata["shape"])
    return numpy_array


def decode_tensor(metadata: dict) -> torch.Tensor:
    decoded_bytes = base64.b64decode(metadata["data"])
    tensor = torch.frombuffer(
        bytearray(decoded_bytes), dtype=getattr(torch, metadata["dtype"])
    )
    tensor = tensor.reshape(eval(metadata["shape"]))
    return tensor.to(metadata["device"])


def decode_image(metadata: dict) -> Image.Image:
    try:
        decoded_bytes = base64.b64decode(metadata["data"])
        image = Image.open(io.BytesIO(decoded_bytes))

        if "size" in metadata and image.size != metadata["size"]:
            image = image.resize(metadata["size"], Image.Resampling.LANCZOS)

        if "mode" in metadata and image.mode != metadata["mode"]:
            image = image.convert(metadata["mode"])

        return image
    except (
        binascii.Error,
        KeyError,
        TypeError,
        ValueError,
        UnidentifiedImageError,
        OSError,
    ) as e:
        raise RuntimeError(f"Image decoding failed: {e}")


def deserialize_data(data):
    if isinstance(data, dict) and "type" in data:
        if data["type"] == "numpy_array":
            return decode_numpy(data)
        elif data["type"] == "tensor":
            return decode_tensor(data)
        elif data["type"] == "image":
            return decode_image(data)
    elif isinstance(data, (list, tuple)):
        return [deserialize_data(item) for item in data]
    elif isinstance(data, dict):
        return {key: deserialize_data(value) for key, value in data.items()}
    else:
        return data


# Helper functions
def unsqueeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unsqueeze the values of a dictionary.
    This converts the data to be batched of size 1.
    """
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data


def squeeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Squeeze the values of a dictionary. This removes the batch dimension.
    """
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v)
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze()
        else:
            squeezed_data[k] = v
    return squeezed_data
