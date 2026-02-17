import torch.nn as nn
import cv2
import torch
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import numpy as np


def get_single_mask_from_video(video, target_resolution=1024, batch_size=1, target_label=4, device=None, processor=None, model=None):
    """
    Generates a tensor of binary masks for a specific semantic class from an input video using a SegFormer segmentation model.

    The function reads the input video frame by frame, resizes each frame to a square `target_resolution`,
    performs batched semantic segmentation using the provided or default SegFormer model, and returns a single tensor
    containing boolean masks for the specified target class.

    Parameters
    ----------
    video : str
        Path to the input video file.
    target_resolution : int, optional, default=1024
        The resolution (height and width) to which each video frame will be resized before segmentation.
        Output masks will have the same resolution.
    batch_size : int, optional, default=1
        Number of frames to process in a single batch during inference. Larger batches are faster on GPU but
        consume more memory.
    target_label : int, optional, default=4
        The class index to extract from the segmentation output. Only pixels with this class will be True
        in the output mask. For the default model, 4 is the label for upper-clothes (shirt).
    device : str or torch.device, optional
        The device on which to run the model ("cuda" or "cpu"). If None, automatically selects GPU if available.
    processor : transformers.SegformerImageProcessor, optional
        A pre-initialized processor for image preprocessing. If None, a default processor is loaded from the model.
    model : torch.nn.Module, optional
        A pre-loaded SegFormer semantic segmentation model. If None, the default `mattmdjaga/segformer_b2_clothes`
        model is loaded.

    Returns
    -------
    torch.BoolTensor
        A boolean tensor of shape `[num_frames, target_resolution, target_resolution]` where `True` indicates
        pixels belonging to the target class (`target_label`) and `False` otherwise.

    Notes
    -----
    - This doc was written by ChatGPT with human edits.
    """


    # Load model configs if none were passed in
    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "mattmdjaga/segformer_b2_clothes"
    if processor == None:
        processor = SegformerImageProcessor.from_pretrained(model_name)
    if model == None:
        model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(device)
    
    # Set model to evaluation mode
    model.eval()

    # Load input video as an opencv object
    cap = cv2.VideoCapture(video)

    # Buffers for storing frames and corresponding masks
    rgb_frames = []
    mask_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and make frame RGB (opencv is BGR)
        frame_512 = cv2.resize(frame, (target_resolution, target_resolution), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(frame_512, cv2.COLOR_BGR2RGB)

        rgb_frames.append(rgb)

        # Run inference on a batch of frames
        if len(rgb_frames) == batch_size:
            inputs = processor(images=rgb_frames, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # run inference 
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits  # [B, 18, H, W]

            # Upsample model output to our target_size
            logits = nn.functional.interpolate(
                logits,
                size=(target_resolution, target_resolution),
                mode="bilinear",
                align_corners=False
            )

            # Extract target_class mask
            pred_classes = logits.argmax(dim=1)          # [batch_size, height, width]
            target_masks = (pred_classes == target_label)  # boolean maks for pixels pixels with our target class

            # Append to list (after moving masks to the cpu)
            mask_list.extend([mask.cpu() for mask in target_masks])

            # Clear residual frames before starting next batch
            rgb_frames.clear()

    # For any frames remaining, make sure to process them too
    if len(rgb_frames) > 0:
        inputs = processor(images=rgb_frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Upsample model output to our target_size
        logits = nn.functional.interpolate(
            logits,
            size=(target_resolution, target_resolution),
            mode="bilinear",
            align_corners=False
        )

        # Extract target_class mask
        pred_classes = logits.argmax(dim=1)
        target_masks = (pred_classes == target_label)

        mask_list.extend([mask.cpu() for mask in target_masks])

    cap.release()

    # convert list of masks to a single tensor with size [num_frames, height, width]
    mask_tensor = torch.stack(mask_list)  # bool tensor
    
    return mask_tensor

def load_segformer_model(
    model_name="mattmdjaga/segformer_b2_clothes",
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
    model.to(device)
    model.eval()

    return processor, model, device

def write_mask_overlay_video(
    input_video,
    output_video,
    mask_tensor,
    mode="color",                 # "color" or "pattern"
    mask_color=(0, 255, 0),        # used in color mode
    alpha=0.35,                   # used in color mode
    overlay_img_path=None          # used in pattern mode
):
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read video")

    h, w = first_frame.shape[:2]

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    # Load overlay image if needed
    if mode == "pattern":
        if overlay_img_path is None:
            raise ValueError("overlay_img_path must be provided for pattern mode")
        overlay_img = cv2.imread(overlay_img_path)
        overlay_img = cv2.resize(overlay_img, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0

    while cap.isOpened() and frame_idx < mask_tensor.shape[0]:
        ret, frame = cap.read()
        if not ret:
            break

        mask = mask_tensor[frame_idx].numpy().astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if mode == "color":
            mask_255 = mask * 255
            mask_bgr = np.zeros_like(frame)
            mask_bgr[mask_255 > 0] = mask_color
            output_frame = cv2.addWeighted(
                frame, 1 - alpha,
                mask_bgr, alpha,
                0
            )

        elif mode == "pattern":
            output_frame = frame.copy()
            output_frame[mask.astype(bool)] = overlay_img[mask.astype(bool)]

        else:
            raise ValueError(f"Unknown mode: {mode}")

        writer.write(output_frame)
        frame_idx += 1

    cap.release()
    writer.release()


def write_mask_overlay_video1(
    input_video,
    output_video,
    mask_tensor,
    mode="color",
    mask_color=(0, 255, 0),
    alpha=0.35,
    overlay_img_path=None,
    padding=10  # extra pixels around detected region
):
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read video")

    h, w = first_frame.shape[:2]

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    # Load overlay image once
    if mode == "pattern":
        if overlay_img_path is None:
            raise ValueError("overlay_img_path must be provided for pattern mode")
        overlay_img_original = cv2.imread(overlay_img_path)
        if overlay_img_original is None:
            raise RuntimeError("Failed to load overlay image")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0

    while cap.isOpened() and frame_idx < mask_tensor.shape[0]:

        ret, frame = cap.read()
        if not ret:
            break

        mask = mask_tensor[frame_idx].numpy().astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if mode == "color":
            mask_255 = mask * 255
            mask_bgr = np.zeros_like(frame)
            mask_bgr[mask_255 > 0] = mask_color

            output_frame = cv2.addWeighted(
                frame, 1 - alpha,
                mask_bgr, alpha,
                0
            )

        elif mode == "pattern":

            ys, xs = np.where(mask > 0)

            # If no mask pixels found, just write original frame
            if len(xs) == 0 or len(ys) == 0:
                writer.write(frame)
                frame_idx += 1
                continue

            # Bounding box
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            # Add padding safely
            x_min = max(0, x_min - padding)
            x_max = min(w, x_max + padding)
            y_min = max(0, y_min - padding)
            y_max = min(h, y_max + padding)

            region_w = x_max - x_min
            region_h = y_max - y_min

            # Resize overlay to region size
            resized_overlay = cv2.resize(
                overlay_img_original,
                (region_w, region_h)
            )

            region_mask = mask[y_min:y_max, x_min:x_max].astype(bool)

            output_frame = frame.copy()

            output_frame[y_min:y_max, x_min:x_max][region_mask] = \
                resized_overlay[region_mask]

        else:
            raise ValueError(f"Unknown mode: {mode}")

        writer.write(output_frame)
        frame_idx += 1

    cap.release()
    writer.release()
