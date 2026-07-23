"""MEMO inference: load_models / generate_video interface.

Wraps the MEMO (Memory-Guided Diffusion) pipeline so the worker can call:
    models = load_models(model_dir, ...)
    generate_video(models, source_image, driving_audio, save_path)
"""

import logging
import os
import tempfile

import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from tqdm import tqdm

from memo.models.audio_proj import AudioProjModel
from memo.models.image_proj import ImageProjModel
from memo.models.unet_2d_condition import UNet2DConditionModel
from memo.models.unet_3d import UNet3DConditionModel
from memo.pipelines.video_pipeline import VideoPipeline
from memo.utils.audio_utils import extract_audio_emotion_labels, preprocess_audio, resample_audio
from memo.utils.vision_utils import preprocess_image, tensor_to_video


logger = logging.getLogger("memo")
logger.setLevel(logging.INFO)


def load_models(
    model_dir: str,
    misc_model_dir: str | None = None,
    resolution: int = 512,
    weight_dtype: str = "bf16",
    enable_xformers: bool = True,
    device: str = "cuda",
) -> dict:
    """Load all MEMO models and return them in a dict.

    Args:
        model_dir: Path to MEMO checkpoint (or HuggingFace model ID like 'memoavatar/memo')
        misc_model_dir: Path to misc models (face_analysis, vocal_separator). If None, uses model_dir.
        resolution: Image resolution (default 512)
        weight_dtype: One of 'fp16', 'bf16', 'fp32'
        enable_xformers: Whether to enable xformers memory-efficient attention
        device: Device to load models on
    """
    if misc_model_dir is None:
        misc_model_dir = model_dir

    # Resolve dtype
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map.get(weight_dtype, torch.bfloat16)

    device = torch.device(device) if isinstance(device, str) else device

    # Ensure face analysis models exist
    face_analysis = os.path.join(misc_model_dir, "misc/face_analysis")
    os.makedirs(os.path.join(face_analysis, "models"), exist_ok=True)
    for model_name in [
        "1k3d68.onnx",
        "2d106det.onnx",
        "face_landmarker_v2_with_blendshapes.task",
        "genderage.onnx",
        "glintr100.onnx",
        "scrfd_10g_bnkps.onnx",
    ]:
        model_path = os.path.join(face_analysis, "models", model_name)
        if not os.path.exists(model_path):
            logger.info(f"Downloading {model_name} to {face_analysis}/models")
            os.system(
                f"wget -q -P {face_analysis}/models https://huggingface.co/memoavatar/memo/resolve/main/misc/face_analysis/models/{model_name}"
            )

    # Ensure vocal separator exists
    vocal_separator = os.path.join(misc_model_dir, "misc/vocal_separator/Kim_Vocal_2.onnx")
    if not os.path.exists(vocal_separator):
        os.makedirs(os.path.dirname(vocal_separator), exist_ok=True)
        logger.info(f"Downloading vocal separator to {vocal_separator}")
        os.system(
            f"wget -q -P {os.path.dirname(vocal_separator)} https://huggingface.co/memoavatar/memo/resolve/main/misc/vocal_separator/Kim_Vocal_2.onnx"
        )

    # Load models
    logger.info("Loading MEMO models...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device=device, dtype=dtype)
    reference_net = UNet2DConditionModel.from_pretrained(
        model_dir, subfolder="reference_net", use_safetensors=True
    )
    diffusion_net = UNet3DConditionModel.from_pretrained(
        model_dir, subfolder="diffusion_net", use_safetensors=True
    )
    image_proj = ImageProjModel.from_pretrained(
        model_dir, subfolder="image_proj", use_safetensors=True
    )
    audio_proj = AudioProjModel.from_pretrained(
        model_dir, subfolder="audio_proj", use_safetensors=True
    )

    vae.requires_grad_(False).eval()
    reference_net.requires_grad_(False).eval()
    diffusion_net.requires_grad_(False).eval()
    image_proj.requires_grad_(False).eval()
    audio_proj.requires_grad_(False).eval()

    # Enable xformers
    if enable_xformers and is_xformers_available():
        reference_net.enable_xformers_memory_efficient_attention()
        diffusion_net.enable_xformers_memory_efficient_attention()

    # Create pipeline
    noise_scheduler = FlowMatchEulerDiscreteScheduler()
    pipeline = VideoPipeline(
        vae=vae,
        reference_net=reference_net,
        diffusion_net=diffusion_net,
        scheduler=noise_scheduler,
        image_proj=image_proj,
    )
    pipeline.to(device=device, dtype=dtype)

    return {
        "pipeline": pipeline,
        "audio_proj": audio_proj,
        "face_analysis": face_analysis,
        "vocal_separator": vocal_separator,
        "device": device,
        "dtype": dtype,
        "resolution": resolution,
    }


def generate_video(
    models: dict,
    source_image: str,
    driving_audio: str,
    save_path: str,
    num_generated_frames_per_clip: int = 16,
    fps: int = 30,
    num_init_past_frames: int = 16,
    num_past_frames: int = 16,
    inference_steps: int = 20,
    cfg_scale: float = 3.5,
    seed: int = 42,
    wav2vec_model: str = "facebook/wav2vec2-base-960h",
    emotion2vec_model: str = "iic/emotion2vec_plus_large",
) -> str:
    """Generate a talking video from a source image and driving audio.

    Args:
        models: Dict returned by load_models()
        source_image: Path to reference portrait image
        driving_audio: Path to driving audio (WAV recommended)
        save_path: Output video path
        num_generated_frames_per_clip: Frames per generation clip
        fps: Output video FPS
        num_init_past_frames: Number of initial past frames
        num_past_frames: Number of past frames for context
        inference_steps: Diffusion inference steps
        cfg_scale: Classifier-free guidance scale
        seed: Random seed
        wav2vec_model: Wav2Vec model name/path
        emotion2vec_model: Emotion2Vec model name/path

    Returns:
        Path to the generated video
    """
    pipeline = models["pipeline"]
    audio_proj = models["audio_proj"]
    face_analysis = models["face_analysis"]
    vocal_separator = models["vocal_separator"]
    device = models["device"]
    dtype = models["dtype"]
    resolution = models["resolution"]

    generator = torch.manual_seed(seed)
    img_size = (resolution, resolution)

    # Preprocess image
    pixel_values, face_emb = preprocess_image(
        face_analysis_model=face_analysis,
        image_path=source_image,
        image_size=resolution,
    )

    # Preprocess audio
    cache_dir = tempfile.mkdtemp(prefix="memo_audio_")
    audio_path = resample_audio(
        driving_audio,
        os.path.join(cache_dir, "audio-16k.wav"),
    )
    audio_emb, audio_length = preprocess_audio(
        wav_path=audio_path,
        num_generated_frames_per_clip=num_generated_frames_per_clip,
        fps=fps,
        wav2vec_model=wav2vec_model,
        vocal_separator_model=vocal_separator,
        cache_dir=cache_dir,
        device=device,
    )

    # Process audio emotion
    audio_emotion, num_emotion_classes = extract_audio_emotion_labels(
        model="memoavatar/memo",
        wav_path=audio_path,
        emotion2vec_model=emotion2vec_model,
        audio_length=audio_length,
        device=device,
    )

    # Generate video clips
    video_frames = []
    num_clips = audio_emb.shape[0] // num_generated_frames_per_clip
    for t in tqdm(range(num_clips), desc="MEMO generating", unit="clip"):
        if len(video_frames) == 0:
            past_frames = pixel_values.repeat(num_init_past_frames, 1, 1, 1)
            past_frames = past_frames.to(dtype=pixel_values.dtype, device=pixel_values.device)
            pixel_values_ref_img = torch.cat([pixel_values, past_frames], dim=0)
        else:
            past_frames = video_frames[-1][0]
            past_frames = past_frames.permute(1, 0, 2, 3)
            past_frames = past_frames[0 - num_past_frames:]
            past_frames = past_frames * 2.0 - 1.0
            past_frames = past_frames.to(dtype=pixel_values.dtype, device=pixel_values.device)
            pixel_values_ref_img = torch.cat([pixel_values, past_frames], dim=0)

        pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)

        audio_tensor = (
            audio_emb[
                t * num_generated_frames_per_clip: min(
                    (t + 1) * num_generated_frames_per_clip, audio_emb.shape[0]
                )
            ]
            .unsqueeze(0)
            .to(device=audio_proj.device, dtype=audio_proj.dtype)
        )
        audio_tensor = audio_proj(audio_tensor)

        audio_emotion_tensor = audio_emotion[
            t * num_generated_frames_per_clip: min(
                (t + 1) * num_generated_frames_per_clip, audio_emb.shape[0]
            )
        ]

        pipeline_output = pipeline(
            ref_image=pixel_values_ref_img,
            audio_tensor=audio_tensor,
            audio_emotion=audio_emotion_tensor,
            emotion_class_num=num_emotion_classes,
            face_emb=face_emb,
            width=img_size[0],
            height=img_size[1],
            video_length=num_generated_frames_per_clip,
            num_inference_steps=inference_steps,
            guidance_scale=cfg_scale,
            generator=generator,
            is_new_audio=t == 0,
        )

        video_frames.append(pipeline_output.videos)

    video_frames = torch.cat(video_frames, dim=2)
    video_frames = video_frames.squeeze(0)
    video_frames = video_frames[:, :audio_length]

    tensor_to_video(video_frames, save_path, driving_audio, fps=fps)

    # Cleanup
    import shutil
    shutil.rmtree(cache_dir, ignore_errors=True)

    return save_path
