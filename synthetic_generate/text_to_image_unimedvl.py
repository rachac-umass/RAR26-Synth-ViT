# Text-to-medical-image generation with General-Medical-AI/UniMedVL
#
# UniMedVL (https://github.com/uni-medical/UniMedVL) is a Bagel-based unified
# medical vision-language model. It is not a plain diffusers pipeline: text-to-image
# generation goes through its own `InterleaveInferencer`, driven by a text-only
# input list with `understanding_output=False`.
#
# Setup (from the UniMedVL repo README):
#   git clone https://github.com/uni-medical/UniMedVL.git
#   cd UniMedVL
#   conda env create -f codes/environment.yaml
#   conda activate unimedvl
#
# Download the checkpoint (config.json, llm_config.json, vit_config.json,
# tokenizer files, ae.safetensors, ema_bf16.safetensors) from:
#   https://huggingface.co/General-Medical-AI/UniMedVL
#
# Then set ROOT and MODEL_PATH below and run:
#   python text_to_image_unimedvl.py

import sys
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

import gc
import shutil
from PIL import Image
import numpy as np
import torch

from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights, dispatch_model
from safetensors.torch import load_file, save_file

# --- Path to the `codes/` directory of the cloned UniMedVL repo ---
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "UniMedVL_repo", "codes")
sys.path.append(ROOT)

# --- Path to the downloaded UniMedVL checkpoint directory ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unimedvl_checkpoint")

from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from modeling.unimedvl import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def convert_checkpoint_to_bf16(input_path: str, output_path: str) -> bool:
    if not os.path.exists(input_path):
        return False
    state_dict = load_file(input_path, device="cpu")
    first_key = next(iter(state_dict))
    if state_dict[first_key].dtype == torch.bfloat16:
        if input_path != output_path:
            shutil.copy(input_path, output_path)
        return True
    bf16_state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
    del state_dict
    gc.collect()
    save_file(bf16_state_dict, output_path)
    del bf16_state_dict
    gc.collect()
    return True


def load_unimedvl(model_path: str, target_device: str = "cuda:0", max_mem_per_gpu: str = "40GiB"):
    """Load UniMedVL (Bagel architecture) and wrap it in an InterleaveInferencer."""
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
    vae_model = vae_model.cpu().to(torch.bfloat16)

    config = BagelConfig(
        visual_gen=True, visual_und=True,
        llm_config=llm_config, vit_config=vit_config, vae_config=vae_config,
        vit_max_num_patch_per_side=70, connector_act="gelu_pytorch_tanh",
        latent_patch_size=2, max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config, vae_model=vae_model)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 32, 16)
    vit_transform = ImageTransform(980, 387, 14)

    # Resolve checkpoint, converting to bf16 on first run for faster loading.
    bf16_path = os.path.join(model_path, "ema_bf16.safetensors")
    fp32_path = os.path.join(model_path, "ema.safetensors")
    if os.path.exists(bf16_path):
        checkpoint_path = bf16_path
    elif os.path.exists(fp32_path):
        convert_checkpoint_to_bf16(fp32_path, bf16_path)
        checkpoint_path = bf16_path
    else:
        raise FileNotFoundError(f"No ema/ema_bf16 checkpoint found under {model_path}")

    cpu_device_map = {name: "cpu" for name, _ in model.named_parameters()}
    model = load_checkpoint_and_dispatch(
        model, checkpoint=checkpoint_path, device_map=cpu_device_map,
        offload_buffers=False, dtype=torch.bfloat16, force_hooks=False,
    )
    torch.cuda.empty_cache()
    gc.collect()

    if torch.cuda.is_available():
        device_map = infer_auto_device_map(
            model, max_memory={0: max_mem_per_gpu},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        model = dispatch_model(model, device_map=device_map)
        vae_model = vae_model.to(device=target_device, dtype=torch.bfloat16)

    model = model.eval()

    inferencer = InterleaveInferencer(
        model=model, vae_model=vae_model, tokenizer=tokenizer,
        vae_transform=vae_transform, vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    return inferencer


def generate_image(
    inferencer: InterleaveInferencer,
    prompt: str,
    height: int = 1024,
    width: int = 1024,
    think: bool = False,
    cfg_text_scale: float = 4.0,
    cfg_img_scale: float = 1.5,
    num_timesteps: int = 50,
    timestep_shift: float = 3.0,
    seed: int = 42,
) -> Image.Image:
    """Pure text -> medical image generation (no input image)."""
    if seed is not None:
        set_seed(seed)

    result = inferencer(
        text=prompt,
        think=think,                 # True: model plans in <think>...</think> before generating
        understanding_output=False,  # False = image-generation task (True would be VQA/text output)
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        cfg_interval=[0.0, 1.0],
        cfg_renorm_type="text_channel",
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        image_shapes=(height, width),  # (H, W); use multiples of 16
    )
    return result["image"]




if __name__ == "__main__":
    inferencer = load_unimedvl(MODEL_PATH)

    prompt = (
        "Generate a good resolution endoscopy image with barrett esophagus and red patch darker than background on barrett esophagus section."
    )

    print(prompt)
    n_images = 10
    base_seed = 32
    rng = np.random.default_rng()
    for i in range(n_images):
        image = generate_image(
            inferencer,
            prompt=prompt,
            height=336,
            width=336,
            cfg_img_scale = 1.0,
            cfg_text_scale=12.0,
            seed=base_seed + i,
            num_timesteps=500,
            think=True,
        )
        image_id = rng.integers(0, 2**63, dtype=np.int64)
        out_path = f"/home/chandraharsha.rachabathuni-umw/Competitions/RARE26_challenge/syn_images/generated_medical_instrument_typeIIscc_{image_id}.png"
        image.save(out_path)
        print(f"Saved generated image to {out_path}")
