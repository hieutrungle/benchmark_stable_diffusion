from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import os
import utils
import timer
import logger
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
import accelerate
from DeepCache import DeepCacheSDHelper
import tomesd


def benchmark(prompt, model_id):

    # Optimization
    torch.backends.cudnn.benchmark = True

    # Benchmarking
    num_devices = [2**i for i in range(1)]
    num_images_per_prompts = [2**i for i in range(10)]
    inference_replication_factor = 1

    for num_prompt in range(1, 2):
        for num_images_per_prompt in num_images_per_prompts:
            prompts = [prompt] * num_images_per_prompt * num_prompt
            for num_device in num_devices:
                logger.log(
                    f"\nnum_device_{num_device}_num_prompt_{num_prompt}_num_images_per_prompt_{num_images_per_prompt}"
                )
                benchmark_single(prompts, model_id)


def benchmark_single(prompts, model_id):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    cache_dir = os.path.join(current_dir, "cache")
    utils.mkdir_if_not_exists(cache_dir)

    # Use the Euler scheduler here instead
    scheduler = EulerDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler", cache_dir=cache_dir
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        use_safetensors=True,
        cache_dir=cache_dir,
    )
    pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)
    pipe.text_encoder.to(memory_format=torch.channels_last)

    pipe = pipe.to("cuda")
    # pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    helper = DeepCacheSDHelper(pipe=pipe)
    helper.set_params(
        cache_interval=3,
        cache_branch_id=0,
    )
    helper.enable()

    tomesd.apply_patch(pipe, ratio=0.5)

    with timer.Timer(logger_fn=logger.log):
        images = pipe(prompts).images
