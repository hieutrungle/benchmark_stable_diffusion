from optimum.graphcore.diffusers import IPUStableDiffusionPipeline
import timeout
import utils
import logger
import os
import torch
import timer


@timeout.timeout(60 * 30)
def benchmark(
    num_prompt,
    num_images_per_prompt,
    n_ipu,
    inference_replication_factor,
    prompt,
):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    cache_dir = os.path.join(current_dir, "cache")
    utils.mkdir_if_not_exists(cache_dir)

    common_ipu_config_kwargs = {
        "enable_half_partials": True,
        "executable_cache_dir": "./exe_cache",
        "inference_replication_factor": inference_replication_factor,
    }

    pipe = IPUStableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2",
        revision="fp16",
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        n_ipu=n_ipu,
        num_prompts=num_prompt,
        num_images_per_prompt=num_images_per_prompt,
        # unet_ipu_config=None,
        # text_encoder_ipu_config=None,
        # vae_ipu_config=None,
        # safety_checker_ipu_config=None,
        common_ipu_config_kwargs=common_ipu_config_kwargs,
    )
    pipe.enable_attention_slicing()

    # image_width = os.getenv("STABLE_DIFFUSION_TXT2IMG_DEFAULT_WIDTH", default=512)
    # image_height = os.getenv("STABLE_DIFFUSION_TXT2IMG_DEFAULT_HEIGHT", default=512)
    image_width = 768  # stabilityai/stable-diffusion-2
    image_height = 768  # stabilityai/stable-diffusion-2
    # pipe(
    #     prompt,
    #     # height=image_height,
    #     # width=image_width,
    #     guidance_scale=7.5,
    # )
    pipe(prompt, guidance_scale=7.5)

    with timer.Timer(logger_fn=logger.log):
        images = pipe(prompt, guidance_scale=7.5).images

    utils.save_images(
        images,
        num_prompt,
        num_images_per_prompt,
        n_ipu,
        inference_replication_factor,
        prompt,
    )
    pipe.detach_from_device()
