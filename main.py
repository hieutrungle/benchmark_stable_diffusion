# import diffusers
# import transformers
# from diffusers import (
#     AutoencoderKL,
#     UNet2DConditionModel,
#     PNDMScheduler,
# )
# from transformers import CLIPTextModel, CLIPTokenizer
import os
import utils
import logger
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import poptorch
import matplotlib.pyplot as plt
import timer
import timeout

from models import PipelinedVAE, PipelinedCLIPTextModel, PipelinedUnet

from optimum.graphcore.diffusers import IPUStableDiffusionPipeline


def mkdir_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def ipu_validation_options(replication_factor=1, device_iterations=1):
    opts = poptorch.Options()
    opts.randomSeed(42)
    opts.deviceIterations(device_iterations)

    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement)
    )

    # Stochastic rounding not needed for validation
    opts.Precision.enableStochasticRounding(False)

    # Half precision partials for matmuls and convolutions
    opts.Precision.setPartialsType(torch.float16)

    opts.replicationFactor(replication_factor)

    # No gradient accumulation for Inference
    opts.Training.gradientAccumulation(1)

    # Return all results from IPU
    opts.outputMode(poptorch.OutputMode.All)

    # Cache compiled executable to disk
    # opts.enableExecutableCaching(CACHE_DIR)

    return opts


def save_images(
    images,
    num_prompt,
    num_images_per_prompt,
    n_ipu,
    inference_device_iteration,
    inference_replication_factor,
    prompt,
):
    num_images = len(images)

    fig, axes = plt.subplots(num_prompt, num_images_per_prompt, constrained_layout=True)
    # fig.set_size_inches(9, 9)

    for i, image in enumerate(images):
        if num_prompt == 1 and num_images_per_prompt == 1:
            ax = axes
        elif num_prompt == 1:
            ax = axes[i % num_images_per_prompt]
        else:
            ax = axes[i // num_images_per_prompt, i % num_images_per_prompt]
        ax.imshow(image)
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")

    fig.suptitle(f"prompt: {prompt}")
    # fig.tight_layout()
    # fig.subplots_adjust(wspace=0, hspace=0)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    image_dir = os.path.join(current_dir, "images")
    utils.mkdir_if_not_exists(image_dir)
    fig.savefig(
        os.path.join(
            image_dir,
            f"n_ipu_{n_ipu}_num_prompt_{num_prompt}_num_images_per_prompt_{num_images_per_prompt}_inference_device_iteration_{inference_device_iteration}_inference_replication_factor_{inference_replication_factor}.png",
        ),
        dpi=150,
    )
    plt.close()


@timeout.timeout()
def benchmark(
    num_prompt,
    num_images_per_prompt,
    n_ipu,
    inference_device_iteration,
    inference_replication_factor,
    prompt,
):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    cache_dir = os.path.join(current_dir, "cache")
    utils.mkdir_if_not_exists(cache_dir)

    common_ipu_config_kwargs = {
        "enable_half_partials": True,
        "executable_cache_dir": "./exe_cache",
        "inference_device_iterations": inference_device_iteration,
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
    pipe(
        "apple",
        height=image_height,
        width=image_width,
        guidance_scale=7.5,
    )

    with timer.Timer(logger_fn=logger.log):
        images = pipe(prompt, guidance_scale=7.5).images

    save_images(
        images,
        num_prompt,
        num_images_per_prompt,
        n_ipu,
        inference_device_iteration,
        inference_replication_factor,
        prompt,
    )
    pipe.detach_from_device()


@timer.Timer(logger_fn=logger.log)
def main():
    n_ipus = [2**i for i in range(5)]
    # num_prompts = 1
    num_images_per_prompts = [4]
    inference_device_iterations = [2**i for i in range(1)]
    inference_replication_factors = [2**i for i in range(1)]
    # num_images_per_prompts = [2**i for i in range(4)]
    # inference_device_iterations = [2**i for i in range(4)]
    # inference_replication_factors = [2**i for i in range(4)]

    prompt = "a shiba inu in a zen garden, acrylic painting"

    for num_prompt in range(1, 2):
        for inference_replication_factor in inference_replication_factors:
            for inference_device_iteration in inference_device_iterations:
                for num_images_per_prompt in num_images_per_prompts:
                    for n_ipu in n_ipus:
                        logger.log(
                            f"\nn_ipu_{n_ipu}_num_prompt_{num_prompt}_num_images_per_prompt_{num_images_per_prompt}_inference_device_iteration_{inference_device_iteration}_inference_replication_factor_{inference_replication_factor}"
                        )
                        try:
                            benchmark(
                                num_prompt,
                                num_images_per_prompt,
                                n_ipu,
                                inference_device_iteration,
                                inference_replication_factor,
                                prompt,
                            )
                        except timeout.TimeoutError:
                            logger.log(
                                f"Timeout: n_ipu_{n_ipu}_num_prompt_{num_prompt}_num_images_per_prompt_{num_images_per_prompt}_inference_device_iteration_{inference_device_iteration}_inference_replication_factor_{inference_replication_factor}"
                            )
                        except Exception as e:
                            logger.log(
                                f"Exception: {e} for n_ipu_{n_ipu}_num_prompt_{num_prompt}_num_images_per_prompt_{num_images_per_prompt}_inference_device_iteration_{inference_device_iteration}_inference_replication_factor_{inference_replication_factor}"
                            )


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Set up
    log_dir = os.path.join(current_dir, "logs")
    utils.mkdir_if_not_exists(log_dir)
    logger.configure(dir=log_dir)
    main()

    # # pretrained model and scheduler
    # vae = PipelinedVAE.from_pretrained(
    #     "CompVis/stable-diffusion-v1-4",
    #     subfolder="vae",
    #     use_safetensors=True,
    #     cache_dir=cache_dir,
    # )
    # tokenizer = CLIPTokenizer.from_pretrained(
    #     "CompVis/stable-diffusion-v1-4",
    #     subfolder="tokenizer",
    #     cache_dir=cache_dir,
    # )
    # text_encoder = PipelinedCLIPTextModel.from_pretrained(
    #     "CompVis/stable-diffusion-v1-4",
    #     subfolder="text_encoder",
    #     use_safetensors=True,
    #     cache_dir=cache_dir,
    # )
    # unet = PipelinedUnet.from_pretrained(
    #     "CompVis/stable-diffusion-v1-4",
    #     subfolder="unet",
    #     use_safetensors=True,
    #     cache_dir=cache_dir,
    # )
    # scheduler = PNDMScheduler.from_pretrained(
    #     "CompVis/stable-diffusion-v1-4",
    #     subfolder="scheduler",
    #     cache_dir=cache_dir,
    # )

    # ipu_conf = {}

    # text_encoder.parallelize(ipu_conf)
    # unet.parallelize(ipu_conf)
    # vae.parallelize(ipu_conf)

    # inf_text_encoder = poptorch.inferenceModel(text_encoder, inference_options)
    # inf_unet = poptorch.inferenceModel(unet, inference_options)
    # inf_vae = poptorch.inferenceModel(vae, inference_options)

    # # Text embeddings
    # prompt = ["a horse eats an apple in the snow"]
    # height = 512
    # width = 512
    # num_inference_steps = 25
    # guidance_scale = 7.5
    # generator = torch.manual_seed(0)
    # batch_size = len(prompt)

    # # Text embeddings
    # text_inputs = tokenizer(
    #     prompt,
    #     padding="max_length",
    #     max_length=tokenizer.model_max_length,
    #     truncation=True,
    #     return_tensors="pt",
    # )
    # with torch.no_grad():
    #     text_embeddings = inf_text_encoder(text_inputs.input_ids)[0]

    # logger.log(f"\ntext_inputs:")
    # logger.log(text_inputs)
    # logger.log(f"tokenizer.model_max_length: {tokenizer.model_max_length}")
    # logger.log(f"text_embeddings.shape: {text_embeddings.shape}")

    # # Unconditional embeddings for padding tokens
    # max_length = text_inputs.input_ids.shape[-1]
    # uncond_inputs = tokenizer(
    #     [""] * batch_size,
    #     padding="max_length",
    #     max_length=max_length,
    #     return_tensors="pt",
    # )
    # uncond_embeddings = inf_text_encoder(uncond_inputs.input_ids)[0]

    # logger.log(f"\nuncond_inputs:")
    # logger.log(uncond_inputs)
    # logger.log(f"max_length: {max_length}")
    # logger.log(f"uncond_embeddings.shape: {uncond_embeddings.shape}")

    # text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # # random noise
    # latents = torch.randn(
    #     (batch_size, inf_unet.config.in_channels, height // 8, width // 8),
    #     generator=generator,
    # )
    # latents = latents * scheduler.init_noise_sigma

    # # main loop
    # scheduler.set_timesteps(num_inference_steps)

    # logger.log(f"scheduler.timesteps: \n{scheduler.timesteps}\n")

    # for t in scheduler.timesteps:
    #     # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    #     latent_model_inputs = torch.cat([latents] * 2)
    #     latent_model_inputs = scheduler.scale_model_input(latent_model_inputs, t)

    #     # predict the noise residual
    #     with torch.no_grad():
    #         noise_pred = inf_unet(
    #             latent_model_inputs, t, encoder_hidden_states=text_embeddings
    #         ).sample

    #     # perform guidance
    #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #     noise_pred = noise_pred_uncond + guidance_scale * (
    #         noise_pred_text - noise_pred_uncond
    #     )

    #     # compute the prev noisy sample x_t -> x_t - 1
    #     latents = scheduler.step(noise_pred, t, latents).prev_sample

    # # Decode the image
    # latents = 1 / 0.18215 * latents
    # with torch.no_grad():
    #     image = inf_vae.decode(latents).sample

    # image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    # image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    # images = (image * 255).round().astype("uint8")
    # image = Image.fromarray(image)
    # image_dir = "./images"
    # utils.mkdir_if_not_exists(image_dir)
    # image.save(os.path.join(image_dir, "basic_deconstruct_pipeline.png"))
