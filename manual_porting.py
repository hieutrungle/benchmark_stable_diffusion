import diffusers
import transformers
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    PNDMScheduler,
    StableDiffusionPipeline,
)
from transformers import CLIPTextModel, CLIPTokenizer
import os
import utils
import logger
import torch
import timer
from models import PipelinedVAE, PipelinedCLIPTextModel, PipelinedUnet

try:
    import poptorch
except ImportError:
    print("poptorch not installed")


def ipu_inference_options(replication_factor=1, device_iterations=1):
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


@timer.Timer(logger_fn=logger.log)
def benchmark(prompt, model_id):
    num_prompts = [2**i for i in range(0, 2)]

    for num_prompt in num_prompts:
        prompts = [prompt] * num_prompt
        logger.log(f"\nnum_prompt_{num_prompt}")
        benchmark_single(prompts, model_id)


def benchmark_single(prompts, model_id):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    cache_dir = os.path.join(current_dir, "cache")
    utils.mkdir_if_not_exists(cache_dir)

    pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir=cache_dir)
    ipu_conf = {}
    pipe.text_encoder = PipelinedCLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        use_safetensors=True,
        cache_dir=cache_dir,
    )
    pipe.text_encoder.parallelize(ipu_conf)
    # pipe.vae = PipelinedVAE.from_pretrained(
    #     model_id,
    #     subfolder="vae",
    #     use_safetensors=True,
    #     cache_dir=cache_dir,
    # )
    # pipe.vae.parallelize(ipu_conf)

    pipe(prompts, guidance_scale=7.5)
    with timer.Timer(logger_fn=logger.log):
        images = pipe(prompts, guidance_scale=7.5).images

    exit()

    # # pretrained model and scheduler
    # vae = PipelinedVAE.from_pretrained(
    #     model_id,
    #     subfolder="vae",
    #     use_safetensors=True,
    #     cache_dir=cache_dir,
    # )
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer",
        cache_dir=cache_dir,
    )
    text_encoder = PipelinedCLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        use_safetensors=True,
        cache_dir=cache_dir,
    )
    unet = PipelinedUnet.from_pretrained(
        model_id,
        subfolder="unet",
        use_safetensors=True,
        cache_dir=cache_dir,
    )
    scheduler = PNDMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        cache_dir=cache_dir,
    )

    ipu_conf = {}

    text_encoder.parallelize(ipu_conf)
    unet.parallelize(ipu_conf)
    vae.parallelize(ipu_conf)

    inference_options = ipu_inference_options(replication_factor=1, device_iterations=1)
    inf_text_encoder = poptorch.inferenceModel(text_encoder, inference_options)
    inf_unet = poptorch.inferenceModel(unet, inference_options)
    inf_vae = poptorch.inferenceModel(vae, inference_options)

    # Text embeddings
    prompt = ["a horse eats an apple in the snow"]
    height = 512
    width = 512
    num_inference_steps = 25
    guidance_scale = 7.5
    generator = torch.manual_seed(0)
    batch_size = len(prompt)

    # Text embeddings
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_embeddings = inf_text_encoder(text_inputs.input_ids)[0]

    # logger.log(f"\ntext_inputs:")
    # logger.log(text_inputs)
    # logger.log(f"tokenizer.model_max_length: {tokenizer.model_max_length}")
    # logger.log(f"text_embeddings.shape: {text_embeddings.shape}")

    # Unconditional embeddings for padding tokens
    max_length = text_inputs.input_ids.shape[-1]
    uncond_inputs = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = inf_text_encoder(uncond_inputs.input_ids)[0]

    # logger.log(f"\nuncond_inputs:")
    # logger.log(uncond_inputs)
    # logger.log(f"max_length: {max_length}")
    # logger.log(f"uncond_embeddings.shape: {uncond_embeddings.shape}")

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # random noise
    latents = torch.randn(
        (batch_size, inf_unet.config.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents * scheduler.init_noise_sigma

    # main loop
    scheduler.set_timesteps(num_inference_steps)

    # logger.log(f"scheduler.timesteps: \n{scheduler.timesteps}\n")

    for t in scheduler.timesteps:
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_inputs = torch.cat([latents] * 2)
        latent_model_inputs = scheduler.scale_model_input(latent_model_inputs, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = inf_unet(
                latent_model_inputs, t, encoder_hidden_states=text_embeddings
            ).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # compute the prev noisy sample x_t -> x_t - 1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Decode the image
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = inf_vae.decode(latents).sample

    # image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    # image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    # images = (image * 255).round().astype("uint8")
    # image = Image.fromarray(image)
    # image_dir = "./images"
    # utils.mkdir_if_not_exists(image_dir)
    # image.save(os.path.join(image_dir, "basic_deconstruct_pipeline.png"))
