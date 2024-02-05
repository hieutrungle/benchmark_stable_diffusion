from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch


def benchmark(prompt):
    model_id = "stabilityai/stable-diffusion-2"

    # Use the Euler scheduler here instead
    scheduler = EulerDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler", cache_dir="./cache"
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, scheduler=scheduler, torch_dtype=torch.float16, cache_dir="./cache"
    )
    pipe = pipe.to("cuda")

    image = pipe(prompt).images[0]

    image.save("astronaut_rides_horse.png")


def benchmark_single(
    num_prompt,
    num_images_per_prompt,
    n_ipu,
    inference_replication_factor,
    prompt,
):
    model_id = "stabilityai/stable-diffusion-2"

    # Use the Euler scheduler here instead
    scheduler = EulerDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler", cache_dir="./cache"
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, scheduler=scheduler, torch_dtype=torch.float16, cache_dir="./cache"
    )
    pipe = pipe.to("cuda")

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]

    image.save("astronaut_rides_horse.png")
