
from diffusers import DDPMPipeline
from pathlib import Path
import os

def load_ddpm_model(model_path:str,
                    device:str="cuda" # device set to cuda by default
                    ) -> DDPMPipeline:
  """
  Parameters
    model_path: folder that contains scheduler, unet, and model_index.json is stored
  """
  ddpm = DDPMPipeline.from_pretrained(model_path)
  print(f"[INFO] Model loaded from {model_path}")
  ddpm = ddpm.to(device) # send to cuda

  return ddpm

def ddpm_generate_imgs(model:DDPMPipeline=None,
                       num_images:int = 1, 
                       save_path:str="./generated_images") -> None:

  """
  Parameters
    model: ddpm model
    num_images: number of images to be generated
    save_path: where the generated images will be saved

  """
  
  assert model is not None, "[ERROR] Pass a model with correct type."

  save_path = Path(save_path)

  # make folder if it doesnt exist
  if not save_path.exists():
    print(f"[INFO] Save folder does not exist, creating folder...")
    os.makedirs(save_path, exist_ok=True)
    print(f"[INFO] {save_path} created")

  print(f"[INFO] Generated images will be saved at {save_path}")

  # generate images
  for num in range(num_images):
      image = ddpm().images[0]
      image.save(save_path / f"{num}.png")
