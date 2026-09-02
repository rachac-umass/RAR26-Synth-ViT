"""
The following is a simple example algorithm.

It is meant to run within a container.

To run the container locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

from pathlib import Path
import json
from glob import glob
import SimpleITK
import numpy as np

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def run():
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key()

    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        ("stacked-barretts-esophagus-endoscopy-images",): interface_0_handler,
    }[interface_key]

    # Call the handler
    return handler()


def interface_0_handler():
    # Read the input
    input_stacked_barretts_esophagus_endoscopy_images = load_image_file_as_array(
        location=INPUT_PATH / "images/stacked-barretts-esophagus-endoscopy",
    )
    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    # Optional: part of the Docker-container image: resources/
    # resource_dir = Path("/opt/app/resources")
    # with open(resource_dir / "some_resource.txt", "r") as f:
    #     print(f.read())

    """ Run your model here """
    # for demonstration we will use the timm classification model from the model directory
    from model.timm_model import TimmClassificationModel
    print('VitLarge_exp_kfold_v1.00')
    model = TimmClassificationModel(
        model_name="vit_large_patch16_dinov3.lvd1689m",
        num_classes=1,
        weights=RESOURCE_PATH / "best_model_dinov3-vitl16-pretrain-upperGI400k_weights_torchload.pth",
    )

    output_stacked_neoplastic_lesion_likelihoods = model.predict(input_stacked_barretts_esophagus_endoscopy_images)


    # Save your output
    write_json_file(
        location=OUTPUT_PATH / "stacked-neoplastic-lesion-likelihoods.json",
        content=output_stacked_neoplastic_lesion_likelihoods,
    )

    return 0


def get_interface_key():
    # The inputs.json is a system generated file that contains information about
    # the inputs that interface with the algorithm
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))

0
def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
