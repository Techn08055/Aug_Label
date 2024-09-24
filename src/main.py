from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw
import requests
import torch
import matplotlib.pyplot as plt  
import matplotlib.patches as patches  
import sys
import os



def run_example(image, task_prompt, text_input=None):
    """Runs an example using the model and processor.

    Args:
        image (PIL.Image): The image to be used for the example.
        task_prompt (str): The task prompt for the example.
        text_input (str, optional): The text input for the example. Defaults to None.

    Returns:
        dict: The parsed answer from the example.
    """
    # Set device and torch dtype
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load the model and processor
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

    # Construct the prompt
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    # Generate the inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    # Generate the text
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # Parse the answer
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer


def convert_to_od_format(data):
    """Converts a dictionary with 'bboxes' and 'bboxes_labels' into a dictionary with separate 'bboxes' and 'labels' keys.

    Parameters:
        data (dict): The input dictionary with 'bboxes', 'bboxes_labels', 'polygons', and 'polygons_labels' keys.

    Returns:
        dict: A dictionary with 'bboxes' and 'labels' keys formatted for object detection results.
    """
    bboxes = data.get('bboxes', [])
    labels = data.get('bboxes_labels', [])

    od_results = {
        'bboxes': bboxes,
        'labels': labels
    }

    return od_results


def write_image(image, data, path, filename):
    """Draws bounding boxes on an image.

    Args:
        image (PIL.Image): The image to draw on.
        data (dict): The data containing the bounding box coordinates and labels.
    """
    file1 = open (path + '/' + filename.split('.')[0] + '.txt', 'w')
    for bbox, label in zip(data['bboxes'], data['labels']):
        file1.write(str(label) + " " + str(int(bbox[0])) + " " + str(int(bbox[1])) + " " + str(int(bbox[2])) + " " + str(int(bbox[3])) + "\n")
    file1.close()


def main():
    """Runs the main function.

    This function loads the model and processor, runs an example, converts the results to object detection format, and writes the image with bounding boxes.
    """
    # Load the image
    path = sys.argv[1]
    text_input = sys.argv[2]
    # text_input = text_input.split(',')
    for filename in os.listdir(path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image = Image.open(path + "/" + filename)
            # Run the example
            task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
            results = run_example(image, task_prompt, text_input)
            # Convert the results to object detection format
            # bbox_results = convert_to_od_format(results['<PHRASE_GROUNDING>'])
            print(results)
            # Write the image with bounding boxes
            write_image(image, results['<CAPTION_TO_PHRASE_GROUNDING>'], path, filename)


if __name__ == '__main__':
    main()

