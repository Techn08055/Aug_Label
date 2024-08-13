from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw
import requests
import torch
import matplotlib.pyplot as plt  
import matplotlib.patches as patches  

def load_model():
    """Loads Florence-2-large model and its processor

    Returns:
        model (AutoModelForCausalLM): Florence-2-large model
        processor (AutoProcessor): Processor for the model
    """
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    return model, processor

def run_example(model,image, processor,task_prompt, text_input=None):
    """Runs an example using the model and processor

    Args:
        model (AutoModelForCausalLM): The model to be used for the example
        image (PIL.Image): The image to be used for the example
        processor (AutoProcessor): The processor to be used for the example
        task_prompt (str): The task prompt for the example
        text_input (str, optional): The text input for the example. Defaults to None.

    Returns:
        dict: The parsed answer from the example
    """
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )

    return parsed_answer

def convert_to_od_format(data):  
    """Converts a dictionary with 'bboxes' and 'bboxes_labels' into a dictionary with separate 'bboxes' and 'labels' keys.  
  
    Parameters:  
    - data: The input dictionary with 'bboxes', 'bboxes_labels', 'polygons', and 'polygons_labels' keys.  
  
    Returns:  
    - A dictionary with 'bboxes' and 'labels' keys formatted for object detection results.  
    """  
    # Extract bounding boxes and labels  
    bboxes = data.get('bboxes', [])  
    labels = data.get('bboxes_labels', [])  
      
    # Construct the output format  
    od_results = {  
        'bboxes': bboxes,  
        'labels': labels  
    }  
      
    return od_results  

def write_image(image, data):
    """Draws bounding boxes on an image

    Args:
        image (PIL.Image): The image to draw on
        data (dict): The data containing the bounding box coordinates and labels
    """
    for bbox, label in zip(data['bboxes'], data['labels']):  
        x1, y1, x2, y2 = bbox 
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle(((x1, y1),(x2, y2)), fill=(200, 100, 0, 127),outline=(0, 0, 0, 127), width=3)
        # draw.rectangle(((280, 10), (10b10, 706)), outline=(0, 0, 0, 127), width=3)
        image.save('orange-cat.png')
    

def main():
    """Runs the main function

    This function loads the model and processor, runs an example, converts the results to object detection format, and writes the image with bounding boxes
    """
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    image = Image.open(requests.get(url, stream=True).raw)
    model, processor = load_model()
    task_prompt = '<OPEN_VOCABULARY_DETECTION>'
    results = run_example(model,image, processor,task_prompt, text_input="a green car")
    print(results)
    bbox_results  = convert_to_od_format(results['<OPEN_VOCABULARY_DETECTION>'])
    print(bbox_results)
    write_image(image, bbox_results)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
if __name__ == '__main__':
    main()
