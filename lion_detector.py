import time
import torch
import torchvision
from PIL import Image

from module import detection
from tempfile import SpooledTemporaryFile
import logging


to_tensor = torchvision.transforms.ToTensor()


class LionDetector:
    def __init__(self, model_path, cuda_wanted=False):
        # Init modules
        print('Loading checkpoint from hard drive... ', end='', flush=True)
        print(f"CUDA desired={cuda_wanted}")
        has_cuda = torch.cuda.is_available()
        print("CUDA available?=", has_cuda)
        self.device = 'cuda' if has_cuda and cuda_wanted else 'cpu'
        print(f"Running inference on {self.device} device")
        print('Building model and loading checkpoint into it... ', end='', flush=True)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.label_names = checkpoint['label_names']
        model = detection.fasterrcnn_resnet50_fpn(
            num_classes=len(self.label_names) + 1, pretrained_backbone=False
        )
        model.to(self.device)
        model.load_state_dict(checkpoint['model'])
        self.model = model.eval()
        print('Init done.')

    def detect(self, image_path, image_name, conf_threshold):
        with torch.no_grad():
            logging.info("Loading image...")
            #print('Loading image... ', end='', flush=True)
            pil_image = Image.open(image_path)
            img_mode = pil_image.mode
            img_size = pil_image.size
            image = to_tensor(pil_image).to(self.device)
            logging.info("Running image through model...")
            #print('Running image through model... ', end='', flush=True)
            tic = time.time()
            outputs = self.model([image])
            toc = time.time()
            time_taken = toc - tic
            logging.info("Done in - {0} seconds".format(str(time_taken)))
            #print(f'Done in {toc - tic:.2f} seconds!')
            logging.info("Saving results to file... ")
            #print(f'Saving results to file... ', end='', flush=True)
            image_dict = {'boxes': []}
            for i, score in enumerate(outputs[0]['scores']):
                if score > conf_threshold:
                    box = outputs[0]['boxes'][i]
                    label = outputs[0]['labels'][i]
                    image_dict['boxes'].append({
                        'conf': float(score),
                        'class': int(label),
                        'ROI': box.tolist()
                    })
            if type(image_path) != SpooledTemporaryFile:
                image_dict['path'] = image_path
            image_dict['size'] = img_size
            image_dict['depth'] = img_mode
            image_dict['name'] = image_name
            logging.info("Detection process completed...")
        return image_dict, time_taken
