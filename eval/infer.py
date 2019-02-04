

# make this import cofnigurable as something you pass to run
import torch
from utils.util import load_image
from PIL import ImageDraw, ImageFont

class ImageInference:

    def __init__(self, model, transforms):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        
        self.transforms = transforms

    def infer(self, path):
        image = load_image(path)

        image_t, _ = self.transforms.apply(image, None)
        label, conf = self.model.predict( image_t.to(self.device) )

        return label, conf


    def draw(self, path, label, conf):
        
        image = load_image(path)
        draw = ImageDraw.Draw(image)        
        font = ImageFont.truetype('utils/Verdana.ttf', 15)
        draw.text((0,0), "%s (%.1f%%)"%(label, 100*conf),(255,0,255), font)
        image.save(path.split('.')[0] + '_pred.png')
