import click
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
print(device)


@click.group()
def cli():
    pass



@cli.command()
@click.option('--model', type=str, required=True)
@click.option('--image', type=str, required=True)
@click.option('--zoom', type=int, required=True)
def test(model, image, zoom):
    img  = Image.open(image).convert('YCbCr')
    img = img.resize((int(img.size[0]*zoom), int(img.size[1]*zoom)), Image.BICUBIC)

    y, cb, cr = img.split()
    img_tensor = transforms.ToTensor()
    input = img_tensor(y).view(1, -1, y.size[1], y.size[0])
    model = torch.load(model).to(device)
    input = input.to(device)

    out = model(input)
    out = out.cpu()
    out_img_y = out[0].detach().numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img = Image.merge('YCbCr', [out_img_y, cb, cr]).convert('RGB')
    out_img.save(f"zoomed_image.jpg")


if __name__ == '__main__':
    cli()
