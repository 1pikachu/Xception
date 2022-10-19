import xception
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
import os
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--precision', default="float32", type=str, help='precision')
    parser.add_argument('--channels_last', default=1, type=int, help='Use NHWC or not')
    parser.add_argument('--jit', action='store_true', default=False, help='enable JIT')
    parser.add_argument('--profile', action='store_true', default=False, help='collect timeline')
    parser.add_argument('--num_iter', default=200, type=int, help='test iterations')
    parser.add_argument('--num_warmup', default=20, type=int, help='test warmup')
    parser.add_argument('--device', default='cpu', type=str, help='cpu, cuda or xpu')
    parser.add_argument('--nv_fuser', action='store_true', default=False, help='enable nv fuser')
    parser.add_argument('--dataset', default="/home2/pytorch-broad-models/imagenet/raw/")
    parser.add_argument('--image_size', default=224, type=int)
    args = parser.parse_args()
    print(args)
    return args

def inference(args, val_loader, model):
    for i, (image, target) in enumerate(val_loader):
        elapsed = time.time()
        model(image)
        elapsed = time.time() - elapsed
        print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)

def main():
    args = parse_args()

    if args.device == "xpu":
        import intel_extension_for_pytorch
    elif args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    model = xception.xception()
    model = model.to(args.device)

    # dataloader
    valdir = os.path.join(args.dataset, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    val_transforms = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        normalize,
    ])
    print('Using image size', args.image_size)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, val_transforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=1, pin_memory=True)

    inference(args, val_loader, model)


if __name__ == "__main__":
    main()
