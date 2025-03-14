from torch import nn
import torch

from model.main import DAY_COUNT


def detailed_hook(layer_name):
    def hook(module, input, output):
        print(f"\nLayer {layer_name}")
        print(f"  Input shape: ({input[0].shape[2]} × {input[0].shape[3]}) → Output shape: ({output.shape[2]} × {output.shape[3]})")
        if isinstance(module, nn.Sequential):
            for sublayer in module:
                if isinstance(sublayer, nn.ConvTranspose2d):
                    print(f"  - ConvTranspose2d: kernel = {sublayer.kernel_size}, stride = {sublayer.stride}, padding = {sublayer.padding}, In channels: {sublayer.in_channels} → Out channels: {sublayer.out_channels}")
                elif isinstance(sublayer, nn.Conv2d):
                    print(f"  - Conv2d: kernel = {sublayer.kernel_size}, stride = {sublayer.stride}, padding = {sublayer.padding}, In channels: {sublayer.in_channels} → Out channels: {sublayer.out_channels}")
                elif isinstance(sublayer, nn.Dropout2d):
                    print(f"  - Dropout2d: p = {sublayer.p}")
                else:
                    print(f"  - {type(sublayer).__name__}")    
    return hook


def get_layer_info(net, dimNoise, channelCount):
    hooks = []
    for name, layer in net.named_children():
        hooks.append(layer.register_forward_hook(detailed_hook(name)))
    try:
        sample = torch.randn(1, dimNoise, 1, 1)
        with torch.no_grad():
            net(sample)
    except:
        sample = torch.randn(1, channelCount, 24, DAY_COUNT)
        with torch.no_grad():
            net(sample)
    for hook in hooks:
        hook.remove()