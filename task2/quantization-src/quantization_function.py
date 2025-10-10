import torch
import copy

def linear_quantize(weight_fp32: torch.Tensor):
    w_min, w_max = weight_fp32.min(), weight_fp32.max()
    max_abs = max(abs(w_min), abs(w_max))
    scale = max_abs / 127
    zero_point = 0
    weight_int8 = torch.clamp(torch.round(weight_fp32 / scale), -128, 127).to(torch.int8)
    return weight_int8, scale, zero_point


def linear_dequantize(weight_int8: torch.Tensor, scale: float, zero_point: int):
    return (weight_int8.float() - zero_point) * scale

def quantize_model(model: torch.nn.Module):
    quantized_model = copy.deepcopy(model)
    quant_params = {}

    with torch.no_grad():
        for name, param in quantized_model.named_parameters():
            if param.requires_grad and param.dtype == torch.float32:
                weight_int8, scale, zero_point = linear_quantize(param.data)
                param.data = linear_dequantize(weight_int8, scale, zero_point)
                quant_params[name] = (scale, zero_point)

    return quantized_model, quant_params