"""PyTorch adaptation of training-free RTC (Physical Intelligence, MIT).

Reference: real-time-chunking-kinetix/src/model.py, commit
9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b, get_prefix_weights/realtime_action.
Third-party notice: docs/third_party/rtc_LICENSE.txt.
"""
import math

import torch

from .config import RTCError


def prefix_weights(delay, overlap, horizon, *, device=None):
    if not 0 <= delay <= overlap <= horizon:
        raise RTCError('Need 0 <= delay <= overlap <= horizon for RTC masking')
    indices = torch.arange(horizon, device=device, dtype=torch.float32)
    c = ((delay-1-indices)/(overlap-delay+1)+1).clamp(0, 1)
    weights = c*torch.expm1(c)/(math.e-1)
    return torch.where(indices < overlap, weights, 0)


def guided_noise_velocity(sample, velocity_fn, sigma, target, weights, beta):
    """Return v_sigma - guidance*VJP for Zero-WAM's decreasing-sigma flow.

    The reference integrates tau=1-sigma upward and predicts v_tau=-v_sigma.
    Thus its x+(1-tau)*v_tau denoiser is x-sigma*v_sigma here, and the
    correction changes sign when passed to Zero-WAM's decreasing-sigma Euler.
    Model parameters stay frozen; only the noisy action input requires grad.
    """
    if torch.is_inference_mode_enabled():
        raise RTCError('RTC VJP requires inference_mode disabled')
    sigma = float(sigma)
    if not 0 < sigma <= 1 or not math.isfinite(beta) or beta <= 0:
        raise RTCError('RTC guidance requires 0<sigma<=1 and positive finite beta')
    with torch.enable_grad():
        x = sample.detach().clone().requires_grad_(True)
        velocity = velocity_fn(x)
        clean = x.float()-sigma*velocity.float()
        error = (target.float()-clean.detach())*weights
        correction, = torch.autograd.grad(clean, x, grad_outputs=error)
    tau = 1-sigma
    gain = beta if tau == 0 else min(beta, (tau*tau+sigma*sigma)/(tau*sigma))
    result = velocity.detach().float()-gain*correction.detach().float()
    if not torch.isfinite(result).all().item():
        raise RTCError('RTC produced nonfinite guided velocity')
    return result.to(sample.dtype)
