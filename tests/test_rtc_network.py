"""Real WebSocket/queue smoke test with a tiny CPU Transformer, no accelerator."""
import asyncio
import threading
import time

import numpy as np
import torch
import websockets.asyncio.server

from evaluation.aloha.rtc_client import RTCPolicy, WebsocketRTCTransport
from wan_va.modules.icl_model import WanICLTransformer3DModel
from wan_va.rtc.server import RTCSession
from wan_va.utils.Simple_Remote_Infer.deploy.websocket_policy_server import WebsocketPolicyServer
from test_rtc import _image, _rtc_model


def test_actual_websocket_guided_queue_swaps_with_tiny_cpu_transformer():
    class Policy:
        def infer(self, message):
            if message.get('reset'):
                model, session = _rtc_model()
                model.dtype = torch.bfloat16
                model.target_prompt_embeds = model.target_prompt_embeds.to(model.dtype)
                model._encode_initial_obs = lambda _: torch.zeros(1, 48, 1, 1, 1, dtype=model.dtype)
                model._encode_obs = lambda obs: torch.zeros(1, 48, len(obs['obs'])//4, 1, 1, dtype=model.dtype)
                model.transformer = WanICLTransformer3DModel(
                    patch_size=(1, 1, 1), num_attention_heads=2, attention_head_dim=18,
                    in_channels=48, out_channels=48, action_dim=4, text_dim=8,
                    freq_dim=4, ffn_dim=16, num_layers=1, rope_max_seq_len=64,
                    action_inner_dim=36, action_ffn_dim=16, attn_window=4,
                    enable_mcp=False, num_mcp_modules=1, mcp_hidden_collect_layers=(0,),
                ).to(dtype=model.dtype).eval().requires_grad_(False)
                model.transformer.rtc_indexed_prune = True
                from dataclasses import replace
                self.session = RTCSession(model, replace(session.config, control_hz=10, initial_latency=.3))
                return dict(enable_rtc=True, image_size=[2, 2], rtc=self.session.metadata())
            return self.session.infer(message)

    ready, stop = threading.Event(), threading.Event()
    address = []
    errors = []
    policy_server = WebsocketPolicyServer(Policy())
    async def serve():
        async with websockets.asyncio.server.serve(policy_server._handler, '127.0.0.1', 0,
                compression=None, max_size=None, ping_interval=None) as server:
            address.append(server.sockets[0].getsockname()[1])
            ready.set()
            while not stop.is_set():
                await asyncio.sleep(.01)
    def run():
        try:
            asyncio.run(serve())
        except Exception as exc:
            errors.append(exc)
            ready.set()
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(timeout=5) and not errors
    client = RTCPolicy(WebsocketRTCTransport(port=address[0]))
    try:
        started = client.start(_image(), dict(prompt='test'))
        assert len(started['warmup_latencies']) == 3
        deadline = time.monotonic()
        issued = []
        for tick in range(24):
            time.sleep(max(0, deadline-time.monotonic()))
            issued.append(client.get_action(_image(tick)))
            deadline += .1
        assert np.isfinite(issued).all()
        assert len(client.measurements) == 2
        assert all(m['server']['guidance_applied'] for m in client.measurements)
        assert all(m['server']['horizon_actions'] == 16 for m in client.measurements)
        assert [m['action_start'] for m in client.measurements] == [8, 16]
    finally:
        client.close()
        stop.set()
        thread.join(timeout=5)
    assert not thread.is_alive() and not errors
