"""Offline, raw-mask InSPyReNet inference for Sprite Studio frame lists.

The server protocol uses one JSON request per stdin line and only JSON events
on stdout. Diagnostics always go to stderr, so Rust can safely supervise a
long-lived model process.
"""
import argparse
import concurrent.futures
import importlib.metadata
import json
import os
import sys


BASE_SIZE = (1024, 1024)


def load_model(checkpoint, device):
    if not os.path.isfile(checkpoint):
        raise RuntimeError("InSPyReNet checkpoint is not a local file")
    import torch

    # Some ROCm builds pair torch with a torchvision wheel whose C++ extension
    # cannot load. Retry only its known missing-NMS-registration failure; the
    # Swin model never calls NMS.
    try:
        import torchvision  # noqa: F401
    except RuntimeError as error:
        if "operator torchvision::nms does not exist" not in str(error):
            raise
        for name in list(sys.modules):
            if name == "torchvision" or name.startswith("torchvision."):
                del sys.modules[name]
        library = torch.library.Library("torchvision", "DEF")
        library.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
    from transparent_background.InSPyReNet import InSPyReNet_SwinB

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("InSPyReNet cuda was requested but PyTorch CUDA is unavailable")
    selected = "cuda:0" if device == "cuda" or (device == "auto" and torch.cuda.is_available()) else "cpu"
    model = InSPyReNet_SwinB(depth=64, pretrained=False, base_size=list(BASE_SIZE), threshold=None)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return model.eval().to(selected), selected, torch


def write_event(value):
    print(json.dumps(value, separators=(",", ":"), sort_keys=True), flush=True)


def runtime_identity(device):
    import cv2
    import numpy
    import torch

    def installed_version(*names):
        for name in names:
            try:
                return importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        return None

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("InSPyReNet cuda was requested but PyTorch CUDA is unavailable")
    selected = "cuda" if device == "cuda" or (device == "auto" and torch.cuda.is_available()) else "cpu"
    identity = {
        "backend": selected,
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": getattr(torch.version, "cuda", None),
        "transparent_background": importlib.metadata.version("transparent-background"),
        "python": sys.version,
        "numpy": installed_version("numpy") or numpy.__version__,
        "opencv": installed_version(
            "opencv-python",
            "opencv-python-headless",
            "opencv-contrib-python",
            "opencv-contrib-python-headless",
        )
        or cv2.__version__,
    }
    if selected == "cuda":
        identity["gpu_name"] = torch.cuda.get_device_name(0)
        identity["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    return identity


def check_sources(sources, expected_count, max_frames):
    if not isinstance(sources, list) or len(sources) != expected_count or not all(isinstance(path, str) for path in sources):
        raise RuntimeError("frame list does not match the approved source frames")
    if not 0 < expected_count <= max_frames:
        raise RuntimeError("expected source frame count is outside supported bounds")


def infer_sources(model, device, torch, sources, output, max_width, max_height, request=None, service=False):
    import cv2
    import numpy as np

    def prepare_source(source):
        bgra = cv2.imread(source, cv2.IMREAD_UNCHANGED)
        if bgra is None or bgra.dtype != np.uint8 or bgra.ndim != 3 or bgra.shape[2] not in (3, 4):
            raise RuntimeError("frame list input is not an RGB/RGBA PNG: " + source)
        height, width = bgra.shape[:2]
        if not width or not height or width > max_width or height > max_height:
            raise RuntimeError("decoded frame exceeds supported dimensions")
        rgb = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2RGB)
        # Keep the established one-frame FP32/no_grad semantics. In particular,
        # the upstream model normalizes predictions over a whole batch, so
        # batching would change masks.
        resized = cv2.resize(rgb, BASE_SIZE[::-1], interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        resized = (resized - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return width, height, resized

    masks = os.path.join(output, "masks")
    os.makedirs(masks, exist_ok=True)
    dimensions = None
    # One CPU preprocessor and one prepared frame bound input memory while the
    # GPU runs the preceding one-frame forward pass. Do not batch: upstream
    # saliency normalization aggregates batch values.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as preparation:
        prepared = preparation.submit(prepare_source, sources[0])
        for index, source in enumerate(sources):
            width, height, resized = prepared.result()
            if index + 1 < len(sources):
                prepared = preparation.submit(prepare_source, sources[index + 1])
            if dimensions is None:
                dimensions = (width, height)
            elif dimensions != (width, height):
                raise RuntimeError("source frame dimensions changed during decode")
            tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).to(device)
            with torch.no_grad():
                prediction = model(tensor)
                prediction = torch.nn.functional.interpolate(prediction, (height, width), mode="bilinear", align_corners=True)
            mask = prediction[0, 0].detach().cpu().numpy()
            if mask.shape != (height, width) or not np.isfinite(mask).all():
                raise RuntimeError("InSPyReNet returned an invalid saliency mask")
            mask = np.clip(mask * 255.0, 0.0, 255.0).astype(np.uint8)
            filename = ("mask-%06d.png" % index) if service else ("%04d.png" % index)
            target = os.path.join(masks, filename)
            temporary = os.path.join(masks, ".mask-%06d.tmp.png" % index)
            if not cv2.imwrite(temporary, mask):
                raise RuntimeError("could not write source-sized model mask")
            os.replace(temporary, target)
            if service:
                write_event({"event": "mask", "request": request, "index": index})
                # Rust only acknowledges after the bounded downstream handoff.
                acknowledgement = sys.stdin.readline()
                if not acknowledgement:
                    raise RuntimeError("mask consumer closed")
                acknowledgement = json.loads(acknowledgement)
                if acknowledgement != {"event": "ack", "request": request, "index": index}:
                    raise RuntimeError("invalid mask acknowledgement")
    return dimensions


def serve(args):
    model, device, torch = load_model(args.checkpoint, args.device)
    write_event({"event": "ready", "runtime": runtime_identity(args.device)})
    for line in sys.stdin:
        request = 0
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError("request is not an object")
            request = value.get("request")
            sources = value.get("frames")
            output = value.get("output")
            if not isinstance(request, int) or not isinstance(output, str):
                raise RuntimeError("request is invalid")
            check_sources(sources, len(sources) if isinstance(sources, list) else 0, args.max_frames)
            dimensions = infer_sources(
                model, device, torch, sources, output, args.max_width, args.max_height, request, True
            )
            write_event({
                "event": "complete",
                "request": request,
                "frame_count": len(sources),
                "width": dimensions[0],
                "height": dimensions[1],
            })
        except Exception as error:
            write_event({"event": "error", "request": request, "message": str(error)})


def one_shot(args):
    if not all((args.frames_list, args.output, args.manifest, args.expected_frame_count, args.max_frames, args.max_width, args.max_height)):
        raise RuntimeError("frame inference requires frame list, output, manifest, and bounds")
    with open(args.frames_list, encoding="utf-8") as handle:
        sources = json.load(handle)
    check_sources(sources, args.expected_frame_count, args.max_frames)
    model, device, torch = load_model(args.checkpoint, args.device)
    dimensions = infer_sources(model, device, torch, sources, args.output, args.max_width, args.max_height)
    temporary = args.manifest + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump({"frame_count": len(sources), "width": dimensions[0], "height": dimensions[1]}, handle, sort_keys=True)
    os.replace(temporary, args.manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--runtime-identity", action="store_true")
    parser.add_argument("--frames-list")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    parser.add_argument("--manifest")
    parser.add_argument("--device", required=True, choices=("auto", "cpu", "cuda"))
    parser.add_argument("--expected-frame-count", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-width", type=int)
    parser.add_argument("--max-height", type=int)
    args = parser.parse_args()
    if args.runtime_identity:
        write_event(runtime_identity(args.device))
    elif args.check:
        model, _, _ = load_model(args.checkpoint, args.device)
        del model
    elif args.server:
        if not all((args.max_frames, args.max_width, args.max_height)):
            raise RuntimeError("server inference requires bounds")
        serve(args)
    else:
        one_shot(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("InSPyReNet runner failed: %s" % error, file=sys.stderr)
        raise
