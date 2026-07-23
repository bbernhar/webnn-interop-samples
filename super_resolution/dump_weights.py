"""Dump ESPCN super-resolution-10 weights to a flat float32 blob + JSON manifest.

The HTML fetches sr10_weights.bin (raw little-endian float32, concatenated in the
order listed below) and sr10_weights.json (name/shape/offset/count) and feeds each
tensor to MLGraphBuilder.constant().
"""
import json
import numpy as np
import onnx
from onnx import numpy_helper

ORDER = [
    "conv1.weight", "conv1.bias",
    "conv2.weight", "conv2.bias",
    "conv3.weight", "conv3.bias",
    "conv4.weight", "conv4.bias",
]

m = onnx.load("super-resolution-10.onnx")
inits = {init.name: numpy_helper.to_array(init) for init in m.graph.initializer}

manifest = []
chunks = []
offset = 0  # in float32 elements
for name in ORDER:
    arr = np.ascontiguousarray(inits[name], dtype="<f4")
    flat = arr.reshape(-1)
    manifest.append({
        "name": name,
        "shape": list(arr.shape),
        "offset": offset,
        "count": int(flat.size),
    })
    chunks.append(flat)
    offset += flat.size

blob = np.concatenate(chunks).astype("<f4")
blob.tofile("sr10_weights.bin")

with open("sr10_weights.json", "w", encoding="utf-8") as f:
    json.dump({"dtype": "float32", "total": int(blob.size), "tensors": manifest}, f, indent=2)

print(f"Wrote sr10_weights.bin ({blob.nbytes} bytes) and sr10_weights.json")
for t in manifest:
    print(f"  {t['name']:14} shape={t['shape']} offset={t['offset']} count={t['count']}")
