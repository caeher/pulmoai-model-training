"""Local slice contact sheet for inspecting native voxel candidate coordinates."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pulmoai.preprocessing.volume import normalize_hu, read_volume


def save_preview(image_path: str | Path, prediction: dict, output: str | Path) -> None:
    volume = read_volume(image_path)
    found = prediction["candidates"][:6]
    if not found:
        found = [{"center_voxel_xyz": [(n - 1) / 2 for n in volume.hu.shape[::-1]], "score": None}]
    canvas = Image.new("RGB", (512, 64 + 300 * len(found)), "black")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "PulmoAI - research only; candidate scores are not malignancy", fill="white")
    draw.text(
        (8, 30),
        "Native voxel planes; orientation is not a radiological display convention",
        fill="white",
    )
    for i, candidate in enumerate(found):
        x, y, z = candidate["center_voxel_xyz"]
        section = int(np.clip(round(z), 0, volume.hu.shape[0] - 1))
        pixels = (normalize_hu(volume.hu[section], (-1000, 400)) * 255).astype(np.uint8)
        # Correct in-plane aspect ratio for anisotropic source spacing.
        aspect = (
            pixels.shape[1]
            * volume.geometry.spacing_xyz[0]
            / (pixels.shape[0] * volume.geometry.spacing_xyz[1])
        )
        width, height = (
            (max(1, round(256 * aspect)), 256)
            if aspect <= 1
            else (480, max(1, round(480 / aspect)))
        )
        tile = Image.fromarray(pixels).convert("RGB").resize((width, height))
        top = 64 + i * 300
        canvas.paste(tile, (8, top))
        if candidate["score"] is not None:
            px = 8 + (x + 0.5) * width / pixels.shape[1]
            py = top + (y + 0.5) * height / pixels.shape[0]
            draw.ellipse((px - 6, py - 6, px + 6, py + 6), outline="red", width=2)
        text = f"z index={section}; score={candidate['score']}"
        draw.text((8, top + 270), text, fill="white")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
