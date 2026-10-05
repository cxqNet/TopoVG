"""Optional original-size GT/prediction images and refined heatmaps."""
import os
from pathlib import Path

def image_font(size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size=size)
    except OSError:
        return ImageFont.load_default()


def wrap_caption(text, font, max_width):
    from PIL import Image, ImageDraw
    draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    lines, line = [], ''
    for word in str(text).split():
        candidate = (line + ' ' + word).strip()
        bounds = draw.textbbox((0, 0), candidate, font=font)
        if line and bounds[2] - bounds[0] > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    lines.append(line)
    return lines


def captioned_image(image, title, expression, legend=None):
    from PIL import Image, ImageDraw
    width, height = image.size
    font = image_font(max(12, min(20, width // 36)))
    lines = wrap_caption(title, font, width - 24) + wrap_caption(expression, font, width - 24)
    if legend:
        lines.extend(wrap_caption(legend, font, width - 24))
    line_height = max(18, min(28, width // 28))
    header = 16 + len(lines) * line_height
    canvas = Image.new('RGB', (width, height + header), (24, 24, 24))
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        draw.text((12, 6 + i * line_height), line, fill='white', font=font)
    return canvas


def atomic_png(path, image):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        image.save(temporary, format='PNG')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_visualizations(sample, row, heatmap, results_root):
    """Render at original image size. Visual rendering never changes decoding."""
    from PIL import Image, ImageDraw
    import numpy as visual_np
    folder = Path(results_root) / sample['id']
    folder.mkdir(parents=True, exist_ok=True)
    with Image.open(sample['_image_path']) as source:
        original = source.convert('RGB')
    width, height = original.size
    if (width, height) != (int(sample['width']), int(sample['height'])):
        raise ValueError(f"{sample['id']}: image dimensions changed during evaluation")
    localized = original.copy()
    draw = ImageDraw.Draw(localized)
    thickness = max(2, round(min(width, height) / 250))
    font = image_font(max(12, min(20, width // 36)))
    for box, color, label in (
        (sample['bbox_xyxy'], (0, 255, 0), 'GT'),
        (row['bbox'], (255, 0, 0), 'PRED'),
    ):
        x1, y1, x2, y2 = box
        coordinates = [max(0, min(width - 1, round(x1))),
                       max(0, min(height - 1, round(y1))),
                       max(0, min(width - 1, round(x2))),
                       max(0, min(height - 1, round(y2)))]
        # A wider GT outline keeps both colors visible when the boxes coincide.
        draw.rectangle(coordinates, outline=color, width=thickness + 2 if label == 'GT' else thickness)
        tx = max(0, min(width - 60, coordinates[0]))
        label_y = coordinates[1] - 22 if label == 'GT' else coordinates[1] + 6
        ty = max(0, min(height - 22, label_y))
        bounds = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(bounds, fill=(0, 0, 0))
        draw.text((tx, ty), label, fill=color, font=font)
    title = f"{sample['id']} | IoU={row['iou']:.6f} ({100 * row['iou']:.2f}%)"
    localization = captioned_image(localized, title, sample['expression'],
                                   'GT: green | Prediction: red')
    localization_path = folder / 'localization.png'
    atomic_png(localization_path, localization)

    field = visual_np.asarray(heatmap, dtype=visual_np.float32)
    if field.ndim != 2 or not visual_np.isfinite(field).all():
        raise ValueError(f"{sample['id']}: invalid visualization heatmap")
    # Fixed [0,1] color scale preserves the meaning of the TFBD threshold.
    values = visual_np.clip(field, 0.0, 1.0)
    red = visual_np.clip(1.5 - visual_np.abs(4 * values - 3), 0, 1)
    green = visual_np.clip(1.5 - visual_np.abs(4 * values - 2), 0, 1)
    blue = visual_np.clip(1.5 - visual_np.abs(4 * values - 1), 0, 1)
    rgb = visual_np.rint(255 * visual_np.stack([red, green, blue], axis=-1)).astype('uint8')
    resampling = getattr(Image, 'Resampling', Image)
    heat_image = Image.fromarray(rgb).resize((width, height), resample=resampling.BILINEAR)
    heat_image = captioned_image(heat_image, f"{sample['id']} | TFBD refined heatmap",
                                sample['expression'], 'Low: blue (0) | High: red (1)')
    heatmap_path = folder / 'heatmap.png'
    atomic_png(heatmap_path, heat_image)
    return {'localization': f"{sample['id']}/localization.png",
            'heatmap': f"{sample['id']}/heatmap.png"}
