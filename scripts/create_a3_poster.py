#!/usr/bin/env python3
"""Generate an A3 exhibition poster for Hot Seat."""

from __future__ import annotations

from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
OUTPUT = ROOT / "output" / "pdf"
DPI = 300
A3_PORTRAIT_PX = (3508, 4961)


COLORS = {
    "page": (2, 3, 6),
    "panel": (14, 15, 21),
    "panel_2": (22, 18, 18),
    "text": (246, 246, 248),
    "muted": (166, 166, 174),
    "soft": (115, 116, 124),
    "hairline": (255, 255, 255, 32),
    "orange": (255, 132, 28),
    "red": (255, 69, 58),
    "blue": (10, 132, 255),
    "green": (48, 209, 88),
    "amber": (255, 159, 10),
}


def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    candidates = {
        "black": [
            "/System/Library/Fonts/SFNS.ttf",
            "/System/Library/Fonts/Supplemental/Arial Black.ttf",
        ],
        "bold": [
            "/System/Library/Fonts/SFNS.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ],
        "regular": [
            "/System/Library/Fonts/SFNS.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ],
    }
    for candidate in candidates.get(weight, candidates["regular"]):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


FONT_TITLE = font(330, "black")
FONT_H1 = font(94, "black")
FONT_H2 = font(72, "bold")
FONT_BODY = font(50, "regular")
FONT_SMALL = font(38, "regular")
FONT_CHIP = font(40, "bold")
FONT_LABEL = font(34, "bold")
FONT_CAPTION = font(33, "regular")


def rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, size[0], size[1]), radius=radius, fill=255)
    return mask


def cover_image(path: Path, size: tuple[int, int], position=(0.5, 0.5)) -> Image.Image:
    image = Image.open(path).convert("RGB")
    src_w, src_h = image.size
    dst_w, dst_h = size
    scale = max(dst_w / src_w, dst_h / src_h)
    new_w, new_h = round(src_w * scale), round(src_h * scale)
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    x = round((new_w - dst_w) * position[0])
    y = round((new_h - dst_h) * position[1])
    return image.crop((x, y, x + dst_w, y + dst_h))


def contain_image(path: Path, size: tuple[int, int], background=(8, 9, 14)) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, background)
    x = (size[0] - image.size[0]) // 2
    y = (size[1] - image.size[1]) // 2
    canvas.paste(image, (x, y))
    return canvas


def paste_rounded(base: Image.Image, image: Image.Image, box: tuple[int, int], radius: int) -> None:
    mask = rounded_mask(image.size, radius)
    base.paste(image, box, mask)


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font_obj, fill, spacing=8):
    draw.multiline_text(xy, text, font=font_obj, fill=fill, spacing=spacing)


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font_obj,
    fill,
    width_chars: int,
    spacing=14,
):
    wrapped = "\n".join(textwrap.wrap(text, width=width_chars))
    draw.multiline_text(xy, wrapped, font=font_obj, fill=fill, spacing=spacing)


def glass_panel(draw: ImageDraw.ImageDraw, box, radius=42, fill=(16, 17, 23, 255)):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=(58, 60, 70, 255), width=2)


def add_background(canvas: Image.Image) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    width, height = canvas.size
    for y in range(height):
        t = y / height
        r = round(2 + 18 * (1 - abs(t - 0.16)))
        g = round(3 + 8 * (1 - abs(t - 0.22)))
        b = round(6 + 8 * (1 - abs(t - 0.60)))
        draw.line((0, y, width, y), fill=(r, g, b, 255))

    for x in range(0, width, 105):
        draw.line((x, 0, x, height), fill=(255, 255, 255, 13), width=1)
    for y in range(0, height, 105):
        draw.line((0, y, width, y), fill=(255, 255, 255, 13), width=1)

    # Heat-like glow behind the title and device image.
    glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    g = ImageDraw.Draw(glow, "RGBA")
    g.ellipse((-580, 120, 1460, 1900), fill=(255, 120, 32, 58))
    g.ellipse((1700, 520, 3960, 2900), fill=(10, 132, 255, 34))
    g.ellipse((720, 3000, 3560, 5600), fill=(255, 69, 58, 30))
    glow = glow.filter(ImageFilter.GaussianBlur(120))
    canvas.alpha_composite(glow)


def draw_chip(draw: ImageDraw.ImageDraw, xy, text, color, width=None):
    x, y = xy
    text_box = draw.textbbox((0, 0), text, font=FONT_CHIP)
    text_w = text_box[2] - text_box[0]
    w = width or text_w + 70
    h = 78
    draw.rounded_rectangle((x, y, x + w, y + h), radius=39, fill=(18, 19, 25, 255), outline=(70, 72, 84, 255), width=2)
    draw.ellipse((x + 24, y + 24, x + 54, y + 54), fill=color)
    draw.text((x + 74, y + 20), text, font=FONT_CHIP, fill=COLORS["text"])


def draw_thermal_pair(canvas: Image.Image, draw: ImageDraw.ImageDraw) -> None:
    section_y = 2200
    draw.text((180, section_y), "What the sensor sees", font=FONT_H2, fill=COLORS["text"])
    draw.text(
        (180, section_y + 86),
        "Warm bodies appear as coarse heat fields - no camera image, no face.",
        font=FONT_BODY,
        fill=COLORS["muted"],
    )

    card_w, card_h = 1518, 1030
    image_h = 840
    gap = 112
    for index, (name, label, caption) in enumerate(
        [
            ("thermal-empty.png", "EMPTY BENCH", "cool background, no body heat"),
            ("thermal-occupied.png", "OCCUPIED", "human warmth crosses the sitting area"),
        ]
    ):
        x = 180 + index * (card_w + gap)
        y = section_y + 205
        glass_panel(draw, (x, y, x + card_w, y + card_h), radius=48)
        img = cover_image(ASSETS / name, (card_w - 42, image_h), position=(0.5, 0.5))
        paste_rounded(canvas, img, (x + 21, y + 21), radius=34)
        draw.text((x + 48, y + image_h + 54), label, font=FONT_LABEL, fill=COLORS["text"])
        draw.text((x + 48, y + image_h + 102), caption, font=FONT_CAPTION, fill=COLORS["muted"])


def draw_deployment_story(canvas: Image.Image, draw: ImageDraw.ImageDraw) -> None:
    y = 3495
    draw.text((180, y), "Deployed in Lab 107", font=FONT_H2, fill=COLORS["text"])
    draw.text(
        (180, y + 82),
        "One thermal view answers two practical questions at the real soldering bench.",
        font=FONT_BODY,
        fill=COLORS["muted"],
    )

    photo_x, photo_y = 180, y + 190
    photo_w, photo_h = 1120, 760
    glass_panel(draw, (photo_x, photo_y, photo_x + photo_w, photo_y + photo_h), radius=42)
    scene = cover_image(
        ASSETS / "lab-107-workstation.jpg",
        (photo_w - 28, photo_h - 28),
        position=(0.5, 0.38),
    )
    paste_rounded(canvas, scene, (photo_x + 14, photo_y + 14), radius=32)
    draw.rounded_rectangle(
        (photo_x + 44, photo_y + photo_h - 100, photo_x + 690, photo_y + photo_h - 36),
        radius=32,
        fill=(0, 0, 0, 210),
        outline=(255, 255, 255, 48),
        width=2,
    )
    draw.text(
        (photo_x + 78, photo_y + photo_h - 84),
        "Soldering workstation, Lab 107",
        font=FONT_CAPTION,
        fill=COLORS["text"],
    )

    panel_x, panel_y = 1390, photo_y
    panel_w, panel_h = 1938, photo_h
    glass_panel(draw, (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h), radius=42)
    draw.text((panel_x + 70, panel_y + 56), "Is someone using the bench?", font=FONT_H1, fill=COLORS["text"])
    draw.text((panel_x + 70, panel_y + 174), "OCCUPANCY", font=FONT_LABEL, fill=COLORS["muted"])

    states = [
        ("FREE", COLORS["green"], panel_x + 70, panel_y + 232),
        ("OCCUPIED", COLORS["red"], panel_x + 930, panel_y + 232),
        ("COOLING", COLORS["blue"], panel_x + 70, panel_y + 610),
        ("HOT TOOL", COLORS["amber"], panel_x + 930, panel_y + 610),
    ]
    for label, color, x, state_y in states:
        draw.ellipse((x, state_y + 6, x + 52, state_y + 58), fill=color)
        draw.text((x + 82, state_y), label, font=FONT_H2, fill=COLORS["text"])

    draw.line(
        (panel_x + 70, panel_y + 380, panel_x + panel_w - 70, panel_y + 380),
        fill=(255, 255, 255, 34),
        width=2,
    )
    draw.text((panel_x + 70, panel_y + 428), "Has a hot tool been left behind?", font=FONT_H1, fill=COLORS["text"])
    draw.text((panel_x + 70, panel_y + 548), "SAFETY", font=FONT_LABEL, fill=COLORS["muted"])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGBA", A3_PORTRAIT_PX, COLORS["page"] + (255,))
    add_background(canvas)
    draw = ImageDraw.Draw(canvas, "RGBA")

    draw.text((180, 170), "CONNECTED ENVIRONMENTS EXHIBITION 2026", font=FONT_LABEL, fill=(255, 255, 255, 165))
    draw_text(draw, (170, 390), "HOT\nSEAT", FONT_TITLE, COLORS["text"], spacing=-22)
    draw_wrapped(
        draw,
        (190, 1065),
        "A privacy-friendly thermal monitor for shared soldering workstations.",
        FONT_H1,
        COLORS["text"],
        23,
        spacing=10,
    )
    draw_wrapped(
        draw,
        (190, 1515),
        "It senses occupancy and hot-tool safety without RGB video or face recognition.",
        FONT_BODY,
        COLORS["muted"],
        39,
        spacing=13,
    )
    draw_chip(draw, (190, 1760), "no RGB camera", COLORS["green"], width=470)
    draw_chip(draw, (690, 1760), "on-device ML", COLORS["blue"], width=470)
    draw_chip(draw, (1190, 1760), "hot-tool alert", COLORS["amber"], width=470)

    device_rect = (1880, 220, 3298, 2110)
    device = cover_image(ASSETS / "device-installed.jpg", (device_rect[2] - device_rect[0], device_rect[3] - device_rect[1]), position=(0.52, 0.50))
    device = ImageEnhance.Contrast(device).enhance(1.04)
    device = ImageEnhance.Color(device).enhance(0.94)
    paste_rounded(canvas, device, (device_rect[0], device_rect[1]), radius=72)
    draw.rounded_rectangle(device_rect, radius=72, outline=(255, 255, 255, 54), width=3)
    draw.rounded_rectangle((2030, 1925, 3190, 2038), radius=56, fill=(0, 0, 0, 215), outline=(255, 255, 255, 58), width=2)
    draw.text((2080, 1956), "FLIR Lepton + Raspberry Pi in Lab 107", font=FONT_CAPTION, fill=COLORS["text"])

    draw_thermal_pair(canvas, draw)
    draw_deployment_story(canvas, draw)

    # Bottom call-to-action strip.
    footer_y = 4620
    draw.line((180, footer_y - 70, 3328, footer_y - 70), fill=(255, 255, 255, 42), width=2)
    draw.text((180, footer_y), "Scan the live Lab 107 dashboard", font=FONT_H2, fill=COLORS["text"])
    draw.text((180, footer_y + 86), "Free / Occupied / Recently used / Hot tool", font=FONT_BODY, fill=COLORS["muted"])

    qr_path = ROOT / "live_lab_dashboard_qr.png"
    if qr_path.exists():
        qr_size = 340
        qr = Image.open(qr_path).convert("RGB").resize((qr_size, qr_size), Image.Resampling.NEAREST)
        qr_box = (2920, footer_y - 42, 2920 + qr_size, footer_y - 42 + qr_size)
        draw.rounded_rectangle((qr_box[0] - 18, qr_box[1] - 18, qr_box[2] + 18, qr_box[3] + 18), radius=36, fill=(255, 255, 255, 255))
        canvas.paste(qr, (qr_box[0], qr_box[1]))

    png_path = OUTPUT / "hot-seat-a3-poster.png"
    pdf_path = OUTPUT / "hot-seat-a3-poster.pdf"
    rgb = canvas.convert("RGB")
    rgb.save(png_path, dpi=(DPI, DPI), quality=95)
    rgb.save(pdf_path, "PDF", resolution=DPI)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
