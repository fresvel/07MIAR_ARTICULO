import os
from io import BytesIO
from urllib.request import urlopen

from bs4 import BeautifulSoup
from PIL import Image


def ensure_dir(path: str) -> None:
    # Crea la carpeta si no existe (incluye subcarpetas)
    os.makedirs(path, exist_ok=True)


def download_image_to_png(img_url: str, out_path: str) -> None:
    # Descarga bytes y abre con PIL de forma segura
    with urlopen(img_url) as resp:
        data = resp.read()

    img = Image.open(BytesIO(data))

    # Asegura modo compatible con PNG (por si viene como RGBA, P, etc.)
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")

    img.save(out_path, format="PNG")


def html_url_parser(url: str, save_dir: str, show: bool = False, wait: bool = False) -> None:
    """
    Descarga imágenes listadas en un index.html y las guarda como PNG.
    """
    ensure_dir(save_dir)

    with urlopen(url) as website:
        html = website.read()

    soup = BeautifulSoup(html, "html5lib")

    for image_id, link in enumerate(soup.find_all("a", href=True)):
        if image_id == 0:
            continue  # normalmente el primer link es "Parent Directory"

        img_url = link["href"]

        out_file = os.path.join(save_dir, f"img-{image_id}.png")

        try:
            if not os.path.isfile(out_file):
                print("[INFO] Downloading image from URL:", img_url)
                download_image_to_png(img_url, out_file)

                if show:
                    Image.open(out_file).show()
            else:
                print("[INFO] skipped:", out_file)

        except KeyboardInterrupt:
            print("[EXCEPTION] Pressed 'Ctrl+C'")
            break
        except Exception as e:
            print("[EXCEPTION]", e)
            continue

        if wait:
            key = input("[INFO] Press any key to continue ('q' to exit)... ")
            if key.lower() == "q":
                break


if __name__ == "__main__":
    URL_TRAIN_IMG = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/index.html"
    URL_TRAIN_GT  = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/map/index.html"

    URL_TEST_IMG  = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/sat/index.html"
    URL_TEST_GT   = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/map/index.html"

    html_url_parser(url=URL_TRAIN_IMG, save_dir="./road_segmentation/training/input/")
    html_url_parser(url=URL_TRAIN_GT,  save_dir="./road_segmentation/training/output/")

    html_url_parser(url=URL_TEST_IMG,  save_dir="./road_segmentation/testing/input/")
    html_url_parser(url=URL_TEST_GT,   save_dir="./road_segmentation/testing/output/")

    print("[INFO] All done!")
