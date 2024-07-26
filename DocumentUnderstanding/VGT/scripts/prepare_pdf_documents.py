from typing import List, Any
import os
import pdf2image
import argparse
from glob import glob


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare PDF documents for VGT')
    parser.add_argument(
        '--root_dir',
        required=True,
        type=str,
        help='Path to the root directory containing the PDF files')
    parser.add_argument(
        '--out_dir',
        required=True,
        type=str,
        help='Path to the output directory')
    parser.add_argument(
        '--img_dpi',
        required=True,
        type=int,
        default=72,
        help='The number of DPI used to convert PDFs to PNGs')
    return parser.parse_args()


def identify_subdirectories(
    root_dir: str
) -> List[str]:
    """ Find the relative paths of the subdirectories in the root directory. """
    pdf_paths = glob(f'{root_dir}/**/*.pdf', recursive=True)
    pdf_paths += glob(f'{root_dir}/*.pdf')
    pdf_paths = [os.path.split(x.replace(root_dir, ''))[0] for x in pdf_paths]

    return list(set(pdf_paths))


def create_output_directories(
    root_dir: str,
    subdirectories: List[str],
) -> None:
    """ Create output directories for each subdirectory. """
    for subdirectory in subdirectories:
        out_dir = f"{root_dir}/{subdirectory}"
        os.makedirs(out_dir, exist_ok=True)


def pfd_to_png(
    pdf_path: str,
    out_dir: str,
    dpi: int = 300
) -> None:
    """ Convert a PDF file to a list of PNG files. """
    images = pdf2image.convert_from_path(pdf_path, dpi=dpi, format='png')
    for i, image in enumerate(images):
        image.save(f"{out_dir}/{i+1}.png", "PNG")


def __main__():
    args = argument_parser()

    # Make the output directories
    subdirectories = identify_subdirectories(args.root_dir)
    create_output_directories(args.out_dir, subdirectories)

    # Convert PDFs to PNGs
    for subdirectory in subdirectories:
        pdf_paths = glob(f'{args.root_dir}/{subdirectory}/*.pdf')
        out_dir = f"{args.out_dir}/{subdirectory}"
        for pdf_path in pdf_paths:
            pfd_to_png(pdf_path, out_dir, dpi=args.img_dpi)
