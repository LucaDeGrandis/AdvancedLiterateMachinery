from typing import List, Any, Tuple
import os
import pdf2image
import argparse
from glob import glob
from tqdm import tqdm
import pickle
import numpy as np
from transformers import AutoTokenizer
import pdfplumber


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
        required=False,
        type=int,
        default=72,
        help='The number of DPI used to convert PDFs to PNGs')
    parser.add_argument(
        '--tokenizer',
        required=False,
        type=str,
        default='google-bert/bert-base-uncased',
        help='Tokenizer to be used')
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
    images = pdf2image.convert_from_path(pdf_path, dpi=dpi, fmt='png')
    for i, image in enumerate(images):
        image.save(f"{out_dir}/{i+1}.png", "PNG")


def return_word_grid(
    pdf_path: str,
) -> List[Any]:
    """Return the word information from a PDF file using pdfplumber.

    Returns:
        List:
            Returns a list of shape (num_pages, num_words, 8)
    """
    pdf = pdfplumber.open(pdf_path)

    word_data = list()
    for page in pdf.pages:
        # extracts words and their bounding boxes
        word_data.append(page.extract_words())

    return word_data


def tokenize(
    tokenizer: str,
    text_body: List[str]
) -> np.array:
    """Tokenize the input text using the provided tokenizer.

    Args
        tokenizer (str): HuggingFace Tokenizer to be used
        text_body (List[str]): List of text to be tokenized.

    Returns
        np.array: Return the tokenized input_ids.
    """
    # tokenize entire list of words
    tokenized_inputs = tokenizer.batch_encode_plus(
        text_body,
        return_token_type_ids=False,
        return_attention_mask=False,
        add_special_tokens=False
    )
    return tokenized_inputs["input_ids"]


def readjust_bbox_coords(bounding_box, tokens):
    """Readjust the bounding box coordinates based on the tokenized input.

    Args:
        bounding_box (List): List of bounding box coordinates in the format (x, y, width, height).
        tokens (List): List of input_ids from the tokenizer.

    Returns:
        List: A list of the adjusted bounding box coordinates.
    """
    adjusted_boxes = []
    for box, _id in zip(bounding_box, tokens):
        if len(_id) > 1:
            # Adjust the width and x-coordinate for each part
            new_width = box[2] / len(_id)
            for i in range(len(_id)):
                adjusted_boxes.append(
                    (box[0] + i * new_width, box[1], new_width, box[3])
                )
        else:
            adjusted_boxes.append((box[0], box[1], box[2], box[3]))
    return adjusted_boxes


def create_grid_dict(tokenizer, page_data):
    """Create a dictionary with the tokenized input,
    bounding box coordinates, and text.

    Parameters
    ----------
    tokenizer : HuggingFace Tokenizer
        The tokenizer to be used.
    page_data : List
        List of word information from pdfplumber.

    Returns
    -------
    Dict
        Returns a dictionary with the tokenized input,
        bounding box coordinates, and text.
    """
    grid = {
        "input_ids": [],
        "bbox_subword_list": [],
        "texts": [],
        "bbox_texts_list": []
    }

    for ele in page_data:
        grid["texts"].append(ele["text"])

        # since expected bbox format is (x,y,width,height)
        grid["bbox_texts_list"].append(
                (
                    ele["x0"],
                    ele["top"],
                    ele["x1"]-ele["x0"],
                    ele["bottom"]-ele["top"]
                )
            )

    input_ids = tokenize(tokenizer, grid["texts"])

    # flatten the input_ids
    grid["input_ids"] = np.concatenate(input_ids)

    grid["bbox_subword_list"] = np.array(
        readjust_bbox_coords(
            grid["bbox_texts_list"],
            input_ids
            )
        )

    grid["bbox_texts_list"] = np.array(grid["bbox_texts_list"])

    return grid


def save_pkl_file(
    grid,
    output_dir,
    output_file,
    model="doclaynet"
) -> None:
    """Save the grid dictionary as a pickle file.

    Args:
        grid (Dict): The grid dictionary to be saved.
        output_dir (str): The path to the output folder.
        output_file (str): The name of the output file.
        model (str, optional): Model that will be used by VGT further. Defaults to "doclaynet".

    Returns:
        None
    """
    if model == "doclaynet" or model == "publaynet":
        extension = "pdf.pkl"
    else:
        extension = "pkl"

    pkl_save_location = os.path.join(
        output_dir,
        f'{output_file}.{extension}')

    with open(pkl_save_location, 'wb') as handle:
        pickle.dump(grid, handle)


def select_tokenizer(tokenizer):
    """Select the tokenizer to be used.

    Args:
        tokenizer (str): The name of the tokenizer to be used.

    Returns:
        tokenizer (HuggingFace Tokenizer): The selected tokenizer.

    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    return tokenizer


def main():
    args = argument_parser()

    # Make the output directories
    subdirectories = identify_subdirectories(args.root_dir)
    create_output_directories(args.out_dir, subdirectories)

    # Convert PDFs to PNGs
    for subdirectory in tqdm(subdirectories):
        pdf_paths = glob(f'{args.root_dir}/{subdirectory}/*.pdf')
        out_dir = f"{args.out_dir}/{subdirectory}"
        for pdf_path in pdf_paths:
            pfd_to_png(pdf_path, out_dir, dpi=args.img_dpi)


if __name__ == '__main__':
    main()
