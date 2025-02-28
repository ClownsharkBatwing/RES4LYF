#!/usr/bin/env python3

import argparse
from PIL import Image
from PIL.PngImagePlugin import PngInfo

def extract_metadata(image_path):
    image = Image.open(image_path)
    metadata = image.info
    return metadata

def replace_metadata(source_image_path, target_image_path, output_image_path):
    metadata = extract_metadata(source_image_path)

    target_image = Image.open(target_image_path)
    
    png_info = PngInfo()
    for key, value in metadata.items():
        png_info.add_text(key, str(value))
    
    target_image.save(output_image_path, pnginfo=png_info)

def main():
    parser = argparse.ArgumentParser(description="Copy metadata from one PNG image to another.")
    parser.add_argument('source', type=str, help="Path to the source PNG image with the metadata.")
    parser.add_argument('target', type=str, help="Path to the target PNG image to replace metadata.")
    parser.add_argument('output', type=str, help="Path for the output PNG image with replaced metadata.")

    args = parser.parse_args()

    replace_metadata(args.source, args.target, args.output)

    print(f"Metadata from '{args.source}' has been copied to '{args.output}'.")

if __name__ == "__main__":
    main()



