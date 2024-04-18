import os

from optimization.image_editor import ImageEditor
from optimization.arguments import get_arguments
import re

if __name__ == "__main__":
    args = get_arguments()
    image_editor = ImageEditor(args)
    image_editor.edit_image()
    # image_editor.reconstruct_image()
