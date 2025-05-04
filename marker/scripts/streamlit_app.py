import os
import sys
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["IN_STREAMLIT"] = "true"

from marker.settings import settings
from marker.config.printer import CustomClickPrinter
from streamlit.runtime.uploaded_file_manager import UploadedFile

import base64
import io
import zipfile
import json
import re
import string
import tempfile
from typing import Any, Dict
import click

import pypdfium2
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered
from marker.schema import BlockTypes

COLORS = [
    "#4e79a7",
    "#f28e2c",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc949",
    "#af7aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ab"
]

with open(
    os.path.join(os.path.dirname(__file__), "streamlit_app_blocks_viz.html"), encoding="utf-8"
) as f:
    BLOCKS_VIZ_TMPL = string.Template(f.read())


@st.cache_data()
def parse_args():
    # Use to grab common cli options
    @ConfigParser.common_options
    def options_func():
        pass

    def extract_click_params(decorated_function):
        if hasattr(decorated_function, '__click_params__'):
            return decorated_function.__click_params__
        return []

    cmd = CustomClickPrinter("Marker app.")
    extracted_params = extract_click_params(options_func)
    cmd.params.extend(extracted_params)
    ctx = click.Context(cmd)
    try:
        cmd_args = sys.argv[1:]
        cmd.parse_args(ctx, cmd_args)
        return ctx.params
    except click.exceptions.ClickException as e:
        return {"error": str(e)}

@st.cache_resource()
def load_models():
    return create_model_dict()


from typing import Tuple # Import Tuple if not already imported, or rely on Python 3.9+ tuple[]

def convert_pdf(fname: str, config_parser: ConfigParser) -> tuple[str, Dict[str, Any], dict]:
    config_dict = config_parser.generate_config_dict()
    config_dict["pdftext_workers"] = 1
    converter_cls = PdfConverter
    converter = converter_cls(
        config=config_dict,
        artifact_dict=model_dict,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service()
    )
    return converter(fname)


def open_pdf(pdf_file):
    stream = io.BytesIO(pdf_file.getvalue())
    return pypdfium2.PdfDocument(stream)


def img_to_html(img, img_alt):
    img_bytes = io.BytesIO()
    img.save(img_bytes, format=settings.OUTPUT_IMAGE_FORMAT)
    img_bytes = img_bytes.getvalue()
    encoded = base64.b64encode(img_bytes).decode()
    img_html = f'<img src="data:image/{settings.OUTPUT_IMAGE_FORMAT.lower()};base64,{encoded}" alt="{img_alt}" style="max-width: 100%;">'
    return img_html


def markdown_insert_images(markdown, images):
    image_tags = re.findall(r'(!\[(?P<image_title>[^\]]*)\]\((?P<image_path>[^\)"\s]+)\s*([^\)]*)\))', markdown)

    for image in image_tags:
        image_markdown = image[0]
        image_alt = image[1]
        image_path = image[2]
        if image_path in images:
            markdown = markdown.replace(image_markdown, img_to_html(images[image_path], image_alt))
    return markdown


@st.cache_data()
def get_page_image(uploaded_file: UploadedFile, page_num, dpi=96):
    """Tries to render a preview image for the uploaded file."""
    file_type = uploaded_file.type
    # Handle PDF
    if "pdf" in file_type:
        try:
            doc = open_pdf(uploaded_file)
            if page_num < len(doc):
                page = doc[page_num]
                png_image = page.render(scale=dpi / 72).to_pil().convert("RGB")
                return png_image
            else:
                # Handle invalid page number for PDF
                return None # Or a placeholder image/error message
        except Exception as e:
            print(f"Error rendering PDF preview: {e}")
            return None # Failed to render PDF page
    # Handle common image types
    elif file_type in ["image/png", "image/jpeg", "image/gif"]:
        try:
            png_image = Image.open(uploaded_file).convert("RGB")
            return png_image
        except Exception as e:
            print(f"Error opening image preview: {e}")
            return None # Failed to open image
    # Handle other types (EPUB, DOCX, etc.) - no direct preview
    else:
        return None


@st.cache_data()
def page_count(pdf_file: UploadedFile):
    if "pdf" in pdf_file.type:
        doc = open_pdf(pdf_file)
        return len(doc) - 1
    else:
        return 1


def pillow_image_to_base64_string(img: Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def block_display(image: Image, blocks: dict | None = None, dpi=96):
    if blocks is None:
        blocks = {}

    image_data_url = (
        'data:image/jpeg;base64,' + pillow_image_to_base64_string(image)
    )

    template_values = {
        "image_data_url": image_data_url,
        "image_width": image.width, "image_height": image.height,
        "blocks_json": blocks, "colors_json": json.dumps(COLORS),
        "block_types_json": json.dumps({
            bt.name: i for i, bt in enumerate(BlockTypes)
        })
    }
    return components.html(
        BLOCKS_VIZ_TMPL.substitute(**template_values),
        height=image.height
    )


st.set_page_config(layout="wide")
col1, col2 = st.columns([.5, .5])

model_dict = load_models()
cli_options = parse_args()


in_file: UploadedFile = st.sidebar.file_uploader("PDF, document, or image file:", type=["pdf", "png", "jpg", "jpeg", "gif", "pptx", "docx", "xlsx", "html", "epub"])

if in_file is None:
    st.stop()

with col1:
    page_count = page_count(in_file)
    page_number = st.number_input(f"Page number out of {page_count}:", min_value=0, value=0, max_value=page_count)
    pil_image = get_page_image(in_file, page_number)
    image_placeholder = st.empty()

    # Display image or placeholder message
    with image_placeholder.container(): # Use container to manage content replacement
        if pil_image:
            block_display(pil_image)
        else:
            st.warning(f"Preview not available for file type '{in_file.type}' or page number {page_number}.")

page_range = st.sidebar.text_input("Page range to parse", value="", placeholder="Leave blank for all pages", help="Specify comma separated page numbers or ranges (e.g., 0,5-10,20). Leave blank to process the entire document.")
output_format = st.sidebar.selectbox("Output format", ["markdown", "json", "html"], index=0)
run_marker = st.sidebar.button("Run Marker")

use_llm = st.sidebar.checkbox("Use LLM", help="Use LLM for higher quality processing", value=False)

if use_llm:
    st.sidebar.subheader("OpenAI Configuration")
    # Use session_state keys for persistence across reruns
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", key="openai_api_key")
    openai_model = st.sidebar.text_input("OpenAI Model Name", value="gpt-4o", key="openai_model")
    openai_base_url = st.sidebar.text_input("OpenAI Base URL (Optional)", key="openai_base_url")
# show_blocks = st.sidebar.checkbox("Show Blocks", help="Display detected blocks, only when output is JSON", value=False, disabled=output_format != "json")
force_ocr = st.sidebar.checkbox("Force OCR", help="Force OCR on all pages", value=False)

# Display active device status
active_device = settings.TORCH_DEVICE_MODEL
if active_device == "cuda":
    st.sidebar.success("✅ Nvidia GPU (CUDA) Acceleration Active")
elif active_device == "mps":
    st.sidebar.success("✅ Apple Silicon (MPS) Acceleration Active")
else:
    st.sidebar.info(f"ℹ️ Running on CPU ({active_device})")

# languages = st.sidebar.text_input(
#     "OCR Languages (optional)",
#     value="",
#     placeholder="Comma-separated, e.g., en,zh",
#     help="Specify languages for OCR if needed (e.g., 'zh' for Chinese). Leave blank for default/auto-detect."
# )

if not run_marker:
    st.stop()

# Run Marker
with tempfile.TemporaryDirectory() as tmp_dir:
    temp_pdf = os.path.join(tmp_dir, 'temp.pdf')
    with open(temp_pdf, 'wb') as f:
        f.write(in_file.getvalue())
    
    # Update the main cli_options dictionary with base settings and languages
    cli_options.update({
        "output_format": output_format,
        "page_range": page_range,
        "force_ocr": force_ocr,
        "use_llm": use_llm, # Pass the checkbox state
        # "languages": languages if languages else None # Add languages if provided
    })

    # Now add the potentially updated LLM options from session state if use_llm is True
    if use_llm:
         # Check if the key exists in session state (meaning the input field was rendered and potentially filled)
         if 'openai_api_key' in st.session_state and st.session_state.openai_api_key:
             cli_options["llm_service"] = "marker.services.openai.OpenAIService"
             cli_options["openai_api_key"] = st.session_state.openai_api_key
             if 'openai_model' in st.session_state and st.session_state.openai_model:
                 cli_options["openai_model"] = st.session_state.openai_model
             if 'openai_base_url' in st.session_state and st.session_state.openai_base_url:
                 cli_options["openai_base_url"] = st.session_state.openai_base_url
         # else: # Optional: Handle if 'Use LLM' is checked but no OpenAI key is provided
         #    # Maybe default back to Gemini or show a warning? For now, do nothing.
    # Ensure llm_service is removed if use_llm is unchecked and was previously set
    elif "llm_service" in cli_options:
         del cli_options["llm_service"]
         # Also remove related keys if they exist
         cli_options.pop("openai_api_key", None)
         cli_options.pop("openai_model", None)
         cli_options.pop("openai_base_url", None)


    config_parser = ConfigParser(cli_options)
    rendered = convert_pdf(
        temp_pdf,
        config_parser
    )
    # Safely get page_range from config, defaulting to None if not specified (all pages)
    config_dict_for_page = config_parser.generate_config_dict()
    page_range_from_config = config_dict_for_page.get("page_range", None)
    # Determine first page for potential debug output, default to 0 if range is not set or empty
    first_page = page_range_from_config[0] if page_range_from_config else 0

text, ext, images = text_from_rendered(rendered)

# Create zip file in memory for download
# Get the base name from the original uploaded file
original_filename = in_file.name
base_filename = os.path.splitext(original_filename)[0]
zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
    # Determine the primary file extension based on output format
    primary_extension = "md" # Default to markdown
    if output_format == "json":
        primary_extension = "json"
    elif output_format == "html":
        primary_extension = "html"

    # Prepare primary output content
    output_content = text
    if output_format == "markdown" and images:
        # Modify markdown text to update image paths
        def replace_image_path(match):
            alt_text = match.group(1)
            original_path = match.group(2)
            # Ensure the new path starts with 'images/' and handles potential leading slashes
            new_path = f"images/{original_path.lstrip('/')}"
            return f"![{alt_text}]({new_path})"
        # Use regex to find markdown image links: ![alt](path)
        output_content = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_image_path, text)

    # Write primary output file (md, json, or html)
    primary_filename = f"{base_filename}.{primary_extension}"
    # Use the potentially modified output_content
    zip_file.writestr(primary_filename, output_content.encode('utf-8') if isinstance(output_content, str) else json.dumps(output_content, indent=2).encode('utf-8'))

    # Write image files if the output format is markdown and images exist
    if output_format == "markdown" and images: # images should be available
        for img_path, img_obj in images.items():
            try:
                img_byte_arr = io.BytesIO()
                # Determine image format, default to PNG if unknown
                img_format = img_obj.format if hasattr(img_obj, 'format') and img_obj.format else 'PNG'
                img_obj.save(img_byte_arr, format=img_format)
                # Ensure img_path is a valid path within the zip, place it in 'images/' subdirectory
                zip_img_path = f"images/{img_path.lstrip('/')}"
                # Create the directory structure if needed (ZipFile handles this implicitly when writing)
                zip_file.writestr(zip_img_path, img_byte_arr.getvalue())
            except Exception as e:
                st.sidebar.error(f"Error saving image {img_path}: {e}") # Optional: show error in UI

# Offer zip file for download via sidebar
st.sidebar.download_button(
    label="Download Results (.zip)",
    data=zip_buffer.getvalue(),
    file_name=f"{base_filename}_marker_output.zip",
    mime="application/zip",
    key="download_zip_button" # Update key if needed
)
with col2:
    if output_format == "markdown":
        text = markdown_insert_images(text, images)
        st.markdown(text, unsafe_allow_html=True)
    elif output_format == "json":
        st.json(text)
    elif output_format == "html":
        st.html(text)
