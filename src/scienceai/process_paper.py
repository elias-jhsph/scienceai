import asyncio
import base64
import io
import logging
import os
import re
import shutil
import tempfile
import warnings
from difflib import SequenceMatcher
from math import atan2, degrees

import fitz
import pytesseract
import requests
from habanero import Crossref
from PIL import Image

from .llm import async_client as client
from .llm import get_config, use_tools

logger = logging.getLogger(__name__)

cr = Crossref()


def normalize_title(title: str) -> str:
    """Normalize a title for comparison by lowercasing and removing punctuation."""
    title = title.lower()
    # Remove common punctuation and extra whitespace
    title = re.sub(r"[^\w\s]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def title_similarity(title1: str, title2: str) -> float:
    """
    Calculate similarity between two titles using SequenceMatcher.
    Returns a value between 0 and 1, where 1 is an exact match.
    """
    norm1 = normalize_title(title1)
    norm2 = normalize_title(title2)
    return SequenceMatcher(None, norm1, norm2).ratio()


async def summarize_paper(text):
    system_message = (
        "Given a block of text, your task is to summarize the text into a concise paragraph. "
        "Do not include any references or citations in the summary. "
        "Do not speak to the user directly, just produce the summary of the text you are given."
    )

    user_message = "Summarize this text:\n\n" + text

    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]

    arguments = {"messages": messages, "model": get_config().default_model}

    response = await client.chat.completions.create(**arguments)
    return response.choices[0].message.content


async def extract_doi(images, incorrect_doi_list=None):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "store_doi",
                "description": "Store the DOI in the database",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doi": {
                            "type": "string",
                            "description": "The DOI to store",
                        },
                    },
                    "additionalProperties": False,
                    "required": ["doi"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "strict": True,
                "name": "keep_searching_for_doi",
                "description": "Keep searching for the DOI",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                    "required": [],
                },
            },
        },
    ]
    system_message = (
        "Given a block of text, your task is to extract the DOI from the text. "
        "The DOI is a unique alphanumeric string that provides a permanent link to the location of an "
        "online resource. It is often found in the header, footer, or metadata of a research paper. "
        "If the DOI is not present in the text, please use the keep_searching_for_doi function. "
        "If the DOI is found, please store it in the database for future reference by using the "
        "store_doi function. "
        "\nExample of DOI: '12.3456/nature123'. If the DOI is in the form of a URL, please extract the "
        "DOI from the URL and store the DOI without the URL format. "
    )
    user_message_prefix = "Extract the DOI from this image"

    if incorrect_doi_list:
        user_message_prefix = (
            ". The DOI is not any of these '"
            + ", ".join(incorrect_doi_list)
            + "'. Extract the correct DOI from this image"
        )

    # Limit to first 2 pages for DOI search
    for image in images[:2]:
        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image,
                        },
                    },
                    {"type": "text", "text": user_message_prefix},
                ],
            },
        ]

        arguments = {"messages": messages, "tools": tools, "model": get_config().default_model}

        retry = 0
        valid_calls = []
        while valid_calls == [] and retry < 2:
            if retry > 0:
                logger.info("Retrying DOI Extraction(" + str(retry) + ")...")
            chat_response = await client.chat.completions.create(**arguments)
            if chat_response.choices[0].message.tool_calls:
                valid_calls = await use_tools(chat_response, arguments, call_functions=False)
                if valid_calls:
                    for call in valid_calls:
                        if call["name"] == "store_doi":
                            return call["parameters"]["doi"]
            retry += 1
    return None


async def process_single_page(
    i,
    image,
    first_page_system_message,
    body_system_message,
    figure_present_system_message,
    figure_system_message,
    figure_present_tools,
):
    logger.info("Processing page " + str(i + 1))

    page_text = "\n\n**Start of Page " + str(i + 1) + "**\n\n"

    messages = [
        {"role": "system", "content": body_system_message},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image,
                    },
                }
            ],
        },
    ]

    if i == 0:
        messages[0]["content"] = first_page_system_message

    arguments = {"messages": messages, "model": get_config().default_model, "temperature": 0.2}

    # Start body text extraction
    body_task = client.chat.completions.create(**arguments)

    # Start figure counting
    messages_fig = [
        {"role": "system", "content": figure_present_system_message},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image,
                    },
                }
            ],
        },
    ]

    arguments_fig = {
        "messages": messages_fig,
        "tools": figure_present_tools,
        "model": get_config().default_model,
        "temperature": 0.2,
        "tool_choice": {"type": "function", "function": {"name": "store_figure_table_count"}},
    }

    fig_count_task = client.chat.completions.create(**arguments_fig)

    # Await both initial tasks
    chat_response_body, chat_response_fig = await asyncio.gather(body_task, fig_count_task)

    body_content = chat_response_body.choices[0].message.content.replace("**PAGE_COMPLETE**", "")

    # LLM-based refusal detection
    async def detect_refusal(content):
        """Use an LLM to detect if the vision model refused to process the page."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "classify_response",
                    "description": "Classify if the text is a refusal to process content or actual extracted content.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "is_refusal": {
                                "type": "boolean",
                                "description": "True if the text is a refusal/apology/inability to help, False if it is actual content.",
                            },
                        },
                        "additionalProperties": False,
                        "required": ["is_refusal"],
                    },
                },
            }
        ]

        detection_messages = [
            {
                "role": "system",
                "content": "Classify the following text as either a refusal to process or actual content.",
            },
            {"role": "user", "content": f"Text to classify:\n\n{content[:1000]}"},
        ]

        try:
            detection_response = await client.chat.completions.create(
                messages=detection_messages,
                model=get_config().default_fast_model,
                temperature=0,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "classify_response"}},
            )

            if detection_response.choices[0].message.tool_calls:
                tool_call = detection_response.choices[0].message.tool_calls[0]
                import json

                args = json.loads(tool_call.function.arguments)
                return args["is_refusal"]
            return False

        except Exception as e:
            warnings.warn(f"Refusal detection failed: {e}. Using fallback heuristics.", stacklevel=2)
            # Fallback to basic heuristics if LLM call fails
            refusal_phrases = ["i'm unable to", "i can't assist", "i cannot", "sorry, i can't"]
            return any(phrase in content.lower() for phrase in refusal_phrases) or len(content.strip()) < 100

    is_refusal = await detect_refusal(body_content)

    if is_refusal:
        warnings.warn(
            f"Vision model refused to process page {i + 1}. Retrying with academic research context.", stacklevel=2
        )
        logger.warning(f"⚠️  Vision model refused page {i + 1}, retrying with explicit academic context...")
        logger.debug(f"Refusal content: {body_content}")

        # Retry with a more explicit academic research prompt
        enhanced_system_message = (
            "You are assisting with legitimate academic research involving systematic literature review. "
            "Your task is to extract text from scanned pages of published research papers for the purpose of "
            "scientific analysis and meta-analysis. This is fair use under academic research guidelines. "
            "Please read the contents of the provided scan of a page from a research paper and convert the text "
            "that is in the main body of the paper to raw text. Do not include any tables, figures, footnotes, "
            "or reference sections. Once you have written out the text in the main body of the paper, "
            "write **PAGE_COMPLETE** and stop."
        )

        if i == 0:
            enhanced_system_message = (
                "You are assisting with legitimate academic research involving systematic literature review. "
                "Your task is to extract text from scanned pages of published research papers for the purpose of "
                "scientific analysis and meta-analysis. This is fair use under academic research guidelines. "
                "Please read the contents of the provided scan of a page from a research paper and convert the text "
                "to raw text. Skip the title, authors, headers, footers, legalese, copyrights, and references. "
                "Include the abstract and any other introductory text as well as the main body of the paper. "
                "Once you have written out the text in the main body of the paper, write **PAGE_COMPLETE** and stop."
            )

        retry_messages = [
            {"role": "system", "content": enhanced_system_message},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image,
                        },
                    }
                ],
            },
        ]

        retry_arguments = {"messages": retry_messages, "model": get_config().default_model, "temperature": 0.2}

        try:
            retry_response = await client.chat.completions.create(**retry_arguments)
            retry_content = retry_response.choices[0].message.content.replace("**PAGE_COMPLETE**", "")

            # Check if retry also resulted in refusal using the same robust detection
            is_retry_refusal = await detect_refusal(retry_content)

            if not is_retry_refusal:
                body_content = retry_content
                logger.info(f"✓ Retry succeeded for page {i + 1}")
            else:
                # Fall back to OCR if retry also failed
                logger.warning(f"⚠️  Retry also refused, falling back to OCR for page {i + 1}...")
                logger.debug(f"Retry refusal content: {retry_content}")
                raise Exception("Retry also refused")

        except Exception:
            # Fall back to OCR
            warnings.warn(f"Vision model retry failed for page {i + 1}. Falling back to OCR.", stacklevel=2)
            try:
                # Extract base64 data from data URI
                if image.startswith("data:image"):
                    image_data = image.split(",", 1)[1]
                else:
                    image_data = image

                image_bytes = base64.b64decode(image_data)
                pil_image = Image.open(io.BytesIO(image_bytes))

                # Use pytesseract to extract text
                ocr_text = pytesseract.image_to_string(pil_image)

                if ocr_text.strip():
                    body_content = ocr_text
                    logger.info(f"✓ OCR extracted {len(ocr_text)} characters from page {i + 1}")
                else:
                    warnings.warn(f"OCR also failed to extract text from page {i + 1}", stacklevel=2)
                    body_content = f"[Text extraction failed for this page]\n\nOriginal response: {body_content}"
            except Exception as e:
                warnings.warn(f"OCR fallback failed for page {i + 1}: {e}", stacklevel=2)
                body_content = f"[Text extraction failed for this page]\n\nOriginal response: {body_content}"

    page_text += body_content

    # Process figure count
    table_figure_count = -1
    if chat_response_fig.choices[0].message.tool_calls:
        valid_calls = await use_tools(chat_response_fig, arguments_fig, call_functions=False)
        if valid_calls:
            for call in valid_calls:
                if call["name"] == "store_figure_table_count":
                    try:
                        table_figure_count = int(call["parameters"]["figure_count"]) + int(
                            call["parameters"]["table_count"]
                        )
                    except Exception:  # nosec
                        table_figure_count = -1

    # If figures/tables present, extract them
    if table_figure_count < 0 or table_figure_count > 0:
        messages_desc = [
            {"role": "system", "content": figure_system_message},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image,
                        },
                    }
                ],
            },
        ]

        arguments_desc = {"messages": messages_desc, "model": get_config().default_model, "temperature": 0.2}

        chat_response_desc = await client.chat.completions.create(**arguments_desc)

        page_text += chat_response_desc.choices[0].message.content.replace("**FIGURES_AND_TABLES_COMPLETE**", "")

    page_text += "\n\n**End of Page " + str(i + 1) + "**\n\n"
    return i, page_text


async def create_cleaned_text(images):
    figure_present_system_message = (
        "Read the contents of the provided scan of a page from a research paper. "
        "Record the number of figures and tables that are present on the page."
    )

    figure_present_tools = [
        {
            "type": "function",
            "function": {
                "name": "store_figure_table_count",
                "description": "Store the count of figures and tables",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "figure_count": {
                            "type": "integer",
                            "description": "The number of figures on the page",
                        },
                        "table_count": {
                            "type": "integer",
                            "description": "The number of tables on the page",
                        },
                    },
                    "additionalProperties": False,
                    "required": ["figure_count", "table_count"],
                },
            },
        }
    ]

    figure_system_message = (
        "Read the contents of the provided scan of a page from a research paper. "
        "For each figure and table in the paper include a '**Figure/Table Description:**' "
        "section that you will author. This section should include a description of what is "
        "being communicated in the figure or table based on your best impression as well as "
        "all text that is found within that figure. Make sure to lay out figure or table text "
        "in a manner that best communicates the intent of the author with the rest of the "
        "output. You should always include a '**Figure/Table Description:**' for every "
        "figure and table you see in the scan. Once you have written out the text for "
        "all figures write **FIGURES_AND_TABLES_COMPLETE**"
    )

    body_system_message = (
        "Read the contents of the provided scan of a page from a research paper. "
        "Convert the tex that is in the main body of the paper to raw text. "
        "Do not include any tables. Do not include any figures. Do not include any footnotes. "
        "Do not include any reference sections. Once you have written out the text in the main "
        "body of the paper write  **PAGE_COMPLETE** and stop."
    )

    first_page_system_message = (
        "Read the contents of the provided scan of a page from a research paper. "
        "Convert the text of paper to raw text. Skip the title, authors, headers, footers, "
        "legalese, copyrights, and references. Include the abstract and any other "
        "introductory text as well as the main body of the paper. Once you have written out "
        "the text in the main body of the paper write **PAGE_COMPLETE** and stop."
    )

    # Limit concurrent page processing to avoid overwhelming the API
    page_semaphore = asyncio.Semaphore(5)

    async def process_page_with_limit(i, image):
        async with page_semaphore:
            return await process_single_page(
                i,
                image,
                first_page_system_message,
                body_system_message,
                figure_present_system_message,
                figure_system_message,
                figure_present_tools,
            )

    tasks = []
    for i, image in enumerate(images):
        tasks.append(process_page_with_limit(i, image))

    results = await asyncio.gather(*tasks)

    # Sort by page index to ensure correct order
    results.sort(key=lambda x: x[0])

    cleaned_text = "".join([res[1] for res in results])

    return cleaned_text


async def confirm_doi(title, images):
    system_message = (
        "Read the contents of the provided scan of a page from a research paper. "
        "Extract the title of the paper from the text"
    )
    tools = [
        {
            "type": "function",
            "function": {
                "strict": True,
                "name": "store_title",
                "description": "Store the title in the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "The title to store",
                        },
                    },
                    "additionalProperties": False,
                    "required": ["title"],
                },
            },
        },
    ]
    arguments = {
        "messages": [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": images[0],
                        },
                    }
                ],
            },
        ],
        "tools": tools,
        "model": get_config().default_model,
        "temperature": 0.2,
        "tool_choice": {"type": "function", "function": {"name": "store_title"}},
    }

    retry = 0
    title_found = False
    stored_title = ""
    while not title_found and retry < 3:
        chat_response = await client.chat.completions.create(**arguments)
        if chat_response.choices[0].message.tool_calls:
            valid_calls = await use_tools(chat_response, arguments, call_functions=False)
            if valid_calls:
                for call in valid_calls:
                    if call["name"] == "store_title":
                        stored_title = call["parameters"]["title"]
                        title_found = True
        retry += 1

    if not title_found:
        return False

    # First, do a quick programmatic similarity check
    sim_score = title_similarity(title, stored_title)
    logger.debug(f"Title similarity score: {sim_score:.3f}")

    # If similarity is very low, reject without LLM call
    if sim_score < 0.5:
        logger.debug(f"Title similarity too low ({sim_score:.3f} < 0.5), rejecting match")
        return False

    # If similarity is very high, accept without LLM call
    if sim_score > 0.95:
        logger.debug(f"Title similarity very high ({sim_score:.3f} > 0.95), accepting match")
        return True

    system_message = (
        "Determine if these two titles refer to THE SAME academic paper (not just similar topics). "
        "Papers can have slightly different title formatting but must be the EXACT same study. "
        "Be STRICT: papers on similar topics but different populations, different methods, "
        "or with different qualifiers (e.g., 'open fractures' vs general fractures, "
        "'rural Indian population' vs other populations) are DIFFERENT papers. "
        "Only return True if you are confident these are the SAME paper."
    )

    tools = [
        {
            "type": "function",
            "function": {
                "strict": True,
                "name": "store_title_similar",
                "description": "Store whether the titles refer to the same paper",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "titles_similar": {
                            "type": "boolean",
                            "description": "True ONLY if both titles refer to the exact same paper, not just similar topics",
                        },
                    },
                    "additionalProperties": False,
                    "required": ["titles_similar"],
                },
            },
        },
    ]

    arguments = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": "Title 1: " + title + "\nTitle 2: " + stored_title},
        ],
        "model": get_config().default_model,
        "temperature": 0.0,
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": "store_title_similar"}},
    }

    logger.info("Checking title similarity with LLM... Title 1: " + title + "\nTitle 2: " + stored_title)

    retry = 0
    is_title_match = None
    while is_title_match is None and retry < 3:
        chat_response = await client.chat.completions.create(**arguments)
        if chat_response.choices[0].message.tool_calls:
            valid_calls = await use_tools(chat_response, arguments, call_functions=False)
            if valid_calls:
                for call in valid_calls:
                    if call["name"] == "store_title_similar":
                        is_title_match = call["parameters"]["titles_similar"]
        retry += 1

    return is_title_match


async def extract_title_and_authors(images):
    system_message = (
        "Read the contents of the provided scan of a page from a research paper. "
        "Extract the title of the paper and the first author's name from the text."
    )
    tools = [
        {
            "type": "function",
            "function": {
                "strict": True,
                "name": "store_metadata",
                "description": "Store the title and author in the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "The title of the paper",
                        },
                        "first_author": {
                            "type": "string",
                            "description": "The first author's name",
                        },
                    },
                    "additionalProperties": False,
                    "required": ["title", "first_author"],
                },
            },
        },
    ]
    arguments = {
        "messages": [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": images[0],
                        },
                    }
                ],
            },
        ],
        "tools": tools,
        "model": get_config().default_model,
        "temperature": 0.2,
        "tool_choice": {"type": "function", "function": {"name": "store_metadata"}},
    }

    retry = 0
    found = False
    stored_title = None
    stored_author = None
    while not found and retry < 3:
        chat_response = await client.chat.completions.create(**arguments)
        if chat_response.choices[0].message.tool_calls:
            valid_calls = await use_tools(chat_response, arguments, call_functions=False)
            if valid_calls:
                for call in valid_calls:
                    if call["name"] == "store_metadata":
                        stored_title = call["parameters"]["title"]
                        stored_author = call["parameters"]["first_author"]
                        found = True
        retry += 1

    if not found:
        return None, None

    return stored_title, stored_author


def rotate_pdf_pages(pdf_path):
    doc = fitz.open(pdf_path)
    modified_pdf_path = pdf_path.replace(".pdf", "_rotated.pdf")
    mod_doc = fitz.open()
    for page in doc:
        text_blocks = page.get_text("dict")["blocks"]
        total_weight = 0
        total_length = 0
        weighted_sum_angles = 0
        blocks_found = False
        # Collect angles of text blocks
        for block in text_blocks:
            if block["type"] == 0:  # Text block
                blocks_found = True
                for line in block["lines"]:
                    dir_vector = line["dir"]
                    for span in line["spans"]:
                        angle = atan2(dir_vector[1], dir_vector[0])
                        text_length = len(span["text"])
                        total_length += text_length
                        weighted_sum_angles += degrees(angle) * text_length
                        total_weight += text_length

        weighted_average_angle = weighted_sum_angles / total_weight if total_weight else 0

        # Calculate the average angle if angles were detected
        if blocks_found and total_weight > 0 and not total_length < 2000:
            average_angle = weighted_average_angle

            # Determine the rotation needed to align text upright
            # We assume that text should be as close to 0 degrees as possible
            # This might need adjustments for specific use cases
            if average_angle != 0:
                # Normalize the average angle to the nearest multiple of 90
                # This is a simplistic approach; more sophisticated logic may be needed
                normalized_angle = 360 - (round(average_angle / 90) * 90)
                page.set_rotation(normalized_angle)
        else:
            pix = page.get_pixmap(dpi=300)
            image = Image.open(io.BytesIO(pix.tobytes()))

            try:
                # Use pytesseract to detect the orientation of the text in the image
                ocr_data = pytesseract.image_to_osd(
                    image, config="--psm 0 -c min_characters_to_try=50", output_type=pytesseract.Output.DICT
                )

                # Extract the rotation angle suggested by pytesseract
                rotation_angle = ocr_data["rotate"]

                # Rotate the page based on the OCR-detected angle
                page.set_rotation(rotation_angle + page.rotation)
            except pytesseract.TesseractError:
                # Ignore tesseract errors during rotation detection
                pass

        src_rect = page.rect  # source page rect
        w, h = src_rect.br  # save its width, height
        src_rot = page.rotation  # save source rotation
        page.set_rotation(0)  # set rotation to 0 temporarily
        page = mod_doc.new_page(width=w, height=h)  # make output page
        page.show_pdf_page(  # insert source page
            page.rect,
            doc,
            page.number,
            rotate=-src_rot,  # use reversed original rotation
        )
    doc.close()
    mod_doc.save(modified_pdf_path)
    mod_doc.close()
    shutil.move(modified_pdf_path, pdf_path)
    return


async def gather_metadata(pdf_path, pages):
    doc = fitz.open(pdf_path)
    found_doi = False
    old_doi_list = None
    crossref_data = None

    # Limit to first 2 pages for metadata search
    for i, page in enumerate(doc):
        if i > 1:
            break

        simple_doi_list = []
        text_blocks = page.get_text("dict")["blocks"]
        total_length = 0
        for block in text_blocks:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    # use regular expression to find DOI
                    for span in line["spans"]:
                        text = span["text"]
                        total_length += len(text)
                        if "10." in text:
                            # use a simple regular expression to find the DOI
                            simple_doi_list += re.findall(r"10\.\d{4,9}\/[-._;()/:a-zA-Z0-9]+", text)

        if total_length < 2000:
            pix = page.get_pixmap(dpi=300)
            image = Image.open(io.BytesIO(pix.tobytes()))

            text_found = pytesseract.image_to_string(image)

            simple_doi_list += re.findall(r"10\.\d{4,9}\/[-._;()/:a-zA-Z0-9]+", text_found)

        if simple_doi_list:
            for doi in simple_doi_list:
                # Clean up DOI - remove trailing punctuation that regex might have caught
                doi = doi.rstrip(".,;:/)")
                try:
                    crossref_data_temp = cr.works(ids=doi)
                    title = crossref_data_temp["message"]["title"][0]
                    logger.debug("EZ Title: " + title)
                    found_doi = await confirm_doi(title, pages)
                    if found_doi:
                        logger.debug("EZ Found DOI: " + str(found_doi))
                    if not found_doi:
                        if not old_doi_list:
                            old_doi_list = [doi]
                        else:
                            old_doi_list.append(doi)
                    else:
                        crossref_data = crossref_data_temp
                        break
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 404:
                        print(f"DOI not found in Crossref: {doi}")
                    else:
                        print(f"Crossref lookup failed for {doi}: {e}")
                    # Assume DOI is wrong if Crossref fails
                    if not old_doi_list:
                        old_doi_list = [doi]
                    else:
                        old_doi_list.append(doi)
                except Exception:
                    import traceback

                    print(traceback.format_exc())
                    # Assume DOI is wrong if it causes an error
                    if not old_doi_list:
                        old_doi_list = [doi]
                    else:
                        old_doi_list.append(doi)
        if found_doi:
            break

    doc.close()

    if not found_doi:
        retry = 0
        while retry < 4 and not found_doi:
            doi = await extract_doi(pages, incorrect_doi_list=old_doi_list)
            if not doi:
                logger.info("DOI not found...")
                break
            else:
                # Clean up DOI
                doi = doi.rstrip(".,;:/)")
                try:
                    crossref_data_temp = cr.works(ids=doi)
                    title = crossref_data_temp["message"]["title"][0]
                    logger.debug("Title: " + title)
                    found_doi = await confirm_doi(title, pages)
                    logger.debug("Found DOI: " + str(found_doi))
                    if not found_doi:
                        if not old_doi_list:
                            old_doi_list = [doi]
                        else:
                            old_doi_list.append(doi)
                    else:
                        crossref_data = crossref_data_temp
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 404:
                        print(f"DOI not found in Crossref: {doi}")
                    else:
                        print(f"Crossref lookup failed for {doi}: {e}")
                    # Assume DOI is wrong if Crossref fails
                    if not old_doi_list:
                        old_doi_list = [doi]
                    else:
                        old_doi_list.append(doi)
                except Exception:
                    import traceback

                    print(traceback.format_exc())
                    # Assume DOI is wrong if it causes an error
                    if not old_doi_list:
                        old_doi_list = [doi]
                    else:
                        old_doi_list.append(doi)
            retry += 1

    if not found_doi:
        warnings.warn("DOI not found", stacklevel=2)
        # Try to search by title if DOI is missing
        title_extracted, author_extracted = await extract_title_and_authors(pages)
        if title_extracted:
            search_query = title_extracted
            if author_extracted:
                search_query += " " + author_extracted

            logger.info(f"Searching Crossref for: {search_query}")
            try:
                # Fetch multiple results to find the best match
                res = cr.works(query=search_query, limit=10)
                if res["message"]["items"]:
                    # Score all results by title similarity
                    scored_results = []
                    for item in res["message"]["items"]:
                        if item.get("title"):
                            found_title = item["title"][0]
                            sim_score = title_similarity(title_extracted, found_title)
                            scored_results.append((sim_score, found_title, item))
                            logger.debug(f"  Candidate: '{found_title}' (similarity: {sim_score:.3f})")

                    # Sort by similarity score (descending)
                    scored_results.sort(key=lambda x: x[0], reverse=True)

                    # Only consider results with reasonable similarity
                    MIN_SIMILARITY_THRESHOLD = 0.6

                    for sim_score, found_title, item in scored_results:
                        if sim_score < MIN_SIMILARITY_THRESHOLD:
                            logger.debug(
                                f"Remaining candidates below threshold ({MIN_SIMILARITY_THRESHOLD}), stopping search"
                            )
                            break

                        logger.info(f"Checking best match: '{found_title}' (similarity: {sim_score:.3f})")

                        # Verify if the found title matches our extracted title
                        is_match = await confirm_doi(found_title, pages)
                        if is_match:
                            logger.info("Title match confirmed!")
                            crossref_data = {"message": item}
                            found_doi = True  # Treat it as found for metadata extraction purposes
                            break
                        else:
                            logger.info(f"Title match rejected: '{found_title}'")

                    if not found_doi:
                        logger.info("No matching paper found in Crossref results")
            except Exception as e:
                logger.error(f"Error searching Crossref by title/author: {e}")

    if found_doi and crossref_data:
        metadata = crossref_data["message"]
        # Check if author field exists and is a non-empty list before accessing
        if metadata.get("author") and len(metadata["author"]) > 0:
            if "given" not in metadata["author"][0]:
                metadata["author"][0] = {"given": "", "family": metadata["author"][0].get("name", "Unknown")}

        dois = []
        if "reference" in metadata:
            for ref in metadata["reference"]:
                if "DOI" in ref:
                    dois.append(ref["DOI"])

        references = []
        ref_number = 0
        # Retrieve metadata for multiple DOIs at once
        if dois:
            try:
                res = cr.works(ids=dois)
                # If only one result, cr.works returns a dict, not a list
                if isinstance(res, dict):
                    res = [res]
            except Exception as e:
                logger.warning(f"Batch reference fetch failed: {e}. Trying individual fetches...")
                res = []
                for doi in dois:
                    try:
                        single_res = cr.works(ids=doi)
                        res.append(single_res)
                    except Exception:  # nosec
                        # Ignore individual reference failures (e.g. 404s)
                        continue

            for item in res:
                try:
                    ref_number += 1
                    data = item["message"]
                    # Format the reference string based on available fields
                    if "author" in data and len(data["author"]) > 0:
                        if "given" in data["author"][0] and "family" in data["author"][0]:
                            author_str = ", ".join(
                                [
                                    author["given"] + " " + author["family"]
                                    for author in data.get("author", [])
                                    if "given" in author and "family" in author
                                ]
                            )
                        elif "name" in data["author"][0]:
                            author_str = ", ".join(
                                [author["name"] for author in data.get("author", []) if "name" in author]
                            )
                        else:
                            author_str = ""
                    else:
                        author_str = ""

                    title_str = data.get("title", [""])[0] if data.get("title", None) else ""
                    journal_str = data.get("container-title", [""])[0] if data.get("container-title", None) else ""
                    volume_str = data.get("volume", "")
                    page_str = data.get("page", "")
                    year_str = str(data["issued"]["date-parts"][0][0]) if data.get("issued", None) else ""
                    doi_str = data.get("DOI", "")
                    reference_str = (
                        f"{ref_number}. {author_str}. {title_str}. {journal_str}, {volume_str}, {page_str}, "
                        f"{year_str}. DOI: {doi_str}"
                    )
                    references.append(reference_str.strip())
                except Exception:  # nosec
                    # Skip malformed references
                    continue
    else:
        if not title_extracted:  # If we didn't extract it above
            title_extracted, author_extracted = await extract_title_and_authors(pages)

        if title_extracted:
            metadata = {"title": [title_extracted], "metadata_status": "Title extracted, DOI not found"}
            if author_extracted:
                metadata["author"] = [{"given": "", "family": author_extracted}]
        else:
            metadata = {"metadata_status": "Title not found, DOI not found"}
        references = []
    return references, metadata


async def process_paper(pdf_path):
    temp_folder = tempfile.mkdtemp()
    os.makedirs(temp_folder, exist_ok=True)
    # Open the PDF file
    # Rotate pages first (synchronous as it uses fitz/PIL)
    rotate_pdf_pages(pdf_path)

    doc = fitz.open(pdf_path)
    # use fitz to create a clear image of each page make sure to use a high DPI
    image_list = []
    for i in range(len(doc)):
        page = doc[i]
        mat = fitz.Matrix(200 / 72, 200 / 72)
        image = page.get_pixmap(matrix=mat)
        # image = page.get_pixmap()
        image_bytes = image.tobytes()
        image_list.append(image_bytes)

    page_images = []
    for image_bytes in image_list:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        page_image = "data:image/png;base64," + base64_image
        page_images.append(page_image)

    output = {}

    # Run metadata gathering and text cleaning in parallel
    metadata_task = gather_metadata(pdf_path, page_images)
    cleaned_text_task = create_cleaned_text(page_images)

    results = await asyncio.gather(metadata_task, cleaned_text_task)

    references, metadata = results[0]
    cleaned_text = results[1]

    title = metadata.get("title", ["Unknown Title"])[0]
    if title == "Unknown Title":
        raise ValueError(f"Failed to extract title/metadata for paper: {pdf_path}")

    output["page_images"] = page_images

    output["cleaned_text"] = title + "\n\n\n" + cleaned_text + "\n\n## REFERENCES\n" + "\n".join(references)
    output["metadata"] = metadata

    output["summary"] = await summarize_paper(output["cleaned_text"])

    return output
