import base64
import os
import tempfile
import shutil

import fitz
from math import atan2, degrees

from .llm import client, use_tools
from pprint import pprint as print

from habanero import Crossref
cr = Crossref()


def summarize_paper(text):
    system_message = ("Given a block of text, your task is to summarize the text into a concise paragraph. "
                      "Do not include any references or citations in the summary. "
                      "Do not speak to the user directly, just produce the summary of the text you are given.")

    user_message = "Summarize this text:\n\n" + text

    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    arguments = {"messages": messages, "model": "gpt-4o"}

    response = client.chat.completions.create(**arguments)
    return response.choices[0].message.content


def extract_doi(images, incorrect_doi_list=None):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "store_doi",
                "description": "Store the DOI in the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doi": {
                            "type": "string",
                            "description": "The DOI to store",
                        },
                    },
                    "required": ["doi"],
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "keep_searching_for_doi",
                "description": "Keep searching for the DOI",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            }
        }
    ]
    system_message = ("Given a block of text, your task is to extract the DOI from the text. "
                      "The DOI is a unique alphanumeric string that provides a permanent link to the location of an "
                      "online resource. It is often found in the header, footer, or metadata of a research paper. "
                      "If the DOI is not present in the text, please use the keep_searching_for_doi function. "
                      "If the DOI is found, please store it in the database for future reference by using the "
                      "store_doi function. "
                      "\nExample of DOI: '12.3456/nature123'. If the DOI is in the form of a URL, please extract the "
                      "DOI from the URL and store the DOI without the URL format. ")
    user_message_prefix = "Extract the DOI from this image"

    if incorrect_doi_list:
        user_message_prefix = (". The DOI is not any of these '" + ", ".join(incorrect_doi_list) +
                               "'. Extract the correct DOI from this image")

    for image in images:
        messages = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": [
                            {
                              "type": "image_url",
                              "image_url": {
                                  "url": image,
                              }
                            },
                            {
                              "type": "text",
                              "text": user_message_prefix
                            }
                          ]
            }
        ]

        arguments = {"messages": messages, "tools": tools, "model": "gpt-4o"}

        retry = 0
        valid_calls = []
        while valid_calls == [] and retry < 5 :
            if retry > 0:
                print("Retrying...")
            chat_response = client.chat.completions.create(**arguments)
            if chat_response.choices[0].message.tool_calls:
                valid_calls = use_tools(chat_response, arguments, call_functions=False)
                if valid_calls:
                    for call in valid_calls:
                        if call["name"] == "store_doi":
                            return call["parameters"]["doi"]
            retry += 1
    return None


def create_cleaned_text(images):

    figure_present_system_message = ("Read the contents of the provided scan of a page from a research paper. "
                                     "Record the number of figures and tables that are present on the page.")

    figure_present_tools = [
        {
            "type": "function",
            "function": {
                "name": "store_figure_table_count",
                "description": "Store the count of figures and tables",
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
                        }
                    },
                    "required": ["figure_count", "table_count"],
                },
            }
        }
    ]

    figure_system_message = ("Read the contents of the provided scan of a page from a research paper. "
                             "For each figure and table in the paper include a '**Figure/Table Description:**' "
                             "section that you will author. This section should include a description of what is "
                             "being communicated in the figure or table based on your best impression as well as "
                             "all text that is found within that figure. Make sure to lay out figure or table text "
                             "in a manner that best communicates the intent of the author with the rest of the "
                             "output. You should always include a '**Figure/Table Description:**' for every "
                             "figure and table you see in the scan. Once you have written out the text for "
                             "all figures write **FIGURES_AND_TABLES_COMPLETE**")

    body_system_message = ("Read the contents of the provided scan of a page from a research paper. "
                           "Convert the tex that is in the main body of the paper to raw text. "
                           "Do not include any tables. Do not include any figures. Do not include any footnotes. "
                           "Do not include any reference sections. Once you have written out the text in the main "
                           "body of the paper write  **PAGE_COMPLETE** and stop.")

    first_page_system_message = ("Read the contents of the provided scan of a page from a research paper. "
                                 "Convert the text of paper to raw text. Skip the title, authors, headers, footers, "
                                 "legalese, copyrights, and references. Include the abstract and any other "
                                 "introductory text as well as the main body of the paper. Once you have written out "
                                 "the text in the main body of the paper write **PAGE_COMPLETE** and stop.")

    cleaned_text = ""

    for i, image in enumerate(images):
        print("Processing page " + str(i + 1))

        cleaned_text += "\n\n**Start of Page " + str(i + 1) + "**\n\n"

        messages = [
            {
                "role": "system",
                "content": body_system_message
            },
            {
                "role": "user",
                "content": [
                            {
                              "type": "image_url",
                              "image_url": {
                                  "url": image,
                              }
                            }
                          ]
            }
        ]

        if i == 0:
            messages[0]["content"] = first_page_system_message

        arguments = {"messages": messages, "model": "gpt-4o", "temperature": 0.2}

        chat_response = client.chat.completions.create(**arguments)

        cleaned_text += chat_response.choices[0].message.content.replace("**PAGE_COMPLETE**", "")

        messages = [
            {
                "role": "system",
                "content": figure_present_system_message
            },
            {
                "role": "user",
                "content": [
                            {
                              "type": "image_url",
                              "image_url": {
                                  "url": image,
                              }
                            }
                          ]
            }
        ]

        arguments = {"messages": messages, "tools": figure_present_tools, "model": "gpt-4o", "temperature": 0.2,
                     "tool_choice": {"type": "function", "function": {"name": "store_figure_table_count"}}}

        retry = 0
        table_figure_count = -1
        while table_figure_count < 0 and retry < 3:
            if retry > 0:
                print("Retrying...")
            chat_response = client.chat.completions.create(**arguments)
            if chat_response.choices[0].message.tool_calls:
                valid_calls = use_tools(chat_response, arguments, call_functions=False)
                if valid_calls:
                    for call in valid_calls:
                        if call["name"] == "store_figure_table_count":
                            try:
                                table_figure_count = (int(call["parameters"]["figure_count"]) +
                                                      int(call["parameters"]["table_count"]))
                            except Exception as e:
                                table_figure_count = -1
            retry += 1

        if table_figure_count < 0 or table_figure_count > 0:
            messages = [
                {
                    "role": "system",
                    "content": figure_system_message
                },
                {
                    "role": "user",
                    "content": [
                                {
                                  "type": "image_url",
                                  "image_url": {
                                      "url": image,
                                  }
                                }
                              ]
                }
            ]

            arguments = {"messages": messages, "model": "gpt-4o", "temperature": 0.2}

            chat_response = client.chat.completions.create(**arguments)

            cleaned_text += chat_response.choices[0].message.content.replace("**FIGURES_AND_TABLES_COMPLETE**", "")

        cleaned_text += "\n\n**End of Page " + str(i + 1) + "**\n\n"

    return cleaned_text


def confirm_doi(title, images):
    system_message = ("Read the contents of the provided scan of a page from a research paper. "
                      "Extract the title of the paper from the text")
    tools = [
        {
            "type": "function",
            "function": {
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
                    "required": ["title"],
                },
            }
        },
    ]
    arguments = {"messages": [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": [
                        {
                          "type": "image_url",
                          "image_url": {
                              "url": images[0],
                          }
                        }
                      ]
        }
    ], "tools": tools, "model": "gpt-4o", "temperature": 0.2,
        "tool_choice": {"type": "function", "function": {"name": "store_title"}}}

    retry = 0
    title_found = False
    while not title_found and retry < 3:
        chat_response = client.chat.completions.create(**arguments)
        if chat_response.choices[0].message.tool_calls:
            valid_calls = use_tools(chat_response, arguments, call_functions=False)
            if valid_calls:
                for call in valid_calls:
                    if call["name"] == "store_title":
                        stored_title = call["parameters"]["title"]
                        title_found = True
        retry += 1

    if not title_found:
        return False

    system_message = ("Are these titles likely to be the same?")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "store_title_similar",
                "description": "Store the title similarity in the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "titles_similar": {
                            "type": "boolean",
                            "description": "Are the titles similar?",
                        },
                    },
                    "required": ["titles_similar"],
                },
            }
        },
    ]

    arguments = {"messages": [{"role": "system", "content": system_message},
                              {"role": "user", "content": "Title 1: " + title + "\nTitle 2: " + stored_title}],
                 "model": "gpt-4o", "temperature": 0.2, "tools": tools,
                 "tool_choice": {"type": "function", "function": {"name": "store_title_similar"}}}

    print("Checking title similarity... "
          "Title 1: " + title + "\nTitle 2: " + stored_title)

    retry = 0
    title_similarity = None
    while title_similarity is None and retry < 3:
        chat_response = client.chat.completions.create(**arguments)
        if chat_response.choices[0].message.tool_calls:
            valid_calls = use_tools(chat_response, arguments, call_functions=False)
            if valid_calls:
                for call in valid_calls:
                    if call["name"] == "store_title_similar":
                        title_similarity = call["parameters"]["titles_similar"]
        retry += 1

    return title_similarity


def rotate_pdf_pages(pdf_path):
    doc = fitz.open(pdf_path)
    for page in doc:
        text_blocks = page.get_text("dict")["blocks"]
        total_weight = 0
        weighted_sum_angles = 0

        # Collect angles of text blocks
        for block in text_blocks:
            if block["type"] == 0:  # Text block
                for line in block['lines']:
                    dir_vector = line['dir']
                    for span in line['spans']:
                        angle = atan2(dir_vector[1], dir_vector[0])
                        text_length = len(span['text'])
                        weighted_sum_angles += degrees(angle) * text_length
                        total_weight += text_length

        weighted_average_angle = weighted_sum_angles / total_weight if total_weight else 0

        # Calculate the average angle if angles were detected
        if total_weight > 0:
            average_angle = weighted_average_angle

            # Determine the rotation needed to align text upright
            # We assume that text should be as close to 0 degrees as possible
            # This might need adjustments for specific use cases
            if average_angle != 0:
                # Normalize the average angle to the nearest multiple of 90
                # This is a simplistic approach; more sophisticated logic may be needed
                normalized_angle = 360 - (round(average_angle / 90) * 90)
                page.set_rotation(normalized_angle)
    modified_pdf_path = pdf_path.replace(".pdf", "_rotated.pdf")
    doc.save(modified_pdf_path)
    doc.close()
    shutil.move(modified_pdf_path, pdf_path)
    return


def gather_metadata(pages):
    retry = 3
    old_doi_list = None
    found_doi = False
    while retry > 0 and not found_doi:
        doi = extract_doi(pages, incorrect_doi_list=old_doi_list)
        try:
            crossref_data = cr.works(ids=doi)
            title = crossref_data["message"]["title"][0]
            print("Title: " + title)
            found_doi = confirm_doi(title, pages)
            print("Found DOI: " + str(found_doi))
            if not found_doi:
                if not old_doi_list:
                    old_doi_list = [doi]
                else:
                    old_doi_list.append(doi)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            retry -= 1
    if not found_doi:
        raise Exception("DOI not found")

    metadata = crossref_data["message"]
    if "given" not in metadata["author"][0]:
        metadata["author"][0] = {"given": "", "family": metadata["author"][0]["name"]}

    dois = []
    for ref in metadata["reference"]:
        if "DOI" in ref:
            dois.append(ref["DOI"])

    references = []
    ref_number = 0
    try:
        # Retrieve metadata for multiple DOIs at once
        res = cr.works(ids=dois)
        for item in res:
            ref_number += 1
            data = item['message']
            # Format the reference string based on available fields
            if 'given' in data['author'][0] and 'family' in data['author'][0]:
                author_str = ', '.join([author['given'] + ' ' + author['family'] for author in data.get('author', []) if
                                        'given' in author and 'family' in author])
            elif 'name' in data['author'][0]:
                author_str = ', '.join([author['name'] for author in data.get('author', []) if 'name' in author])
            else:
                author_str = ''
            title_str = data.get('title', [''])[0] if data.get('title', None) else ''
            journal_str = data.get('container-title', [''])[0] if data.get('container-title', None) else ''
            volume_str = data.get('volume', '')
            page_str = data.get('page', '')
            year_str = str(data['issued']['date-parts'][0][0]) if data.get('issued', None) else ''
            doi_str = data.get('DOI', '')
            reference_str = f"{ref_number}. {author_str}. {title_str}. {journal_str}, {volume_str}, {page_str}, " \
                            f"{year_str}. DOI: {doi_str}"
            references.append(reference_str.strip())
    except Exception as e:
        references.append(f"Details not retrieved due to error: {e}")
    return references, metadata


def process_paper(pdf_path):
    temp_folder = tempfile.mkdtemp()
    os.makedirs(temp_folder, exist_ok=True)
    # Open the PDF file
    rotate_pdf_pages(pdf_path)
    doc = fitz.open(pdf_path)
    # use fitz to create a clear image of each page make sure to use a high DPI
    image_list = []
    for i in range(len(doc)):
        page = doc[i]
        mat = fitz.Matrix(200 / 72, 200 / 72)
        image = page.get_pixmap(matrix=mat)
        #image = page.get_pixmap()
        image_bytes = image.tobytes()
        image_list.append(image_bytes)

    # # write the first image to a file to test
    # with open("page0.png", "wb") as f:
    #     f.write(image_list[0])

    page_images = []
    for image_bytes in image_list:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        page_image = "data:image/png;base64," + base64_image
        page_images.append(page_image)

    output = {}
    references, metadata = gather_metadata(page_images)
    title = metadata["title"][0]
    output["page_images"] = page_images
    output["cleaned_text"] = (title + "\n\n\n" +
                              create_cleaned_text(page_images) + "\n\n## REFERENCES\n" + "\n".join(references))
    output["metadata"] = metadata

    output["summary"] = summarize_paper(output["cleaned_text"])

    return output
