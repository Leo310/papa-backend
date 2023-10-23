import re
from typing import List, Optional, Tuple, cast

from llama_index.schema import Document

from obsidiantools.api import Vault


def parse_md_header(markdown_text: str) -> List[Tuple[Optional[str], str]]:
    """Convert a markdown file to a dictionary.
    The keys are the headers and the values are the text under each header.
    """
    markdown_tups: List[Tuple[Optional[str], str]] = []

    lines = markdown_text.split("\n")

    current_header = None
    current_text = ""

    for line in lines:
        header_match = re.match(r"^#\s", line)
        if header_match:
            if current_header is not None:
                if current_text == "" or None:
                    continue
                markdown_tups.append((current_header, current_text))

            current_header = line
            current_text = ""
        else:
            current_text += line + "\n"
    markdown_tups.append((current_header, current_text))

    if current_header is not None:
        # pass linting, assert keys are defined
        markdown_tups = [
            (re.sub(r"#", "", cast(str, key)).strip(), re.sub(r"<.*?>", "", value))
            for key, value in markdown_tups
        ]
    else:
        markdown_tups = [
            (key, re.sub("<.*?>", "", value)) for key, value in markdown_tups
        ]

    return markdown_tups


def remove_images(content: str) -> str:
    """Get a dictionary of a markdown file from its path."""
    pattern = r"!{1}\[\[(.*)\]\]"
    return re.sub(pattern, "", content)


def remove_hyperlinks(content: str) -> str:
    """Get a dictionary of a markdown file from its path."""
    pattern = r"\[(.*?)\]\((.*?)\)"
    return re.sub(pattern, r"\1", content)


def parse_md(
    knowledge_base: Vault,
    filename: str,
    remove_hyperlinks: bool,
    remove_images: bool,
    errors: str = "ignore",
) -> List[Tuple[Optional[str], str]]:
    """Parse file into tuples."""
    content = knowledge_base.get_source_text(filename)
    properties = knowledge_base.get_front_matter(filename)
    for key, value in properties.items():
        if isinstance(value, list):
            properties[key] = ", ".join(value)
        if value is None:
            properties[key] = ""
    if remove_hyperlinks:
        content = remove_hyperlinks(content)
    if remove_images:
        content = remove_images(content)
    return properties, parse_md_header(content)


def load_document(
    knowledge_base: Vault,
    filename: str,
    filepath: str,
    remove_hyperlinks: bool = False,
    remove_images: bool = False,
) -> List[Document]:
    """Parse file into string."""
    properties, parsed_md_header = parse_md(
        knowledge_base, filename, remove_hyperlinks, remove_images
    )
    properties["file_name"] = filename
    results = []
    for header, text in parsed_md_header:
        doc = Document(
            id_=f"{filepath}#{header}" if header else f"{filepath}",
            text=f"\n\n{header}\n{text}" if header else text,
            metadata=properties or {},
        )
        results.append(doc)
    return results
