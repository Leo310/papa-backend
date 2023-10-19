import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

from llama_index.schema import Document


def markdown_to_tups(markdown_text: str) -> List[Tuple[Optional[str], str]]:
    """Convert a markdown file to a dictionary.

    The keys are the headers and the values are the text under each header.

    """
    markdown_tups: List[Tuple[Optional[str], str]] = []

    lines = markdown_text.split("\n")

    current_header = None
    current_text = ""

    for line in lines:
        header_match = re.match(r"^#+\s", line)
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


def parse_tups(
    filepath: Path, remove_hyperlinks: bool, remove_images: bool, errors: str = "ignore"
) -> List[Tuple[Optional[str], str]]:
    """Parse file into tuples."""
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
        # parse frontmatter/properties
        pattern = r"---\n(.*?)\n---"
        properties = re.search(pattern, content, re.DOTALL)
        parsed_properties = {}
        if properties:
            properties = properties.group(1)
            for key, value in yaml.safe_load(properties).items():
                if value:
                    if isinstance(value, list):
                        parsed_properties[key] = ", ".join(value)
                    else:
                        parsed_properties[key] = value
    if remove_hyperlinks:
        content = remove_hyperlinks(content)
    if remove_images:
        content = remove_images(content)
    return parsed_properties, markdown_to_tups(content)


def load_document(
    file: Path,
    remove_hyperlinks: bool = False,
    remove_images: bool = False,
) -> List[Document]:
    """Parse file into string."""
    properties, tups = parse_tups(file, remove_hyperlinks, remove_images)
    properties["file_name"] = file.name
    results = []
    file_path_in_knowledgebase = file.__str__().split("knowledge_base/")[1]
    i = 0
    for header, value in tups:
        if header is None:
            id_ = f"{file_path_in_knowledgebase}:{i}"
            i += 1
            results.append(
                Document(
                    id_=id_,
                    text=value,
                    metadata=properties or {},
                )
            )
        else:
            id_ = f"{file_path_in_knowledgebase}#{header}"
            results.append(
                Document(
                    id_=id_,
                    text=f"\n\n{header}\n{value}",
                    metadata=properties or {},
                )
            )
    return results
