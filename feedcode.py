import os
import argparse
from pathlib import Path
import fnmatch
import tiktoken

from constants import ALL_EXCLUSIONS

from aider.io import InputOutput
from aider.models import Model


SUPPORTED_LANGS = {
    "python": ["py"],
    "javascript": ["js", "jsx", "ts", "tsx"],
}

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens


def get_extensions_for_langs(langs_str: str = None) -> list[str]:
    """Get file extensions for specified languages or all supported languages if none specified."""
    if not langs_str:
        # Return all extensions if no languages specified
        return [ext for exts in SUPPORTED_LANGS.values() for ext in exts]
    
    langs = [lang.strip().lower() for lang in langs_str.split(",")]
    extensions = []
    for lang in langs:
        if lang in SUPPORTED_LANGS:
            extensions.extend(SUPPORTED_LANGS[lang])
    return extensions

def process_directory(directory, exclusions, extensions, use_aider=False):
    tree = []
    content = []
    directory = Path(directory)

    if use_aider:
        from aider.repomap import RepoMap

        io = InputOutput()
        model = Model("gpt-3.5-turbo")
        repo_map = RepoMap(root=directory, io=io, main_model=model)

    def should_exclude(path: Path):
        for pattern in exclusions:
            if path.match(pattern):
                return True
        return False

    def process_item(item, prefix="", is_last=False):
        if should_exclude(item):
            return

        current_prefix = "└── " if is_last else "├── "
        tree.append(f"{prefix}{current_prefix}{item.name}")

        if item.is_dir():
            contents = sorted(
                item.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())
            )
            for i, subitem in enumerate(contents):
                is_last_subitem = i == len(contents) - 1
                extension = "    " if is_last else "│   "
                process_item(subitem, prefix + extension, is_last_subitem)
        elif item.suffix[1:] in extensions:  # Remove the dot from suffix for comparison
            if use_aider:
                # Get sparse representation using RepoMap
                repo_content = repo_map.get_repo_map([], [item])
                if repo_content:
                    content.append(f"# File: {item}\n\n{repo_content}\n\n")
            else:
                with open(item, "r", encoding="utf-8") as f:
                    try:
                        file_content = f.read()
                        content.append(f"# File: {item}\n\n{file_content}\n\n")
                    except UnicodeDecodeError as e:
                        print(f"Error reading {item}: {e}")
                        return

    contents = sorted(
        directory.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())
    )
    for i, item in enumerate(contents):
        process_item(item, is_last=(i == len(contents) - 1))

    return tree, content


def main(directory, output_file, exclusions, langs, use_aider=False):
    extensions = get_extensions_for_langs(langs)
    tree, content = process_directory(directory, exclusions, extensions, use_aider)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Directory Tree:\n")
        f.write("\n".join(tree))
        f.write("\n\n# Concatenated Files:\n\n")
        f.write("".join(content))

    num_tokens = num_tokens_from_string("".join(content[: int(len(content) * 0.6)]))

    print("Num tokens: ", f"{num_tokens:,}")
    print(f"Output written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concatenate files in a directory and generate a directory tree, with exclusions."
    )
    parser.add_argument("directory", help="The directory to process")
    parser.add_argument(
        "-o",
        "--output",
        default="output.txt",
        help="The output file name (default: output.txt)",
    )
    parser.add_argument(
        "-e", "--exclude", nargs="*", default=[], help="Glob patterns for exclusions"
    )
    parser.add_argument(
        "-l", "--lang",
        help="Comma-separated list of languages to process (e.g., 'python,javascript'). If not specified, all supported languages will be processed."
    )
    parser.add_argument(
        "--aider",
        action="store_true",
        help="Use Aider's RepoMap to generate sparse file representations"
    )
    args = parser.parse_args()

    # Default exclusions
    default_exclusions = [
        ".git/*",
        ".idea/*",
        ".vscode/*",
        "tests/*",
        "*/tests/*",
        "test_*",
        "*_test.py"
    ] + ALL_EXCLUSIONS

    # Combine user-provided exclusions with default exclusions
    exclusions = default_exclusions + args.exclude
    print(f"Exclusion patterns: {exclusions}")

    if args.lang:
        for lang in args.lang.split(","):
            if lang not in SUPPORTED_LANGS:
                print(f"Unsupported language: {lang}")
                print(f"Supported languages: {', '.join(SUPPORTED_LANGS.keys())}")
                exit(1)

    main(args.directory, args.output, exclusions, args.lang, args.aider)
