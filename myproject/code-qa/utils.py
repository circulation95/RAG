import nbformat


def get_cells(ipynb_filepath):
    # ipynb 파일 읽기
    with open(ipynb_filepath, "r", encoding="utf-8") as file:
        notebook = nbformat.read(file, as_version=4)

    # 코드셀과 마크다운 셀 구분
    cells = []
    for cell in notebook.cells:
        c = dict()
        c["type"] = cell.cell_type
        c["content"] = cell.source
        cells.append(c)
    return cells


def write_to_md(filename, output_cells):
    body = ""
    for cell in output_cells:
        if cell["type"] == "code":
            body += f"\n```python\n{cell['content']}\n```\n"
        elif cell["type"] == "markdown":
            body += f"\n{cell['content']}\n"

    # 파일로 저장
    with open(filename, "w", encoding="utf-8") as file:
        file.write(body)
    return filename


def convert_notebook_to_md(notebook_path):
    cells = get_cells(notebook_path)
    md_path = notebook_path.replace(".ipynb", ".md")
    write_to_md(md_path, cells)
    return md_path
