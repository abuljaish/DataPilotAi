import ast
from pathlib import Path


def test_app_starts_server_on_main_execution():
    file_path = Path(__file__).with_name('app.py')
    source = file_path.read_text(encoding='utf-8')
    tree = ast.parse(source)

    has_app_run = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == 'run' and node.func.value.id == 'app':
                has_app_run = True
                break

    assert has_app_run, 'app.run(...) is missing from the Flask startup block.'
