from __future__ import annotations

import ast
import re
from pathlib import Path

from querdex.schemas import Section

_JS_BLOCK_RE = re.compile(
    r"(?:function\s+(?P<fn>[A-Za-z_$][\w$]*)\s*\(|class\s+(?P<class>[A-Za-z_$][\w$]*)\s*)",
    re.MULTILINE,
)


class PythonCodeParser:
    source_format = "code_py"

    def parse(self, path: Path, doc_id: str) -> list[Section]:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        sections: list[Section] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                line = int(getattr(node, "lineno", 1))
                end_line = int(getattr(node, "end_lineno", line))
                snippet = "\n".join(source.splitlines()[line - 1 : end_line]).strip()
                docstring = ast.get_docstring(node) or ""
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                sections.append(
                    Section(
                        section_id=f"sec_{len(sections) + 1:04d}",
                        doc_id=doc_id,
                        content=snippet or docstring or getattr(node, "name", "anonymous"),
                        page_number=max(1, line),
                        source_format=self.source_format,
                        metadata={
                            "kind": kind,
                            "name": getattr(node, "name", "anonymous"),
                            "line_start": line,
                            "line_end": end_line,
                            "docstring": docstring,
                        },
                    )
                )

        if not sections and source.strip():
            sections.append(
                Section(
                    section_id="sec_0001",
                    doc_id=doc_id,
                    content=source.strip(),
                    page_number=1,
                    source_format=self.source_format,
                    metadata={"kind": "module", "name": path.name},
                )
            )
        return sections


class JSCodeParser:
    source_format = "code_js"

    def parse(self, path: Path, doc_id: str) -> list[Section]:
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        sections: list[Section] = []

        for match in _JS_BLOCK_RE.finditer(source):
            name = match.group("fn") or match.group("class") or "anonymous"
            start_idx = source[: match.start()].count("\n")
            line_no = start_idx + 1
            snippet = "\n".join(lines[max(0, start_idx - 1) : min(len(lines), start_idx + 12)]).strip()
            sections.append(
                Section(
                    section_id=f"sec_{len(sections) + 1:04d}",
                    doc_id=doc_id,
                    content=snippet or name,
                    page_number=max(1, line_no),
                    source_format=self.source_format,
                    metadata={
                        "kind": "class" if match.group("class") else "function",
                        "name": name,
                        "line_start": line_no,
                    },
                )
            )

        if not sections and source.strip():
            sections.append(
                Section(
                    section_id="sec_0001",
                    doc_id=doc_id,
                    content=source.strip(),
                    page_number=1,
                    source_format=self.source_format,
                    metadata={"kind": "module", "name": path.name},
                )
            )
        return sections
