# modulos/model_latex.py
import io
import os
from contextlib import redirect_stdout
from typing import List, Optional


class ModelSummaryToLatex:
    """
    Exporta el model.summary() de Keras a una tabla LaTeX (longtable) con líneas,
    guardando el resultado en un .tex dentro de la carpeta assets (por defecto).

    Uso típico:
        exporter = ModelSummaryToLatex()
        tex_path = exporter.save(model, out_dir="assets", filename="model_summary.tex")
    """

    def __init__(
        self,
        n_cols: int = 4,
        table_env: str = "longtable",
        col_spec: str = "|l|l|r|l|",
        font_cmd: str = r"\scriptsize",
        escape_latex: bool = True,
    ):
        self.n_cols = n_cols
        self.table_env = table_env
        self.col_spec = col_spec
        self.font_cmd = font_cmd
        self.escape_latex = escape_latex

    def _capture_summary(self, model) -> List[str]:
        buf = io.StringIO()
        with redirect_stdout(buf):
            model.summary()
        return buf.getvalue().splitlines()

    def _escape(self, s: str) -> str:
        # Escapado mínimo para LaTeX
        # Nota: mantenemos paréntesis y corchetes; escapamos caracteres conflictivos comunes.
        repl = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        out = []
        for ch in s:
            out.append(repl.get(ch, ch))
        return "".join(out)

    def _extract_rows(self, lines: List[str]) -> List[List[str]]:
        rows: List[List[str]] = []
        expected_cols = self.n_cols
        current_row: Optional[List[str]] = None

        def _strip_parts(raw: str) -> List[str]:
            parts = [p.strip() for p in raw.split("|")]
            if parts and parts[0] == "":
                parts = parts[1:]
            if parts and parts[-1] == "":
                parts = parts[:-1]
            return parts

        def _flush_row() -> None:
            nonlocal current_row
            if current_row and any(c.strip() for c in current_row):
                if self.escape_latex:
                    current_row = [self._escape(c) for c in current_row]
                rows.append(current_row)
            current_row = None

        for l in lines:
            has_vert = any(v in l for v in ("│", "┃", "║", "|"))
            if not has_vert:
                if any(ch in l for ch in ("┏", "┓", "┗", "┛", "┡", "┢", "┣", "┫", "┝", "┥", "┬", "┴", "┼", "─", "═", "━")):
                    _flush_row()
                continue

            norm = l.replace("│", "|").replace("┃", "|").replace("║", "|")
            if "Layer (type)" in norm:
                header_cols = _strip_parts(norm)
                if header_cols:
                    expected_cols = len(header_cols)
                current_row = None
                continue

            parts = _strip_parts(norm)
            if expected_cols and len(parts) != expected_cols:
                continue

            if current_row is None:
                current_row = [""] * expected_cols
            for i, part in enumerate(parts):
                if part:
                    if current_row[i]:
                        current_row[i] = f"{current_row[i]} {part}"
                    else:
                        current_row[i] = part

        _flush_row()

        return rows

    def to_latex(
        self,
        model,
        headers: Optional[List[str]] = None,
        caption: Optional[str] = None,
        label: Optional[str] = None,
    ) -> str:
        lines = self._capture_summary(model)
        rows = self._extract_rows(lines)

        if headers is None:
            headers = ["Layer", "Output Shape", "Params", "Connected to"]

        if len(headers) != self.n_cols:
            raise ValueError(f"headers debe tener {self.n_cols} elementos.")

        env = self.table_env
        spec = self.col_spec

        parts = []
        parts.append(self.font_cmd)
        parts.append(f"\\begin{{{env}}}{{{spec}}}")
        parts.append("\\hline")
        parts.append(" {} \\\\".format(" & ".join(headers)))
        parts.append("\\hline")

        for r in rows:
            parts.append(" {} \\\\".format(" & ".join(r)))
            parts.append("\\hline")

        if caption:
            # En longtable el caption va dentro; lo ponemos al final para simplificar.
            parts.append(f"\\caption{{{self._escape(caption) if self.escape_latex else caption}}}\\\\")
        if label:
            parts.append(f"\\label{{{label}}}")

        parts.append(f"\\end{{{env}}}")

        return "\n".join(parts) + "\n"

    def save(
        self,
        model,
        out_dir: str = "assets",
        filename: str = "model_summary.tex",
        headers: Optional[List[str]] = None,
        caption: Optional[str] = None,
        label: Optional[str] = None,
    ) -> str:
        os.makedirs(out_dir, exist_ok=True)
        tex = self.to_latex(model, headers=headers, caption=caption, label=label)

        out_path = os.path.join(out_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(tex)

        return out_path
