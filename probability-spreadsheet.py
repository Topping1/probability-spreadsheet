import sys
import re
import xml.etree.ElementTree as ET
import numpy as np

from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QLineEdit, QAbstractItemView, QToolBar,
    QAction, QFileDialog, QMessageBox, QDialog, QHBoxLayout, QLabel, QPushButton
)

# For interactive matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Global variable for decimal formatting.
DECIMALS = 3

# ====================================================
# Distribution Infrastructure and Monte Carlo Helpers
# ====================================================

class Distribution:
    """
    A simple wrapper for Monte Carlo sampling.
    Arithmetic is overloaded so that operations between distributions (or scalars)
    produce new Distribution instances.
    
    Added freeze/unfreeze methods so that once computed, the samples are cached
    until a 'calculate' operation is requested.
    """
    def __init__(self, sampler):
        self.sampler = sampler
        self._frozen_sample = None

    def sample(self, n=5000):
        if self._frozen_sample is not None:
            return self._frozen_sample
        return self.sampler(n)

    def freeze(self, n=5000):
        self._frozen_sample = self.sampler(n)

    def unfreeze(self):
        self._frozen_sample = None

    def __add__(self, other):
        if isinstance(other, Distribution):
            return Distribution(lambda n=5000: self.sample(n) + other.sample(n))
        else:
            return Distribution(lambda n=5000: self.sample(n) + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Distribution):
            return Distribution(lambda n=5000: self.sample(n) - other.sample(n))
        else:
            return Distribution(lambda n=5000: self.sample(n) - other)

    def __rsub__(self, other):
        if isinstance(other, Distribution):
            return other.__sub__(self)
        else:
            return Distribution(lambda n=5000: other - self.sample(n))

    def __mul__(self, other):
        if isinstance(other, Distribution):
            return Distribution(lambda n=5000: self.sample(n) * other.sample(n))
        else:
            return Distribution(lambda n=5000: self.sample(n) * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Distribution):
            return Distribution(lambda n=5000: self.sample(n) / other.sample(n))
        else:
            return Distribution(lambda n=5000: self.sample(n) / other)

    def __rtruediv__(self, other):
        if isinstance(other, Distribution):
            return other.__truediv__(self)
        else:
            return Distribution(lambda n=5000: other / self.sample(n))

    def mean(self):
        return float(np.mean(self.sample()))

    def median(self):
        return float(np.percentile(self.sample(), 50))

    def minimum(self):
        return float(np.min(self.sample()))

    def maximum(self):
        return float(np.max(self.sample()))

    def percentile(self, p):
        return float(np.percentile(self.sample(), p))

    def __repr__(self):
        try:
            m = self.mean()
            med = self.median()
            return f"Distribution(mean={m:.{DECIMALS}f}, median={med:.{DECIMALS}f})"
        except Exception:
            return "Distribution(<error>)"

def constant(val):
    """Return a constant value as a Distribution."""
    return Distribution(lambda n=5000: np.full(n, val))

# ----------------------------
# Distribution Factory Functions
# ----------------------------

def normal(mean, stddev, min_val=None, max_val=None):
    def sampler(n=5000):
        samples = np.random.normal(mean, stddev, n)
        if min_val is not None:
            while True:
                mask = samples < min_val
                if not np.any(mask):
                    break
                samples[mask] = np.random.normal(mean, stddev, np.sum(mask))
        if max_val is not None:
            while True:
                mask = samples > max_val
                if not np.any(mask):
                    break
                samples[mask] = np.random.normal(mean, stddev, np.sum(mask))
        return samples
    return Distribution(sampler)

def uniform(min_val, max_val):
    return Distribution(lambda n=5000: np.random.uniform(min_val, max_val, n))

def exponential(rate):
    return Distribution(lambda n=5000: np.random.exponential(1/rate, n))

def poisson(mean_val):
    return Distribution(lambda n=5000: np.random.poisson(mean_val, n))

# ----------------------------
# Combining Functions for Distributions
# ----------------------------

def mmax(distA, distB):
    if not isinstance(distA, Distribution):
        distA = constant(distA)
    if not isinstance(distB, Distribution):
        distB = constant(distB)
    return Distribution(lambda n=5000: np.maximum(distA.sample(n), distB.sample(n)))

def choose(distA, distB):
    if not isinstance(distA, Distribution):
        distA = constant(distA)
    if not isinstance(distB, Distribution):
        distB = constant(distB)
    def sampler(n=5000):
        a = distA.sample(n)
        b = distB.sample(n)
        choices = np.random.randint(0, 2, n)
        return np.where(choices == 0, a, b)
    return Distribution(sampler)

# ----------------------------
# Functions Converting Distributions to Scalars
# ----------------------------

def min_func(dist):
    if isinstance(dist, Distribution):
        return dist.minimum()
    return dist

def mean_func(dist):
    if isinstance(dist, Distribution):
        return dist.mean()
    return dist

def max_func(dist):
    if isinstance(dist, Distribution):
        return dist.maximum()
    return dist

def perc(dist, p):
    """Return the p-th percentile of a distribution. Usage: perc(B2, 50)"""
    if isinstance(dist, Distribution):
        return dist.percentile(p)
    try:
        return float(dist)
    except:
        return dist

# ----------------------------
# Build a Safe Evaluation Environment for Formulae
# ----------------------------

def get_safe_env():
    env = {
        "normal": normal,
        "uniform": uniform,
        "exponential": exponential,
        "poisson": poisson,
        "mmax": mmax,
        "choose": choose,
        "min": min_func,
        "mean": mean_func,
        "max": max_func,
        "perc": perc,
    }
    return env

# ====================================================
# Spreadsheet Data Model (with Monte Carlo support)
# ====================================================

class SpreadsheetModel:
    def __init__(self):
        # Key: (row, col) ; Value: dict with "expr" (raw expression) and "value" (computed value)
        self.cell_data = {}
        # Dependency graphs.
        self.dependencies = {}         # cell -> set(dependencies)
        self.reverse_dependencies = {} # cell -> set(cells that depend on it)
        # Regex to find cell references (e.g. A1, B2)
        self.cell_ref_pattern = re.compile(r"\b([A-Za-z])(\d+)\b")

    def set_cell(self, row, col, expression):
        key = (row, col)
        self.cell_data[key] = {"expr": expression, "value": None}
        self._update_dependencies(key, expression)
        self._recalculate(key)

    def get_cell_value(self, row, col):
        key = (row, col)
        if key in self.cell_data:
            return self.cell_data[key]["value"]
        return ""

    def get_cell_expr(self, row, col):
        key = (row, col)
        if key in self.cell_data:
            return self.cell_data[key]["expr"]
        return ""

    def _update_dependencies(self, key, expression):
        if key in self.dependencies:
            # Remove this cell from any old dependencies' reverse_dependencies
            for dep in self.dependencies[key]:
                if dep in self.reverse_dependencies:
                    self.reverse_dependencies[dep].discard(key)
        new_deps = set()
        if expression.startswith("="):
            for match in self.cell_ref_pattern.finditer(expression[1:]):
                col_letter, row_number = match.groups()
                dep_col = ord(col_letter.upper()) - ord('A')
                dep_row = int(row_number) - 1
                new_deps.add((dep_row, dep_col))
        self.dependencies[key] = new_deps
        for dep in new_deps:
            self.reverse_dependencies.setdefault(dep, set()).add(key)

    def _eval_formula(self, formula, current_key):
        safe_env = get_safe_env()
        refs = set(re.findall(r"\b[A-Za-z]\d+\b", formula))
        for ref in refs:
            row = int(ref[1:]) - 1
            col = ord(ref[0].upper()) - ord('A')
            val = self.get_cell_value(row, col)
            if isinstance(val, str):
                # Attempt to convert numeric strings
                try:
                    val = float(val)
                except:
                    pass
            safe_env[ref] = val
        try:
            result = eval(formula, {"__builtins__": None}, safe_env)
        except Exception as e:
            result = f"Error: {e}"
        return result

    def _evaluate_cell(self, key):
        if key not in self.cell_data:
            return
        expression = self.cell_data[key]["expr"]
        if expression.startswith("="):
            formula = expression[1:]
            value = self._eval_formula(formula, key)
        else:
            # Try direct float conversion
            try:
                value = float(expression)
            except ValueError:
                value = expression
        # If the computed value is a Distribution, freeze it (if not already frozen).
        if isinstance(value, Distribution):
            if value._frozen_sample is None:
                value.freeze()
        self.cell_data[key]["value"] = value

    def _recalculate(self, key, visited=None):
        """
        Recursively recalculate the specified cell,
        then recalc all reverse dependencies of that cell.
        """
        if visited is None:
            visited = set()
        if key in visited:
            return
        visited.add(key)
        self._evaluate_cell(key)
        if key in self.reverse_dependencies:
            for dependent in self.reverse_dependencies[key]:
                self._recalculate(dependent, visited)

    def recalc_all_cells(self):
        """
        Recalculate every cell in the spreadsheet, respecting dependencies.
        """
        visited = set()
        for key in self.cell_data.keys():
            self._recalculate(key, visited)

    def recalc_distribution_cells(self):
        """
        For each cell that holds a Distribution, unfreeze and re-evaluate.
        (Not strictly necessary now, but left for reference.)
        """
        for key, cell in self.cell_data.items():
            value = cell["value"]
            if isinstance(value, Distribution):
                value.unfreeze()
                self._evaluate_cell(key)

    def clear(self):
        self.cell_data.clear()
        self.dependencies.clear()
        self.reverse_dependencies.clear()

    def to_xml(self):
        root = ET.Element("spreadsheet")
        for (row, col), cell in self.cell_data.items():
            cell_elem = ET.SubElement(root, "cell")
            cell_elem.set("row", str(row))
            cell_elem.set("col", str(col))
            expr_elem = ET.SubElement(cell_elem, "expr")
            expr_elem.text = cell["expr"]
        return ET.tostring(root, encoding="unicode")

    def from_xml(self, xml_string):
        try:
            root = ET.fromstring(xml_string)
            self.clear()
            for cell_elem in root.findall("cell"):
                row = int(cell_elem.get("row", "-1"))
                col = int(cell_elem.get("col", "-1"))
                expr_elem = cell_elem.find("expr")
                if expr_elem is not None and row >= 0 and col >= 0:
                    expr = expr_elem.text or ""
                    self.set_cell(row, col, expr)
        except Exception as e:
            print("Error loading XML:", e)

# ====================================================
# Detail Plot Dialog for Distribution Cells (Interactive Matplotlib)
# ====================================================

class DetailPlotDialog(QDialog):
    def __init__(self, distribution, raw_expr, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detailed Distribution Plot")
        self.distribution = distribution
        self.raw_expr = raw_expr
        self.resize(600, 500)
        layout = QVBoxLayout(self)

        # Create a matplotlib figure and canvas.
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Add the navigation toolbar.
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.nav_toolbar)

        # Add a label to display cursor coordinates.
        self.coordLabel = QLabel("Cursor: ")
        layout.addWidget(self.coordLabel)

        # Add a label to show summary statistics.
        self.summaryLabel = QLabel()
        layout.addWidget(self.summaryLabel)

        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.plot_distribution()

    def on_mouse_move(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.coordLabel.setText(f"Cursor: x={event.xdata:.{DECIMALS}f}, y={event.ydata:.{DECIMALS}f}")
        else:
            self.coordLabel.setText("Cursor: ")

    def plot_distribution(self):
        # Get fresh samples (for detailed view, we may use more samples).
        samples = self.distribution.sampler(10000)
        ax = self.figure.add_subplot(111)
        ax.clear()
        num_bins = 50
        bins = np.linspace(np.min(samples), np.max(samples), num_bins + 1)
        hist, bin_edges = np.histogram(samples, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # Plot as a line connecting the binned values.
        ax.plot(bin_centers, hist, marker='o', linestyle='-')
        # Draw a vertical line at the mean.
        mean_val = np.mean(samples)
        ax.axvline(mean_val, color='red', linestyle='--', label="mean")
        ax.legend()
        ax.set_title(self.raw_expr)
        self.canvas.draw()

        # Compute summary statistics.
        minimum = np.min(samples)
        perc10 = np.percentile(samples, 10)
        mean_val = np.mean(samples)
        perc90 = np.percentile(samples, 90)
        maximum = np.max(samples)
        summary_text = (
            f"Minimum: {minimum:.{DECIMALS}f}    "
            f"10th percentile: {perc10:.{DECIMALS}f}    "
            f"Mean: {mean_val:.{DECIMALS}f}    "
            f"90th percentile: {perc90:.{DECIMALS}f}    "
            f"Maximum: {maximum:.{DECIMALS}f}"
        )
        self.summaryLabel.setText(summary_text)

# ====================================================
# Custom Widget for Distribution Cells (Binned Line Plot)
# ====================================================

class DistributionCellWidget(QWidget):
    def __init__(self, distribution, raw_expr, parent=None):
        super().__init__(parent)
        self.distribution = distribution
        self.raw_expr = raw_expr  # e.g., "=normal(2.25, 0.5, 0.0)"
        # Increase the minimum height to avoid clipping.
        self.setMinimumHeight(80)

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        # Reserve header area (20 pixels high) for text.
        header_height = 20
        plot_rect = QRectF(rect.left(), rect.top() + header_height, rect.width(), rect.height() - header_height)

        # Draw header background.
        painter.fillRect(0, 0, rect.width(), header_height, QColor(240, 240, 240))
        painter.setPen(Qt.black)
        header_text = self._create_header_text()
        painter.drawText(5, 15, header_text)

        # Draw a sparkline using binned (histogram) data.
        try:
            samples = self.distribution.sample(5000)
        except Exception:
            samples = np.array([])
        if samples.size > 0:
            num_bins = 50
            bins = np.linspace(np.min(samples), np.max(samples), num_bins + 1)
            hist, bin_edges = np.histogram(samples, bins=bins)
            # Compute bin centers.
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            max_count = np.max(hist) if np.max(hist) > 0 else 1

            points = []
            for center, count in zip(bin_centers, hist):
                x = plot_rect.left() + ((center - bins[0]) / (bins[-1] - bins[0])) * plot_rect.width()
                y = plot_rect.bottom() - ((count / max_count) * plot_rect.height())
                points.append(QPointF(x, y))

            if points:
                pen = QPen(QColor(100, 150, 240))
                pen.setWidth(1)
                painter.setPen(pen)
                painter.drawPolyline(*points)

        painter.end()

    def _create_header_text(self):
        expr = self.raw_expr.strip()
        if expr.startswith("="):
            inner = expr[1:]
            for func in ["normal", "uniform", "exponential", "poisson"]:
                if inner.startswith(func + "("):
                    try:
                        params_str = inner[len(func)+1 : inner.index(")")]
                        params = [p.strip() for p in params_str.split(",")]
                        if func == "normal" and len(params) >= 2:
                            text = f"{func}({params[0]}, {params[1]})"
                        else:
                            text = f"{func}({', '.join(params)})"
                        mean_val = self.distribution.mean()
                        text += f"  μ: {mean_val:.{DECIMALS}f}"
                        return text
                    except Exception:
                        break
        try:
            mean_val = self.distribution.mean()
            return f"{expr}  μ: {mean_val:.{DECIMALS}f}"
        except Exception:
            return expr

    def mouseDoubleClickEvent(self, event):
        # On double click, open the detailed plot dialog.
        dialog = DetailPlotDialog(self.distribution, self.raw_expr, self)
        dialog.exec_()

# ====================================================
# Spreadsheet Widget (View) with Formula Bar and Cell Widgets
# ====================================================

class SpreadsheetWidget(QTableWidget):
    def __init__(self, rows=20, columns=10):
        super().__init__(rows, columns)
        self.setHorizontalHeaderLabels([chr(ord('A') + i) for i in range(columns)])
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.model_obj = SpreadsheetModel()
        self._updating = False
        self.cellChanged.connect(self.on_cell_changed)
        # Record default sizes.
        self.regularRowHeight = self.rowHeight(0) if self.rowCount() > 0 else 25
        self.regularColWidth = self.columnWidth(0) if self.columnCount() > 0 else 80

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            row = self.currentRow()
            col = self.currentColumn()
            if row >= 0 and col >= 0:
                self.model_obj.set_cell(row, col, "")
                self.update_cell_and_dependents(row, col)
            return
        super().keyPressEvent(event)

    def on_cell_changed(self, row, col):
        if self._updating:
            return
        self._updating = True
        try:
            item = self.item(row, col)
            if item is None:
                item = QTableWidgetItem("")
                self.setItem(row, col, item)
            expr = item.text()
            self.model_obj.set_cell(row, col, expr)
            self.update_cell_and_dependents(row, col)
        finally:
            self._updating = False

    def update_cell_and_dependents(self, row, col):
        """Update the specified cell and any dependent cells' display/widgets.
           This does NOT itself adjust row/column size; we do that in a separate pass.
        """
        value = self.model_obj.get_cell_value(row, col)
        raw_expr = self.model_obj.get_cell_expr(row, col)
        key = (row, col)

        if isinstance(value, Distribution):
            # Install a DistributionCellWidget
            widget = DistributionCellWidget(value, raw_expr)
            self.setCellWidget(row, col, widget)
            if self.item(row, col):
                self.item(row, col).setText("")
        else:
            # Remove any existing widget, show the value as text
            self.removeCellWidget(row, col)
            if isinstance(value, float):
                display_text = f"{value:.{DECIMALS}f}"
            else:
                display_text = str(value)
            self._set_table_item(row, col, display_text)

        # Update dependents as well
        if key in self.model_obj.reverse_dependencies:
            for (dep_row, dep_col) in self.model_obj.reverse_dependencies[key]:
                self.update_cell_and_dependents(dep_row, dep_col)

    def _set_table_item(self, row, col, text):
        self._updating = True
        try:
            item = self.item(row, col)
            if item is None:
                item = QTableWidgetItem(text)
                self.setItem(row, col, item)
            else:
                item.setText(text)
        finally:
            self._updating = False

    def set_current_cell_expression(self, expression):
        row = self.currentRow()
        col = self.currentColumn()
        if row < 0 or col < 0:
            return
        self.model_obj.set_cell(row, col, expression)
        self.update_cell_and_dependents(row, col)
        # After changing the cell, re-adjust row/col sizes if needed.
        self.adjust_row_col_sizes()

    def get_current_cell_expression(self):
        row = self.currentRow()
        col = self.currentColumn()
        if row < 0 or col < 0:
            return ""
        return self.model_obj.get_cell_expr(row, col)

    def clear_spreadsheet(self):
        """Clear both the model and the QTableWidget."""
        self.model_obj.clear()
        for row in range(self.rowCount()):
            for col in range(self.columnCount()):
                self.removeCellWidget(row, col)
                self.setItem(row, col, QTableWidgetItem(""))

    def refresh_all_cells(self):
        """Re-run update_cell_and_dependents on every cell, then adjust row/col sizes."""
        for row in range(self.rowCount()):
            for col in range(self.columnCount()):
                self.update_cell_and_dependents(row, col)
        self.adjust_row_col_sizes()

    def adjust_row_col_sizes(self):
        """
        Scan the entire sheet to see which rows/columns have Distribution cells.
        Any row/column that has at least one distribution cell gets enlarged;
        otherwise it is set to the default size.
        """
        distribution_in_row = [False] * self.rowCount()
        distribution_in_col = [False] * self.columnCount()

        # First pass: note which rows/cols have a Distribution
        for row in range(self.rowCount()):
            for col in range(self.columnCount()):
                val = self.model_obj.get_cell_value(row, col)
                if isinstance(val, Distribution):
                    distribution_in_row[row] = True
                    distribution_in_col[col] = True

        # Second pass: actually resize
        for r in range(self.rowCount()):
            if distribution_in_row[r]:
                self.setRowHeight(r, self.regularRowHeight * 3)
            else:
                self.setRowHeight(r, self.regularRowHeight)

        for c in range(self.columnCount()):
            if distribution_in_col[c]:
                self.setColumnWidth(c, self.regularColWidth * 2)
            else:
                self.setColumnWidth(c, self.regularColWidth)

# ====================================================
# Main Window with Formula Bar, Toolbar, and Decimal Settings
# ====================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Probability Spreadsheet")

        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        new_action = QAction("New", self)
        new_action.triggered.connect(self.new_spreadsheet)
        toolbar.addAction(new_action)

        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_spreadsheet)
        toolbar.addAction(save_action)

        load_action = QAction("Load", self)
        load_action.triggered.connect(self.load_spreadsheet)
        toolbar.addAction(load_action)

        calc_action = QAction("Calculate", self)
        calc_action.triggered.connect(self.calculate_distributions)
        toolbar.addAction(calc_action)

        # Add a textbox for number of decimals.
        self.decimalBox = QLineEdit()
        self.decimalBox.setFixedWidth(50)
        self.decimalBox.setText(str(DECIMALS))
        self.decimalBox.setToolTip("Enter number of decimals")
        toolbar.addWidget(self.decimalBox)

        # Add a FIX button.
        fix_button = QPushButton("FIX")
        fix_button.clicked.connect(self.fix_decimals)
        toolbar.addWidget(fix_button)

        self.formulaBar = QLineEdit()
        self.formulaBar.setPlaceholderText("Enter a value, formula, or distribution (e.g., =normal(2.25, 0.5, 0.0))")
        self.formulaBar.returnPressed.connect(self.on_formula_entered)

        self.spreadsheet = SpreadsheetWidget(rows=20, columns=10)
        self.spreadsheet.itemSelectionChanged.connect(self.on_selection_changed)

        layout = QVBoxLayout()
        layout.addWidget(self.formulaBar)
        layout.addWidget(self.spreadsheet)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def on_formula_entered(self):
        expr = self.formulaBar.text()
        self.spreadsheet.set_current_cell_expression(expr)
        # Sync the formula bar with the final stored expression
        self.formulaBar.setText(self.spreadsheet.get_current_cell_expression())

    def on_selection_changed(self):
        expr = self.spreadsheet.get_current_cell_expression()
        self.formulaBar.setText(expr)

    def new_spreadsheet(self):
        if QMessageBox.question(self, "Confirm New", "Clear the spreadsheet?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.spreadsheet.clear_spreadsheet()

    def save_spreadsheet(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Spreadsheet", "", "XML Files (*.xml)")
        if filename:
            xml_str = self.spreadsheet.model_obj.to_xml()
            try:
                with open(filename, "w") as f:
                    f.write(xml_str)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file:\n{e}")

    def load_spreadsheet(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Spreadsheet", "", "XML Files (*.xml)")
        if filename:
            try:
                with open(filename, "r") as f:
                    xml_str = f.read()
                # Clear the old spreadsheet to avoid collisions/crashes
                self.spreadsheet.clear_spreadsheet()
                # Load new data and refresh
                self.spreadsheet.model_obj.from_xml(xml_str)
                self.spreadsheet.refresh_all_cells()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")

    def calculate_distributions(self):
        """
        Now recalculates the entire spreadsheet, including all
        distributions and dependencies.
        """
        self.spreadsheet.model_obj.recalc_all_cells()
        self.spreadsheet.refresh_all_cells()

    def fix_decimals(self):
        global DECIMALS
        text = self.decimalBox.text().strip()
        try:
            new_decimals = int(text)
            if new_decimals < 0:
                raise ValueError("Negative value")
            DECIMALS = new_decimals
            self.spreadsheet.refresh_all_cells()
        except Exception as e:
            QMessageBox.warning(self, "Invalid Input", f"Please enter a non-negative integer.\nError: {e}")

# ====================================================
# Main Application
# ====================================================

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(900, 600)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
