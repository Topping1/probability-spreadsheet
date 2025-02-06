# Probability-Spreadsheet: A Probabilistic Spreadsheet
A simple proof-of-concept probabilistic spreadsheet. Python recreation of Montesheet

**Probability-Spreadsheet** is an experimental spreadsheet application built with PyQt5 and Python, allowing you to seamlessly mix **probability distributions** with classical spreadsheet-style arithmetic. It is a Python recreation of [Montesheet](https://github.com/dps/montesheet) See also [Probabilistic Spreadsheet](https://blog.singleton.io/posts/2021-11-24-probabilistic-spreadsheet/). This project is NOT affiliated with Montesheet.

## Screenshots

![image](https://github.com/user-attachments/assets/fc15065d-203b-47ea-9fa5-d28123a85cb4)

![image](https://github.com/user-attachments/assets/41193654-ec9c-4e14-81e0-f15765d445c1)


## Features

- **Basic Spreadsheet Operations**  
  - Type values or formulas into cells (e.g., `=A1 + B2`)  
  - Cell references: refer to a cell by its column letter + row number (`A1`, `B3`, etc.)  
  - Standard arithmetic and referencing of other cells

- **Probabilistic Distributions**  
  - Use built-in distribution functions like `normal`, `uniform`, `exponential`, and `poisson`  
  - Combine them with normal arithmetic: `=normal(10,2) + uniform(0,5)`  
  - Each distribution cell automatically displays a small **sparkline** of its distribution  
  - Double-click a distribution cell to see a **detailed histogram plot** with summary statistics

- **Monte Carlo Engine**  
  - Under the hood, each distribution is sampled (e.g., 5,000 times by default)  
  - The distribution’s mean and sparkline are shown in the cell header  
  - You can recalculate the entire spreadsheet to regenerate new random samples  
  - Composite cells referencing distributions will also update

- **Dependency Tracking**  
  - Changing any cell (or distribution) automatically invalidates dependent cells  
  - The spreadsheet re-evaluates only what is necessary  
  - On “Calculate,” the entire network of cells is re-sampled and re-evaluated

- **Automatic Sizing**  
  - Cells containing distributions are automatically resized (larger height and width)  
  - Regular numeric or text cells remain in their default size  
  - The row or column containing at least one distribution stays large, even if other cells in that row or column are numeric

- **Load/Save via XML**  
  - Save your spreadsheet to an XML file (with `File → Save`)  
  - Load a previously saved spreadsheet via `File → Load`  
  - The distribution formulas (e.g. `=normal(10,2)`) and references are all restored from the XML

- **Decimal Precision**  
  - A small text box in the toolbar lets you specify the number of decimals to display (default is 3)  
  - Press the **FIX** button to apply the new precision to all cells

## Installation & Usage

1. **Install Requirements**  
   - Python 3.x  
   - `pip install pyqt5 matplotlib numpy`

2. **Run the Script**  
   ```bash
   python probability-spreadsheet.py
   ```
   A PyQt5 window should appear with 20 rows × 10 columns, a formula bar, and a toolbar.

3. **Entering Data**  
   - Click a cell to select it. In the formula bar at the top, type either:
     - A plain number, e.g. `123.45`
     - A formula referencing another cell, e.g. `=A1 + 2.5`
     - A distribution formula, e.g. `=normal(10,2,0,20)` (normal distribution truncated between 0 and 20)  
   - Press **Enter** or click away to set the cell.

4. **Recalculation**  
   - Click **Calculate** on the toolbar to force the entire spreadsheet to re-sample distributions and update dependent cells.

5. **Double-Click on Distribution Cells**  
   - Any cell showing a sparkline (which means it contains a distribution) can be double-clicked  
   - A dialog opens, displaying a histogram and key statistics (min, 10th percentile, mean, 90th percentile, max)

6. **Saving and Loading**  
   - **Save**: Choose “Save” in the toolbar, select an XML file location  
   - **Load**: Choose “Load” in the toolbar, pick an XML file that was previously saved. The script will restore your spreadsheet’s expressions.

## Built-in Distribution Functions

You can use the following distribution functions inside your spreadsheet formulas:

- **normal(mean, std, min, max)**  
  - Creates a normal distribution with given `mean` and `std`.  
  - Optional `min` and `max` parameters let you truncate the distribution.  
- **uniform(min, max)**  
  - Uniform distribution in [min, max].  
- **exponential(rate)**  
  - Exponential distribution with the given rate (mean = 1/rate).  
- **poisson(mean)**  
  - Poisson distribution with given mean.  
- **mmax(x, y)**  
  - Takes the element-wise maximum of distributions (or numbers) x and y.  
- **choose(x, y)**  
  - Randomly picks x or y (each with 50% chance) at each sample.  

Additionally, you can call:
- **mean(x)** — returns the minimum, mean, or maximum of distribution `x` as a single scalar.  
- **min(x)**  
- **max(x)**  
- **perc(x, p)** — returns the *p*-th percentile (0–100) of distribution `x`.  

## Example Calculation

Here’s a simple demonstration:

1. In **A1**, type:
   ```
   =normal(10, 2)
   ```
   This cell will contain a Normal(μ=10, σ=2) distribution.

2. In **B1**, type:
   ```
   =uniform(0, 5)
   ```
   This cell will contain a Uniform(0, 5) distribution.

3. In **C1**, type:
   ```
   =A1 + B1
   ```
   This will be the sum of the Normal and Uniform distributions. A sparkline will appear showing the distribution of the sum. 

4. Click **Calculate** to resample. Cells **A1**, **B1**, and **C1** will update to show new sparkline shapes and mean values.  
   - If you double-click **C1**, you’ll see a histogram of the sum.  

5. Try referencing a single statistic in another cell. For instance, in **D1**, type:
   ```
   =mean(C1)
   ```
   This will convert the distribution in **C1** into a single numeric mean value.  

6. Click **Calculate** again, and you’ll see **D1** update with the new mean of the sum distribution.

## Contributing

If you find a bug or want to add features, feel free to open issues or submit pull requests on GitHub.

## License

This project is provided “as-is” under the MIT License. Feel free to use and modify it as you wish.
