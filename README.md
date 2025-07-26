![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-854FF5?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-14637D?style=for-the-badge&logo=seaborn&logoColor=white)
![ipywidgets](https://img.shields.io/badge/ipywidgets-FFB000?style=for-the-badge&logo=jupyter&logoColor=white)

# Bayesian Bayesian A/B Test Toolkit 📊

This repository contains a self-contained notebook that provides an end-to-end workflow for conducting a **Bayesian A/B test**. It guides you from uploading raw data to receiving an automated, risk-aware recommendation, all managed through an intuitive control panel. No need to modify any code to run your analysis.

This toolkit is available in two versions to fit your preferred environment.

---

## Key Features

* **Interactive UI**: An easy-to-use control panel allows you to upload your data, select a Bayesian prior, and define your business's risk tolerance with simple widgets.
* **Automated Reporting**: Generates a final report in plain English, including a clear verdict, a summary for stakeholders, and an explanation of the key metrics.
* **Risk-Aware Decisions**: Instead of relying on p-values, the analysis centers on **Expected Loss**, quantifying the business risk of choosing any given variant.
* **Rich Visualizations**: Automatically plots the posterior distributions for each variant, providing immediate visual intuition about performance and uncertainty.

---

## Two Versions Available

To ensure maximum accessibility, this toolkit is offered in two formats:

1.  **Jupyter Notebook**: Ideal for running on a local machine. The `AB_Test_Toolkit.ipynb` file uses standard `ipywidgets` for the control panel.
2.  **Google Colab**: Perfect for running in the cloud with zero setup. This version is adapted to use Colab's native file handlers for a smooth user experience.

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GwC2Z1jYcdkw-ci1Mdy3EVkmptg8uH5-?usp=sharing)

---

## Why Bayesian A/B Testing? 🤔

Traditional A/B testing, which relies on p-values, answers the question: *“Is it surprising to see this data if the variants are identical?”* This is often not what a business leader wants to know.

A Bayesian approach directly answers the questions that matter:

1.  **What is the probability that Variant B is actually better than Variant A?**
2.  **If I choose Variant B, how much do I stand to lose if it's secretly the wrong choice?**
3.  **Does this result meet the risk threshold our team agreed on beforehand?**

This toolkit bridges the gap between statistical analysis and business objectives by focusing on probability and risk.

---

## How the Notebook Works: An End-to-End Flow 🧪

The notebook is structured as a sequence of cells, each performing a specific task.

1.  **Setup**: Installs and imports all necessary Python libraries.
2.  **Control Panel**: Displays an interactive UI for the user to upload their `.xlsx` data file and configure the analysis parameters (Bayesian prior and risk tolerance). The UI is adapted for either Jupyter or Google Colab.
3.  **Data Loading & Validation**: Loads the uploaded data into a pandas DataFrame and validates that it contains the required columns: `variant`, `reach`, and `conversion`.
4.  **Posterior Calculation**: Applies the Beta-Binomial conjugate update rule to combine the selected prior with the observed data, calculating the posterior distribution (`posterior_alpha` and `posterior_beta`) for each variant.
5.  **Visualization**: Generates and displays a ridgeline plot of the posterior distributions, including shaded 95% credible intervals for easy visual comparison.
6.  **Monte Carlo Simulation**: Draws 100,000 random samples from each variant's posterior distribution to create an empirical dataset for metric calculation.
7.  **Metrics Calculation**: Uses the simulation data to compute the two key business metrics: **Probability to be Best** and **Expected Loss (Risk)**.
8.  **Automated Report & Recommendation**: Interprets the results and generates a final, formatted Markdown report with a clear verdict, a summary for stakeholders, and an explanation of the expected uplift.

---

## How to Use 🚀

### Option A: Local Jupyter Notebook

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/dsandux/AB-Test-Toolkit](https://github.com/dsandux/AB-Test-Toolkit)
    cd AB-Test-Toolkit
    ```
2.  **Prepare your data** as an `.xlsx` file with the columns `variant`, `reach`, and `conversion`. An example `data_ab.xlsx` is included.
3.  **Run the notebook:**
    ```bash
    jupyter lab AB_Test_Toolkit.ipynb
    ```
4.  **Use the Control Panel** in the second cell to upload your file and confirm settings.
5.  Run the remaining cells from top to bottom.

### Option B: Google Colab

1.  **Click the "Open in Colab" badge** above or use [this link](https://colab.research.google.com/drive/1GwC2Z1jYcdkw-ci1Mdy3EVkmptg8uH5-?usp=sharing).
2.  **Run the cells** from top to bottom.
3.  When you run the "Control Panel" cell, a file upload button will appear. Use it to upload your `.xlsx` data file.
4.  The analysis will proceed automatically after the upload is complete.

---

## Data Format

Your data must be in an Excel file (`.xlsx`) with the following structure:

| variant | reach  | conversion |
| ------- | ------ | ---------- |
| A       | 10000  | 950        |
| B       | 10050  | 1020       |
| C       | 9980   | 965        |

---

## License

Distributed under the **GNU General Public License v3.0**. You’re free to use, modify, and distribute this project, provided derivatives remain under the same license. See the `LICENSE` file for full details.
