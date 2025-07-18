![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge\&logo=pandas\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge\&logo=numpy\&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-854FF5?style=for-the-badge\&logo=scipy\&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge\&logo=matplotlib\&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-14637D?style=for-the-badge\&logo=seaborn\&logoColor=white)
![ipywidgets](https://img.shields.io/badge/ipywidgets-FFB000?style=for-the-badge\&logo=jupyter\&logoColor=white)

---

# Interactive Bayesian A/B Test Toolkit 📊

This repository contains a single, self-contained Jupyter Notebook that walks you through every step of a **Bayesian A/B test** — from uploading raw data to receiving an automated, risk-aware recommendation. An intuitive control panel built with **ipywidgets** lets you set priors, choose your risk tolerance, and upload your experiment data without touching any code.

---

## What is this?

* **A reusable notebook** that takes an Excel file with three columns (`variant`, `reach`, `conversion`) and returns:

  1. Posterior distributions for each variant
  2. The probability each variant is best
  3. The expected loss (business risk) of picking any variant
  4. A plain-English recommendation you can paste directly into a report or Slack message

* **Interactive UI**: file-upload widget, prior selector (Jeffreys vs Uniform), and a risk-tolerance slider so analysts and non-analysts alike can run the test.

---

## Why Bayesian A/B Testing? 🤔

Traditional A/B testing relies on p-values, which answer the question *“How surprising is my data if nothing is happening?”* — not very helpful for product managers.

A Bayesian workflow:

1. Gives you the **probability a variant is truly better**.
2. Quantifies **expected loss** — how much you’d give up if you chose the wrong variant.
3. Lets teams decide up-front how much risk (`RISK_THRESHOLD`) they are willing to accept, creating an immediate bridge between statistics and business objectives.

---

## How it Works 🧪

1. **Prior Belief**
   Select either a *Jeffreys prior* (0.5, 0.5) — recommended for most tests — or a *Uniform prior* (1, 1).

2. **Data Upload**
   Drag-and-drop your `.xlsx` file in the control panel. The notebook validates column names and shows the first five rows.

3. **Posterior Calculation**
   Each variant’s posterior is `Beta(prior_alpha + conversions, prior_beta + failures)`.

4. **Visualization**
   Posterior ridgeline plot with shaded 95 % credible intervals for instant visual intuition.

5. **Monte Carlo Simulation**
   Draws **100 000 random samples** per variant to compute:

   * **Probability to be Best**
   * **Expected Loss (Risk)**

6. **Automated Recommendation**
   If the lowest risk ≤ `RISK_THRESHOLD`, the notebook prints:

   ```
   ✅ Verdict: Deploy Variant 'B'.
   Risk: 0.0144 (< 1% threshold).
   ```

   Otherwise, it advises collecting more data.

---

## How to Use 🚀

1. **Clone the repo**

   ```bash
   git clone https://github.com/dsandux/AB-Test-Toolkit
   cd AB-Test-Toolkit
   ```

2. **Prepare your data**

You must put your date in an Excel file like the example below:

   | variant | reach  | conversion |
   | ------- | ------ | ---------- |
   | A       | 10 000 | 950        |
   | B       | 10 050 | 1 020      |

   Save as **`data_ab.xlsx`** in the project root.

3 **Run the notebook**

   ```bash
   jupyter lab AB_Test_Toolkit.ipynb
   ```

5. **Use the Control Panel**

   1. Upload `data_ab.xlsx`
   2. (Optional) Change the prior or risk tolerance
   3. Click **Confirm Selections**
   4. Run the remaining cells top-to-bottom

The final cell prints the recommendation and shows a neatly formatted results table.

---

## Open-Source License

Distributed under the **GNU General Public License v3.0**. You’re free to use, modify, and distribute this project, provided derivatives remain under the same license. See the `LICENSE` file for full details.
