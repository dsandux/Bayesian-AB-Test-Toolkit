![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-854FF5?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-14637D?style=for-the-badge&logo=seaborn&logoColor=white)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=for-the-badge)](https://www.gnu.org/licenses/gpl-3.0)
-----

# Bayesian A/B Test Analysis 📊

[](https://www.gnu.org/licenses/gpl-3.0)

This project provides a complete, end-to-end framework for analyzing A/B test results using **Bayesian statistics**. It moves beyond traditional p-values to deliver more intuitive and actionable business insights. The entire analysis is contained within a single, well-documented Jupyter Notebook.

-----

## What is this?

This is a reusable tool that takes raw A/B test data (number of visitors and conversions for each variant) and produces a clear, data-driven recommendation. The output isn't just a "winner" but also quantifies the certainty of the result and the business risk associated with the decision.

The final output is an automated conclusion, like this:

```
--- Automated Recommendation ---
Decision Threshold (Max Acceptable Risk): 0.0100

✅ Recommendation: Implement Variant 'B'.
Justification: This variant has the lowest risk (0.000144), which is below the acceptable threshold.
```

-----

## Why Bayesian A/B Testing? 🤔

Traditional A/B testing often relies on p-values, which can be confusing. A p-value tells you the probability of seeing your data *if there was no real difference between variants*. This doesn't directly answer the questions business stakeholders have.

A **Bayesian approach** is more direct and intuitive. It allows us to answer questions like:

  * "What is the **probability that Variant B is better** than Variant A?"
  * "If we choose Variant B, what is our **expected loss (or risk)** if it isn't actually the best?"

This framework provides two key metrics that are much more useful for decision-making:

1.  **Probability of Being Best**: The likelihood that a variant is the true winner.
2.  **Expected Loss**: The average "cost of being wrong." It quantifies the risk of choosing a variant if it's not the optimal one.

By focusing on risk, we can make decisions based on a pre-defined risk tolerance (`RISK_THRESHOLD`), aligning statistical analysis directly with business objectives.

-----

## How it Works 🧪

The notebook follows a clear statistical methodology:

1.  **Prior Belief**: It starts with an uninformative **Beta(1, 1) prior**, which assumes any conversion rate between 0% and 100% is equally likely before seeing the data.

2.  **Data Loading**: The script loads the test data (visitors and conversions) from an Excel file.

3.  **Posterior Calculation**: It uses the observed data to update the prior beliefs, resulting in a **posterior Beta distribution** for each variant. This distribution represents our updated understanding of each variant's true conversion rate.

4.  **Visualization**: The posterior distributions are plotted, giving a clear visual representation of each variant's performance and the certainty of our estimate. A variant with a taller, narrower curve is estimated with more certainty.

5.  **Monte Carlo Simulation**: To calculate our metrics, the script runs a Monte Carlo simulation, drawing `100,000` random samples from each variant's posterior distribution.

6.  **Metrics Calculation**: Using the simulation results, it calculates the **Probability of Being Best** and the **Expected Loss (Risk)** for each variant.

7.  **Automated Decision**: Finally, it compares the lowest risk variant against the predefined `RISK_THRESHOLD` to provide a clear, automated recommendation.

-----

## How to Use It 🚀

This project is designed to be simple to use.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/dsandux/AB-Test-Toolkit
    ```
2.  **Install libraries:** Make sure you have the required Python libraries installed: `pandas`, `numpy`, `scipy`, `matplotlib`, and `seaborn`.
3.  **Prepare your data:**
      * Create an Excel file named `data_ab.xlsx`.
      * Ensure it contains three columns: `variante`, `visitantes`, and `conversoes`.
      * Place this file in the same directory as the Jupyter Notebook.
4.  **Run the notebook:** Open the `.ipynb` file and run the cells from top to bottom. The final cell will output the automated recommendation. You can adjust the `RISK_THRESHOLD` in the parameter definition cell to match your business's risk tolerance.

-----

## Open Source License

This project is open source and available under the **GNU General Public License v3.0**. Feel free to use, modify, and distribute it. See the `!LICENSE` file for more details.
