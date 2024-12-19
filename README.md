# housing-price-prediction-ML
# Housing Prices Prediction Project

This project uses machine learning techniques to predict housing prices based on various features. The dataset is fetched, prepared, and analyzed in a Jupyter Notebook, with Random Forest Regression used as the primary model. Cross-validation is implemented to evaluate the model's performance.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Dependencies](#dependencies)
5. [Running the Project](#running-the-project)
6. [Results](#results)
7. [Credits and Acknowledgments](#acknowledgments)

---

## Project Overview

This project demonstrates:

- Fetching and extracting data from a remote source.
- Data cleaning, feature engineering, and exploratory data analysis.   
- Building and evaluating a machine learning model using Random Forest Regression.
- Performing cross-validation to measure model performance.

---

## Dataset

The dataset used for this project is the 'California Housing Prices' dataset from the StatLib repository. This dataset was based on data from the 1990 California cen
sus.
It has been added to this GitHub repository and includes housing data stored in a CSV file, which is automatically downloaded and extracted during the notebook's execution.

### Data Features

The dataset contains:

- Various housing-related features such as location, price, size, and more.
- Labels (target variable) representing housing prices.

---

## Project Structure

The project is organized as follows:

- **Jupyter Notebook**: Contains all the code for data loading, preprocessing, model building, and evaluation.
- **Dataset**: Downloaded and extracted automatically into the `dataset` directory of your current workspace.

---

## Dependencies

To run this project, ensure you have the following installed:

- Python 3.8+
- Jupyter Notebook
- NumPy
- Pandas
- Scikit-learn
- tarfile (standard library)
- six (for compatibility)

Install dependencies using pip:

```bash
pip install numpy pandas scikit-learn
```

---

## Running the Project

1. Clone the GitHub repository:

   ```bash
   git clone https://github.com/kitkat1424/housing-price-prediction-ML.git
   ```

2. Navigate to the project directory:

   ```bash
   cd housing-price-prediction-ML
   ```

3. Open the Jupyter Notebook:

   ```bash
   jupyter notebook housing_prices.ipynb
   ```

4. Run all cells in the notebook to:

   - Fetch and extract the dataset.
   - Perform data preprocessing.
   - Train and evaluate the model.

---

## Results

The project tests Linear Regression, Decision Tree Regressor and Random Forest Regression on the training data set. It settles on the use of Random Forest Regression to predict housing prices. Model performance is evaluated using:

- **RMSE (Root Mean Squared Error)**: Evaluated on both training data and cross-validation folds.
- **Cross-validation**: Ensures robust evaluation by splitting the dataset into multiple training and test sets. The mean RMSE is then computed.

---

## Acknowledgments

- Geron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (2nd ed.). O'Reilly Media.
- Libraries such as Scikit-learn, Pandas, and NumPy were instrumental in building this project.

---

