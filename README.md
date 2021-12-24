# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

The objective of this project is to transform a python code inside a notebook into a production-ready code.
Best practice of software engineering were applied, including:
- clean and modular code
- refactoring code
- optimizing code for efficiency
- writing documentation
- PEP8 & Linting
- Catching errors
- Writing tests & logs

Dependencies and libraries
- scikit-learn
- shap
- pylint
- autopep8
- matplotlib
- seaborn

## Running Files
1. Create a virtual enviroment

   ```
   $ pip install virtualenv
   ```
   
   ```
   $ virtualenv .venv
   ```
    
2. Activate the environment

    ```
   $ source .venv/bin/activate
   ```

3. Install dependencies and libraries

    ```
   $ pip install -r requirements.txt
   ```

4. Run churn_library.py

   ```
   $ python churn_library.py
   ```

5. For testing and logging run churn_script_logging_and_tests.py  

   ```
   $ python churn_script_logging_and_tests.py
   ```

6. Check log file in logs folder  

   ```
   $ cat logs/churn_library.log
   ```
