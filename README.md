# Credit-Card-Fraud-Detection-Comparative-Analysis
 Analysing Different Machine Learning, 1D CNN and MLP algorithms for Credit Card fraud detection and implementing a dashboard for visualisation using Python and Streamlit

 ## General Discussion Regarding the wishfulness for this analysis
 This study explores a range of supervised learning and deep learning techniques to identify the most effective anomaly detection model for credit card fraud. Five supervised learning algorithms are evaluated:   Random Forest, K-Nearest Neighbours (KNN), Support Vector Machine (SVM), Gradient Boosting, and Logistic Regression, along with two deep learning approaches: Multilayer Perceptron(MLP) and 1D Convolutional Neural Networks (CNN).

 Furthermore, a Streamlit dashboard is developed to visualise model performance, including the ROC curves, allowing stakeholders to interactively explore each algorithm’s effectiveness. This dashboard offers a practical tool for real-time monitoring and comparison of fraud detection models. The ultimate goal is to identify the most accurate and reliable model for detecting fraudulent transactions while providing a user-friendly platform to assist financial institutions in decision making and fraud prevention efforts.


 ## Initial Look at the STreamlit Dashboard to get a broader view of what the aim was

### Uploading the file to be studied and visualised
 ![home](https://github.com/user-attachments/assets/e24d1f70-1a95-4f02-9310-dd38bfc43683)
### Choosing the algorithm and method to be evaluated
![choice](https://github.com/user-attachments/assets/2fd387bb-7640-4c98-94b2-f0b61934fd35)
### Evaluation Metrics of the method measured
 ![CLass Distribution](https://github.com/user-attachments/assets/75169010-e6f7-44c0-989e-e4fd980c9043)
![cf](https://github.com/user-attachments/assets/8bfcded0-92e0-4fa8-91a2-e6e6b2e903dd)
![classification report](https://github.com/user-attachments/assets/c6dd76a2-cdd7-44f1-9184-d27988a62d3d)




 ### Installation Command for the prerequisite libraries to run the Streamlit App

 Python Requirement - 3.8 or higher 3.12 recommended (Evaluation was done primarily on Python 3.12)

 pip install -r resources.txt

 ### Command to Run the Streamlit App

 streamlit run app4.py (As for this example the filename is 'app4.py')

 ### Generalised command

 streamlit run <app_name>.py
