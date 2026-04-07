# Brain-Hemorrhage-Segmentation-App
An app for exploring results of predicted binary masks of hemorrhages based on CT scans of the brain with true binary masks.

## Overview


In this project a machine learning pipeline was developed for brain hemorrhage analysis from CT scans, and combined methods of computer vision and image segmentation models as well as traditional machine learning approaches. The project focused both on predicting binary masks correlated with five subtypes of hemorrhages as well as hemorrhage classification. At its end, an interactive web application was deployed to visualize results of the project.


## Objectives


The objective of the project was to segment hemorrhage regions in brain CT scans and compare the performance results of both U-Net and ResU-Net segmentation models. Additionally, traditional machine learning models such as linear regression, Random Forest, etc., were used for hemorrhage classification, leading to the analysis of challenges such as data leakage, class imbalance, and irrelevant engineered features. The final goal was to create an interactive interface for model exploration.


## Dataset

The dataset for this project was provided by the company Zeta Surgical, and was composed of CT brain scan images organized into multiple hemorrhage categories along with accompanying features such as polygon-related data in CSV format. Each scan had multiple views (such as brain window, bone window, etc.), and labels included hemorrhage type classifications as well as the quality of the label.


## Methodology - Segmentation


For the segmentation leg of this project, first polygon annotations were parsed from the dataset CSV files. The coordinates were converted into binary masks using OpenCV, and multi-view image inputs (stacked channels) were constructed, too. From there, a TensorFlow pipeline was built and a U-Net model was configured, and the model was then trained for segmentation and the task of predicting binary masks. After multiple iterations of the U-Net model were deployed, ResU-Net with residual connections rather than standard convolution layers was also implemented in an attempt to improve model accuracy. Model accuracy was evaluated using Dice coefficient and IoU. 


## Methodology - Classification


For the hemorrhage classification portion of the project, first, multiple CSV files were combined into a unified dataset and hemorrhage type labels were created at the row level for every associated image. Next, label inconsistencies were addressed (deciding whether to use 'Majority Label,' considered the best, or 'Correct Label' if no Majority Label was available), and features were engineered from the previously created segmentation masks, such as number of polygons per brain image, area statistics, and spatial and geometric features. The issue of data leakage was resolved by converting data from image-level to row-level, and multiple machine learning models (logistic regression, linear models, LDA/QDA, Neural Network/MLP, Random Forest) were trained with the task of accurately predicting hemorrhage type.


## Models Used


- U-Net (binary cross entropy loss/BCE)
- U-Net + BCE + Dice + IoU
- ResU-Net
- Logistic Regression
- Linear Regression (OLS, Ridge, Lasso)
- LDA/QDA
- Random Forest
- Neural Network/MLP


## Results


The BCE-only U-Net model failed in its creation of predictive masks due to accuracy being geared towards pixel-level instead of mask overlap, meaning the predicted probability of each pixel being a hemorrhage was low. At higher thresholds this resulted in blank masks, and at lower thresholds, noisy masks with irrelevant polygon area. Dice-based loss significantly improved image segmentation quality in the U-Net model, though the ResU-Net model showed negligible improvement.


The classification models initially demonstrated inflated performance due to probable data leakage as well as the deletion of polygon-related features during data cleaning. Once data leakage was fixed, however, and a host of engineered features were introduced, model performance dropped significantly, highlighting the role data leakage plays in falsely high accuracy as well as demonstrating that the engineered spatial features, while assumed to be relevant, were not strong enough to enhance predictions of hemorrhage type. 


## Interactive App


An application was built using Streamlit that allows users to:


- View CT scans
- Compare true and predicted masks as well as probability maps
- Switch between U-Net and ResU-Net models
- Adjust probability thresholds to observe their impact on mask quality


It is deployed in the cloud for public access. The link is as follows:


https://brain-hemorrhage-segmentation-app-fxuflb7mf9tyusncpubyjp.streamlit.app/


<img width="1821" height="898" alt="Screenshot 2026-04-07 at 2 09 37 PM" src="https://github.com/user-attachments/assets/5622c647-3f94-43b8-985f-2ddcd437028a" />



## How to Run Locally


- Clone repository
- Install dependencies from requirements.txt
- Run: streamlit run brain_hemorrhage_segmentation_explorer.py
- Open browser at local host


## Project Structure


- app_data -> folder with sample cases for visualization
- brain_hemorrhage_segmentation_explorer.py -> Streamlit app
- Notebooks -> model deployment
- Data preprocessing scripts


## Future Improvements


- Improve segmentation with larger datasets 
- Apply more advanced model architecture (attention U-Net, transformers)
- Improve feature engineering for classification
- Address class imbalance more effectively (class weights, resampling, metrics such as F1 score, recall, precision)
- Train models for longer and more epochs
- Deploy real-time inderence instead of precomputed results
