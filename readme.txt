
Artificial Intelligence Project

The most used method of testing for COVID is using a swab to extract a sample and then test that sample for the presence of virus. Another method is to take a blood test and look for the antibodies in the blood sample. Either of these methods take a couple of days or more depending on the priority. Chest X-ray (CXR) is a relatively cheap and accessible method for examining various lung problems. Hence why the CXR images are so readily available as completed datasets. The approach used in this project and the skeleton adopted are from the project: Abolfazl Zargari Khuzani, Morteza Heidari, Ali Shariati, "COVID-Classifier: An automated machine learning model to assist in the diagnosis of COVID-19 infection in chest x-ray images," medRxiv, doi: https://doi.org/10.1101/2020.05.09.20096560, 2020

The dataset was taken from here: http://14.139.62.220/covid_19_models/

We took 1,950 samples of COVID-19, 416 of Pnuemonia and 192 CXRs of healthy lungs. The dataset is filtered on some of the following criterias: Must be adult, clarity of lung fluids, must be PA as it's usually preferred in terms of image quality and is more common so introducing unnecessary idiosyncrasies in a dataset may prove counterproductive in terms of extracting pattern from the data.

A GUI is implemented on which we can upload a CXR image to check whether it is COVID-19 infected or not.

Steps:

Run preprocess_images.py 


Then run extract_features.py which extract texture features from our image dataset. 

Run evaluate_features.py 


Then run train_model.py to train the neural net 



Finally run GUI.py and select any CXR image to classify it. (Change the path in predict_model.py accordingly.)



A detailed analysis of the code is included in the report.

The research paper on this COVID classifier is also included.

