import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline

from library_files.utils import *

class NLP_Pred: 
    """
    NLP model used to predict values such as vendor names or city and states from 
    a list of values provide via a csv file. 

    Parameter: 
        - _df: training dataset dataframe
        - test_size: by default is set to 0.2
        - random_state: by default is set to 12272023
        - n_estimator: by default is set to 100
        - *args: used to pass in column names from training dataset - can pass in as many
          as required

    """

    @validate_df(columns=set()) # columns=set() argument does not check for specfiic column names
    def __init__(self, _df:pd.DataFrame, test_size=0.2, random_state=12272023, n_estimator=100, *args, **kwargs):
        self.test_size = test_size
        self.random_state = random_state
        self.n_estimator = n_estimator
        self.labels = args
        self.data = _df

    def nlp_train_model(self, accuracy_show=False): 
        """
        Uses the validate_df decorator to verify the training dataset file 
        by first checking that a dataframe was passed in as a parameter. 

        Parameters: 
        - accuracy_show: can set either to True or False. Default is set to False 
          and is used to show the accuracy scores

        Returns: 
        - numpy array
        
        """
        
        # encode the training dataset column into numerical labels
        label_col = self.labels[0]
        feature_col = self.labels[1]

        # label_encoder = LabelEncoder()
        self.label_encoder = LabelEncoder()
        # self.data['predicted_labels'] = label_encoder.fit_transform(self.data[label_col])
        self.data['predicted_labels'] = self.label_encoder.fit_transform(self.data[label_col])

        # split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
        self.data[feature_col], 
        self.data['predicted_labels'], 
        test_size=self.test_size, 
        random_state=self.random_state
        )

        # create a pipeline with TF-IDF Vectorization and RandomForestClassifier
        # n_estimators by using 50,100 and 300, the results are no significant change. 
        # Choosing n_estimators = 100 as an optimal model.
        self.model = make_pipeline(
            TfidfVectorizer(),
            RandomForestClassifier(n_estimators=self.n_estimator)
            )

        # train the model with the training dataset
        self.model.fit(X_train, y_train) 
        y_pred = self.model.predict(X_test)

        # generate a classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        accuracy = report['accuracy']
        macro_precision = report['macro avg']['precision']
        macro_recall = report['macro avg']['recall']
        macro_f1_score = report['macro avg']['f1-score']
        weighted_precision = report['weighted avg']['precision']
        weighted_recall = report['weighted avg']['recall']
        weighted_f1_score = report['weighted avg']['f1-score']

        if accuracy_show: 
            print(f'accuracy: \t{accuracy:.2f}')
            print(f'macro avg: \t{macro_precision:.2f} {macro_recall:.2f} {macro_f1_score:.2f}')
            print(f'weighted avg: \t{weighted_precision:.2f} {weighted_recall:.2f} {weighted_f1_score:.2f}')

        return self.model

    def predict_vendor_name(self, new_vendor_key):

        if not hasattr(self, 'model'):
            self.model = self.nlp_train_model(accuracy_show=False)
        new_vendor_key = new_vendor_key.lower()
        predicted_label = self.model.predict([new_vendor_key])[0]

        # Ensure label_encoder is a fitted class attribute
        predicted_name = self.label_encoder.inverse_transform([predicted_label])[0]
        
        return predicted_name