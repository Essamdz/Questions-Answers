import pandas as pd
import re
from sklearn.metrics import f1_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from numpy.linalg import norm
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer 
      

class QuestionAnswers:
    
    def __init__(self,file):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3))#, stop_words='english' )  
        self.file=file


    def convert_xml_to_dataframe(self):
        try:
            xml_data = open(self.file, 'r').read()  # Read file
            root = ET.XML(xml_data)  # Parse XML
        except Exception:  
            raise Exception("Cannot read the xml file or not exist")
            
        all_data=[]
        Columns_name=[]
        for i, child in enumerate(root):
            data_in_one_row=[]
            for subchild in child:
                if subchild.tag=="MetaInfo":
                    data_in_one_row.append(subchild.get('TaskID'))
                elif subchild.tag=="Annotation":
                    data_in_one_row.append(subchild.get('Label'))
                else:
                    data_in_one_row.append(subchild.text)
                if i==0:
                    Columns_name.append(subchild.tag)
                    
            all_data.append(data_in_one_row)
    
        df_all_data = pd.DataFrame(all_data)  # Write in DF and transpose it
        df_all_data .columns = Columns_name  # Update column names
        if len(df_all_data)==0:
            raise Exception("No Data to be processed")
        return df_all_data 
    
    def construct_features(self,ReferenceAnswers:list):
        vector=self.vectorizer.fit_transform(list(ReferenceAnswers))
        return vector
    
    def extract_labels_from_Annotation_column(self,annotation:str):
        annotation_to_numbers=re.findall("\d",annotation)
        return annotation_to_numbers.index("1")+1
    
    
    def add_label_column_to_dataframe(self,df_all_data):
        df_all_data['Label']=df_all_data.Annotation.apply(self.extract_labels_from_Annotation_column)
        return df_all_data
    
    def convert_ReferenceAnswers_as_list_of_answers(self,ReferenceAnswers:str):
        list_of_answers=re.sub("\d:","",ReferenceAnswers)[1:-1].split("\n")
        return list_of_answers
    
    def add_list_of_answers_as_new_columns(self,df_all_data):
        df_all_data['list_of_answers']=df_all_data.ReferenceAnswers.apply(self.convert_ReferenceAnswers_as_list_of_answers)
        return df_all_data

    def text_to_tfidf_Features(self,text:str):
        features_of_text=self.vectorizer.transform([text])
        features_matrix=features_of_text.T.todense()
        features_vector= np.asarray(features_matrix).reshape(-1)
        if len(features_vector)==0:
            raise Exception("No Data to be processed")
        return features_vector
        
 
    def list_of_text_to_tfidf_Features(self,list_of_text:list):
        feature_list=[]
        for text in list_of_text:
            feature_list.append(self.text_to_tfidf_Features(text))
        return feature_list
    
    def new_column_for_Answers_tfidf_features(self,df_all_data):
        df_all_data['Answers_tfitf_Features']=df_all_data.Answer.apply(self.text_to_tfidf_Features)
        return df_all_data
    
    def new_column_for_ReferenceAnswers_tfidf_features(self,df_all_data):  
        df_all_data['list_of_ReferenceAnswers_tfitf_Features']=df_all_data.list_of_answers.apply(self.list_of_text_to_tfidf_Features)
        return df_all_data   
    
    
    def distance(self,array1,array2):
            return np.linalg.norm(array1 - array2)

    
    def find_the_highest_similarity_answer(self,list_of_array,array1):
        highest_similarity=np.inf
        for array2 in list_of_array:
            new_similarity=self.distance(array1,array2)
            if new_similarity<highest_similarity:
                highest_similarity=new_similarity
                highest_array=array2
        return highest_array
            
    def add_new_column_contain_highest_similarity(self,df_all_data):
            list_of_highest=[]
            list_answers=list(df_all_data.list_of_ReferenceAnswers_tfitf_Features)
            for i, array1 in enumerate(df_all_data.Answers_tfitf_Features):
                the_highest=self.find_the_highest_similarity_answer(list_answers[i], array1)
                list_of_highest.append(the_highest)
            df_all_data['highest_answers']=list(list_of_highest)
            return df_all_data 
        
    def concatenate_all_features(self,df_all_data):
        list_of_concatenated_arrays=[]
        for i in range(len(df_all_data)):
            concatenated_arrays=np.concatenate((df_all_data.Answers_tfitf_Features[i],df_all_data.highest_answers[i]), axis = 0)
            list_of_concatenated_arrays.append(concatenated_arrays)
        df_all_data['all_features']=list_of_concatenated_arrays
        return df_all_data
    
    
    def costruct_tfidf_features_and_store_in_dataframe(self):
        df_all_data=self.convert_xml_to_dataframe()
        self.construct_features(df_all_data.ReferenceAnswers)
        df_all_data=self.add_label_column_to_dataframe(df_all_data)
        df_all_data=self.add_list_of_answers_as_new_columns(df_all_data)
        #tfidf(list(df_all_data.ReferenceAnswers))
        print("\nAdd new_columns\n")
        df_all_data=self.new_column_for_Answers_tfidf_features(df_all_data)
        print("Convert answers to tfidf features\n")
        df_all_data=self.new_column_for_ReferenceAnswers_tfidf_features(df_all_data)
        print("convert reference Answers to tfidf features\n")
        df_all_data=self.add_new_column_contain_highest_similarity(df_all_data)
        df_all_data=self.concatenate_all_features(df_all_data)
        return df_all_data
    
    def train_and_test(self) :
        
        df_all_data=self.costruct_tfidf_features_and_store_in_dataframe()
        train_features, test_features, train_labels, test_labels = train_test_split(list(df_all_data.all_features), list(df_all_data.Label),test_size=0.2)
        classifier = RandomForestClassifier()
        classifier.fit(train_features, train_labels)
        predictions=classifier.predict(test_features)

        score=f1_score(test_labels, predictions,  average='micro')
        print("\nRandom Forest Classifier F1 Score=",score)
        
        from sklearn import metrics
        print()
        print(metrics.classification_report(test_labels, predictions))
        return score

  