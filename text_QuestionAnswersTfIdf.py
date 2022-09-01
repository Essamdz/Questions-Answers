import unittest
import pandas as pd
from QuestionAnswersTfIdf import QuestionAnswers 
import numpy
import numpy as np

class TestQuestionAnswers(unittest.TestCase):  
    file="grade_data.xml"
    df_for_testing = pd.DataFrame( {
              "Answer": ["An equal force always balancing","an object will stay at rest or at constant"],
              "ReferenceAnswers":["\n1:force balancing \n2:an object wil \n","\n1:hi\n2:there\n"],
              "Answers_tfitf_Features":[np.array([1,3,4]),np.array([7,8,9])],
              "highest_answers":[np.array([3,3,3]),np.array([9,9,9])]
            })

    def test_convert_xml_to_dataframe(self):
        question_answers=QuestionAnswers(TestQuestionAnswers.file)
        self.assertRaises(Exception, question_answers.convert_xml_to_dataframe())
        self.assertIs(type(question_answers.convert_xml_to_dataframe()), type(pd.DataFrame())) 
    
    def test_construct_features(self):
        question_answers=QuestionAnswers(TestQuestionAnswers.file)
        corpus = [
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the first document?',
        ]
        self.assertEqual(question_answers.construct_features(corpus).shape[0],len(corpus))
                
        
    def test_extract_labels_from_Annotation_column(self):
        question_answers=QuestionAnswers(TestQuestionAnswers.file)
        s="correct(1)|correct_but_incomplete(0)|contradictory(0)|incorrect(0)"
        self.assertEqual(question_answers.extract_labels_from_Annotation_column(s),1)
        s="correct(0)|correct_but_incomplete(0)|contradictory(1)|incorrect(0)"
        self.assertEqual(question_answers.extract_labels_from_Annotation_column(s),3)


    def test_convert_ReferenceAnswers_as_list_of_answers(self):
        question_answers=QuestionAnswers(TestQuestionAnswers.file)
        st="\n1:hi\n2:there\n"
        self.assertEqual(question_answers.convert_ReferenceAnswers_as_list_of_answers(st),["hi","there"])
        self.assertIsNotNone(question_answers.convert_ReferenceAnswers_as_list_of_answers(st))

    def test_add_list_of_answers_as_new_columns(self):
        question_answers=QuestionAnswers(TestQuestionAnswers.file)
        df_temp=question_answers.add_list_of_answers_as_new_columns(TestQuestionAnswers.df_for_testing)
        self.assertIn("list_of_answers", df_temp.columns)
        
    def test_cosine_similarity(self):
        question_answers=QuestionAnswers(TestQuestionAnswers.file)
        a=np.array([1,2,3])
        b=np.array([1,2,3])
        c=np.array([1,1,1])
        self.assertEqual( question_answers.distance(a,b),0) 
        self.assertNotEqual( question_answers.distance(a,c),0) 

    def test_find_the_highest_similarity_answer(self):
        question_answers=QuestionAnswers(TestQuestionAnswers.file)
        a=[np.array([1,2,2]),np.array([5,5,5]),np.array([10,20,20])]
        b=np.array([1,2,3])
        self.assertEqual(list(question_answers.find_the_highest_similarity_answer(a,b)),[1,2,2])
        
    def test_concatenate_all_features(self):
        question_answers=QuestionAnswers(TestQuestionAnswers.file)
        df=question_answers.concatenate_all_features(TestQuestionAnswers.df_for_testing)
        #df.all_features
        self.assertIn("all_features",df.columns)
        self.assertEqual(list(list(df.all_features)[0]) ,[1,3,4,3,3,3])
        
    def test_output(self):
        question_answers=QuestionAnswers(TestQuestionAnswers.file)
        self.assertTrue(question_answers.train_and_test()>.25)

if __name__ == '__main__':
    unittest.main()