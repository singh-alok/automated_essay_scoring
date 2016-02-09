__author__ = 'alok'
import pandas as pd,math, numpy as np
from core.algo.scorer import Manager
from scan import settings
from random import shuffle
def divide_test_and_train(text,scores,threshold):
    text_used =[]
    scores_used =[]
    for itr, score in enumerate(scores):
        if score > threshold: continue
        text_used.append(text[itr])
        scores_used.append(score)

    unique_scores = list(set(scores_used))
    train_text, train_scores, test_text, test_scores = [],[],[],[]
    # dividing in test and train(0.7 and 0.3) for every score in data
    for us in unique_scores:
        index = [iter for iter, su in enumerate(scores_used) if su ==us]
        temp_text = [text_used[i] for i in index]
        temp_scores = [scores_used[i] for i in index]
        train_samples = [1]*(int(0.7*len(temp_scores))) + [0]*(len(temp_scores) - int(0.7*len(temp_scores)))
        shuffle(train_samples)
        for itr, ts in enumerate(train_samples):
            if ts == 1:
                train_text.append(temp_text[itr])
                train_scores.append(temp_scores[itr])
            else:
                test_text.append(temp_text[itr])
                test_scores.append(temp_scores[itr])
    return train_text, train_scores, test_text, test_scores

if __name__  ==  "__main__":
    input_file = "/Users/alok/Documents/Data/Text_Mining/AES/The kings school_que_a_sec_abc_g123.csv"
    tab1= pd.read_csv(input_file,header=0)
    essay = list(tab1['text'])
    for itr, e in enumerate(essay):
        try:
            if math.isnan(e): essay[itr] = ""
        except:
            continue
    score = list(tab1['score_swathi'])
    threshold = 10
    mean_error = []
    std_error = []
    for i in range(20):
        print i,"th iteration"
        train_essay, train_score, test_essay,test_score = divide_test_and_train(essay,score, threshold)
        print "samples for training is ", len(train_essay)
        print "sample for testing is ", len(test_essay)
        id ='a'
        manager = Manager(essay)
        model_loc = manager.create_model(id, text=train_essay, scores=train_score, MODEL_PATH = settings.MODEL_PATH)
        print 'model is created and stored in ', model_loc
        predicted_score=[manager.score_essay(text = te, MODEL_PATH = model_loc) for itera, te in enumerate(test_essay)]
        diff_score = [abs(sc - predicted_score[i][0])/10.0 for i,sc in enumerate(test_score)]
        mean_error.append(np.mean(diff_score))
        std_error.append(np.std(diff_score))
    print 'average of error is ', np.mean(mean_error), mean_error
    print 'standard deviation of error is ',np.mean(std_error), std_error