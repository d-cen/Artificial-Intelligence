"""
Stop, this file should not be open!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




























Seriously, stop Scrolling, you have no idea what you gonna see!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!





























Really? You think you are ready?????????????????????





























Ok! fine, here you go :)
"""
import numpy as np

def super_secret_function(test_dir, func):

    (test_data, test_label) = np.load(test_dir)

    score = 0
    for idx in range(len(test_data)):
        if int(func(test_data[idx])) == int(test_label[idx]):
            score += 1

    return score / len(test_data)

