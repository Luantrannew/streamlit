# Import Library
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
import warnings
warnings.filterwarnings("ignore")

path = os.getcwd() + "\movie_preprocessing5.csv"

movie = pd.read_csv(path,encoding='utf-8')
#movie=movie1[:5000]
print(movie.head())