# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Capstone Project: Analysis of Song Lyrics and Audio Features for Genre Classification

## Initial Aims

##### Can song lyrics improve a model for genre classification?

- Add lyrics to the FMA dataset and genre tags to the Kaggle dataset using Genius and Deezer APIs respectively.


- Train a model using audio features from the FMA dataset.


- Perform NLP on song lyrics from the Kaggle dataset and train a model with the resulting features.


## Project Description and Findings
The validity of song genre categorisation is constantly up for debate, not least because song classification is, inherently, subjective, with different people often holding conflicting opinions as to which genre a certain band, album or track should fall within. Furthermore, the large catalogue of sub-genres that exist within each category only serve to blur the lines even further. The aim of this project was to investigate whether Machine Learning algorithms can make any sense of these subjective tags and provide a model for classifying a songs genre. How much of a song’s classifying character is contained within the structure of its sound and how much is down to the content of its lyrics?

The project looked into two datasets: [The Free Music Archive](http://freemusicarchive.org/) (FMA) dataset and a Kaggle dataset of song lyrics entitled: [Every song you have heard(almost)!](https://www.kaggle.com/artimous/every-song-you-have-heard-almost/home). Intuitively, Genre seems to be a product of the melodic features of a song; the shape, structure and mood of a track are integral in categorising it as one thing or another. The distinction in song lyrics on the other hand is maybe less clear. Although selecting a Hip-Hop lyric from a group of Country lyrics would often be relatively easy, trying to distinguish Rock from Folk could prove far more challenging. It was these more subtle differences in the lyrics, which could potentially go unnoticed simply by inspection, that were the main focus of this project. The hope was that they could be identified and amplified using Machine Learning.

The metric of success was the overall accuracy of prediction. No one genre is more important than another and so the absolute accuracy of each model is an acceptable method for gauging performance. The degree to which song lyrics are useful when it comes to predicting genre was determined by comparing a model trained solely using lyrics to a model trained solely using audio features. The final step was to train a model using both audio features and lyrics to verify the effectiveness (or not) of the lyrics to improve classification.

The findings were disappointing, despite extensive wrangling and feature engineering on the lyrics from the large Kaggle dataset, no model performed better than the baseline accuracy. Most attempts to train a model resulted in a model that would default to predicting the majority class for all observations. Randomly under sampling the training set to produce balanced classes lead to models that did, at least, make predictions, however the accuracy scores of these models on the hold out test set were very low, with a high training score suggesting severe overfitting. Sentiment analysis of the lyrics produced similar results, no model produced an accuracy greater than the baseline.

The audio features, as expected, performed far better as predictors of genre. A number of models were tested, with two (a Support Vector Machine and a Multilayer Perceptron neural network) considerably out performing all others. Once tuned the Support Vector Machine produced accuracy scores on the test set that were 100% better than the baseline. It is this Support Vector Machine that the web application uses to make predictions.

A closer look at the song lyrics revealed that there was very little difference in their content for each genre. Even with extensive stop word removal, the most frequently occurring words in each genre category were almost entirely the same. It seems that there is too much noise in the data for the model to predict effectively, going forward a closer look will have to be taken at some of the examples that were predicted correctly and some of those that were misclassified to try and determine what information in the lyrics, if any, is distinct for each genre.

## Project Notebooks
### [Analysis of Audio Features - Free Music Archive Dataset](./audio_feature_analysis.ipynb)

An exploration of the Free Music Archive dataset, with modeling on the extracted audio features.

### [Analysis of Song Lyrics - Kaggle Dataset](./lyric_analysis.ipynb)

Genre tags are added to the dataset of song lyrics using Deezer API. A selection of NLP techniques are then used on the lyrics in an attempt to produce a working model.

### [Web Scraping](./web_scraping.ipynb)

This notebook contains the functions used for web scraping, originally the hope was to obtain lyrics for the Free Music Archive data set but as it turned out, lyrics could only be found for 2,000 of the 160,000 tracks. Attempts to fetch lyrics were made to a number of different platforms using API requests, requests/Beautiful Soup and Selenium.

### [Flask Web Application](http://ahoward.pythonanywhere.com/)

The application categorises songs using the model trained on the audio features. First, it extracts the audio features from the uploaded audio file using a Librosa script adapted from the one used by the Free Music Archive team; it then uses these features to produce an overall genre prediction, along with the probabilities of the song belonging to each of the 10 categories.

## Next Steps

- Train a model using the small subset of the FMA dataset for which lyrics could be obtained, using both lyrics and audio features.


- (IN DEVELOPMENT) Write a script that extracts audio features from an mp3 file using LibROSA and develop a Flask app for genre classification.
