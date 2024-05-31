# Music Genre Classification Using Decision Tree Algorithms

Recommendation systems are becoming increasingly popular within the streaming world, and part of their success is predicated on algorithms that can accurately determine which genres of music that a user is favorable to. Music has nominal and categorical characteristics that can allow one to deduce the genre if the correct training methods are applied. The goal of this research was to apply various decision tree algorithms to a large dataset of songs to determine which algorithms have the best accuracy and which features are most important for predicting music genres.

## Key Findings

The best-performing decision tree classifier was the gradient boost, with an accuracy of 76 percent, a recall of 76 percent, and a precision score of 77 percent. The Decision Stump Classifier was the worst performing classifier and not very accurate, with a score of 20 percent. The recall also had a score of 20 percent but the precision only had a score of four percent.

All of the decision tree classifiers had popularity as the feature with the highest importance, followed by danceability, acousticness, and speechiness. Mode had the lowest importance amongst all features.

## Libraries

- Libraries: Pandas, NumPy, SciKit Learn, Matplotlib
- Dataset: Kaggle Music Genre
- Language: Python

## File Structure

- `classification_notebook/`: Contains the Jupyter notebook and CSV file for classification.
- `documentation/`: Contains the research paper document in .pdf and .txt format.
