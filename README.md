# Face Recognition
Face Recognition with SVM classifier using PCA, ICA, NMF, LDA reduced face vectors

## Repository Structure
- The folders ```PCA, ICA, NMF, LDA and DATASET``` consists of all the images and classification report for ech algorithm respectively.
- The files ```pca.py | ica.py | nmf.py | lda.py``` consists of algorithm implementation for each algorithm respectively.
- The document ```Report.docx``` present in the root of the source code contains all the textual document of the project.
- The document ```todo-mom.docx``` present in the root of the source code contains all the todos of each individual and minutes of meeting of the group.
- The ```requirements.txt``` file contains the project dependencies.

## Dependencies
- Python3
- Run ```pip install -r requirements.txt``` to install required Python libraries

## Steps to run each algorithm individually
- Clone the repository
- Run ```pip install -r requirements.txt``` to install required Python libraries
- For PCA, run the command ```python pca.py```
- For ICA, run the command ```python ica.py```
- For NMF, run the command ```python nmf.py```
- For LDA, run the command ```python lda.py```


# Experimental Result
### Dataset used
- Labelled faces in the wild [http://vis-www.cs.umass.edu/lfw/]
- Faces which has has more than 100 samples were used.

### PCA (Principal Component Analysis)
Eigenfaces | Prediction | Classification Report
--- | --- | ---
![Eigenface generated](https://raw.githubusercontent.com/tulsyanp/tcd-ai-group-project/master/PCA/figure_readme/faces.png) | ![Prediction](https://raw.githubusercontent.com/tulsyanp/tcd-ai-group-project/master/PCA/figure_readme/prediction.png) | ![Classification report](https://raw.githubusercontent.com/tulsyanp/tcd-ai-group-project/master/PCA/figure_readme/pca_result.png)


### LDA (Linear Discriminant Analysis)
FisherFaces | Prediction | Classification Report
--- | --- | ---
![Eigenface generated](https://raw.githubusercontent.com/tulsyanp/tcd-ai-group-project/master/LDA/figure_readme/faces.png) | ![Prediction](https://raw.githubusercontent.com/tulsyanp/tcd-ai-group-project/master/LDA/figure_readme/prediction.png) | ![Classification report](https://raw.githubusercontent.com/tulsyanp/tcd-ai-group-project/master/LDA/figure_readme/lda_result.png)


### ICA (Independent Component Analysis)
Eigenfaces | Prediction | Classification Report
--- | --- | ---
![Eigenface generated](https://raw.githubusercontent.com/tulsyanp/tcd-ai-group-project/master/ICA/figure_readme/faces.png) | ![Prediction](https://raw.githubusercontent.com/tulsyanp/tcd-ai-group-project/master/ICA/figure_readme/prediction.png) | ![Classification report](https://raw.githubusercontent.com/tulsyanp/tcd-ai-group-project/master/ICA/figure_readme/ica_result.png)


### NMF (Non-negative Matrix Factorization)
Eigenfaces | Prediction | Classification Report
--- | --- | ---
![Eigenface generated](https://raw.githubusercontent.com/tulsyanp/tcd-ai-group-project/master/NMF/figure_readme/faces.png) | ![Prediction](https://raw.githubusercontent.com/tulsyanp/tcd-ai-group-project/master/NMF/figure_readme/prediction.png) | ![Classification report](https://raw.githubusercontent.com/tulsyanp/tcd-ai-group-project/master/NMF/figure_readme/nmf_result.png)


# Group Members
1. Prateek Tulsyan - 19303677
2. Mrinal Jhamb - 19301913
3. Shubham Dhupar - 19304374
4. Rushikesh Joshi - 19300976