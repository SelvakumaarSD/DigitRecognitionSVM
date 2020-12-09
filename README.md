# DigitRecognitionSVM
SVM is used to train and classify the MNIST Dataset.
The dataset consists of the pixel values of each digit which forms the features. 
Each image contains a 28x28 pixel info and each pixel forms a feature.
Hence, the dataset has 784 feature columns along with one classifier column to identify which digit it belongs to. There are 10000 data rows (samples).
The sample is split in a 20:80 ratio of 20% making for the test (validation set). With the remaining 80% the model is trained and the optimal train size is chosen such that 
maximum accuracy score is achieved.

Libraries used:
- Data Manipulation / Pre-Processing: numpy, pandas, math, time
- ML Predictive Modelling: sklearn
- Visualization: matplotlib, seaborn

Methods:
- pandas.read_csv(): Method helps read the contents of input csv file
- pandas.to_csv(): Method helps write the results data frame to a output csv file
- pandas.describe(): Describes the data frame passed as input to this function. Shows the data type of columns, count rows of values, etc.
- sns.countplot(): helps convert categorical variable into dummy/indicator variables
- sklearn.model_selection.train_test_split(): Splits the into dataset into training and validation (or test) sets based on the ratio provided. A range of training sizes are used here to find the optimal size with highest prediction accuracy.
- Sklearn.svm.SVC(): Constructor of class SVM classifier in python sklearn library. When this constructor is called, the input data frame is converted into a feature matrix (of size n x 784, where n is the training size).
9
- fit(): Build a SVM classifier from the input training set. The input data points are applied to different kernel functions (linear, non-linear, polynomial, etc.) and the models are derived each time.
- score(): Returns the mean accuracy on the given validation set for the different models derived in previous step. Based on the accuracy score, the model with highest score is chosen for prediction
- predict(): Predict the class of input validation sample or test set
