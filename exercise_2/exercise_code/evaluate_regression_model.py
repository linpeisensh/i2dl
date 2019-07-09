import pickle
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from exercise_code.transforms import Transforms
import pandas as pd

def preprocess_y(data):
    mat_train_y = np.matrix(y_raw)
    prepro_y_test = MinMaxScaler()
    prepro_y_test.fit(mat_train_y)
    scaled_data = pd.DataFrame(prepro_y_test.transform(mat_train_y))
    y_row_vector = np.array(scaled_data)
    return y_row_vector.reshape(-1,1)

	
df_test= pd.read_csv('datasets/house_price_data.csv')
# Remove id column
df_test = df_test.drop('Id', 1)
y_raw = df_test.SalePrice
X_raw = df_test.drop('SalePrice', axis=1)
# center the SalePrice values
y_test_preprocessed = preprocess_y(y_raw)

def evaluate_regression_model(model_path):

    modeldict = pickle.load(open(model_path, 'rb'))
    fc_net = modeldict['fully_connected_net']
    if fc_net is None:
        raise ValueError('The model you have saved is of the type None. Please check')
    transforms = modeldict['transforms']
    # Apply the transformations on the input data
    transformed_X = transforms.apply_transforms_on_test_data(test_dataset_X=X_raw)
    if not transformed_X.shape[0]== X_raw.shape[0]:
        raise ValueError('Invalid Transform function. You should not remove the data elements') 	
    X_test = transformed_X.reshape(transformed_X.shape[0],-1)
    
    y_pred =fc_net.loss(X_test)
    mse_loss = np.sqrt(metrics.mean_squared_error(y_test_preprocessed, y_pred))
        
    score = 0.001/mse_loss # improve
    return score
