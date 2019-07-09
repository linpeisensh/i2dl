import numpy as np

class Transforms(object):
    """
    The min and max values for features are stored during training and
    would be used to scale data during validation/test. Since all members are static
    we are using static variables as well
    """
    min_data = None
    max_data = None
    min_label = None
    max_label = None


    @staticmethod
    def apply_data_transforms(test_dataset_X=None, mode='train'):
        """

        :param test_dataset_X: Input data to be processed
        :param mode: flag to check if mode is training or validation
        :return: test_X: the processed data as a numpy array
        """
        if test_dataset_X is None:
            return test_dataset_X
        test_numeric_X = Transforms.get_only_numeric_attributes(test_dataset_X)
        test_data_min_max_scaled = Transforms.min_max_scalar(unscaled_data=test_numeric_X, mode=mode, invoker='data')
        test_X = Transforms.convert_dataframe_to_numpy(test_data_min_max_scaled)
        # Return min and max values only when user has not provided them
        return test_X


    @staticmethod
    def apply_labels_transforms(test_dataset_y=None, mode='train'):
        """
        :param test_dataset_y: The labels to be formatted
        :param mode: flag to check if mode is training or validation
        :return: y_min_max_scaled: Processed data value
        """

        if test_dataset_y is None:
            return test_dataset_y
        # Since the y values are numeric by default we skip it
        y_min_max_scaled = Transforms.min_max_scalar(unscaled_data=test_dataset_y,  mode=mode, invoker='label')
        y_min_max_scaled = Transforms.convert_dataframe_to_numpy(y_min_max_scaled)
        return y_min_max_scaled


    @staticmethod
    def get_only_numeric_attributes(data=None, verbose=False):
        """
        :param data: dataframe with numeric and non numeric values
        :param verbose:  if True prints extra information
        :return: numeric_data: dataframe containing only numeric attributes
        """
        if data is None:
            return data
        # Look at the numerical values and then fill the missing values with 0
        # This will save us from SettingWithCopyWarning

        numeric_data = data.select_dtypes([np.number]).copy(deep=True)
        numeric_data.fillna(0, inplace=True)
        if verbose:
            print('Shape of the processed data with numerical features:', numeric_data.shape)
        return numeric_data

    @staticmethod
    def min_max_scalar(unscaled_data=None, mode='train', invoker=None):
        """

        :param unscaled_data: dataframe which is unscaled
        :param mode: flag to check if mode is training or validation
        :param invoker: data/label indicating the invoking function
        :return: normalized_df : min-max scaled
        """

        if unscaled_data is None:
            return unscaled_data

        # Use the mode is training, update the minimum and maximum
        # values based on the training data whereas for validation case
        # use these precomputed values

        if invoker == 'data':
            if mode == 'train':
                Transforms.min_data = unscaled_data.min()
                Transforms.max_data = unscaled_data.max()
            return (unscaled_data - Transforms.min_data) / (Transforms.max_data - Transforms.min_data)
        elif invoker == 'label':
            if mode == 'train':
                Transforms.min_label = unscaled_data.min()
                Transforms.max_label = unscaled_data.max()
            return (unscaled_data - Transforms.min_label) / (Transforms.max_label - Transforms.min_label)

        
        
    @staticmethod
    def convert_dataframe_to_numpy(dataframe):
        """
        :param dataframe: pandas dataframe
        :return: a numpy array representing the dataframe
        """
        return np.array(dataframe)

    @staticmethod
    def prepare_dictionary(x_train=None, x_val=None, y_train=None, y_val=None):
        """
        :param x_train: X_train data
        :param x_val: X_val data
        :param y_train: y_train data
        :param y_val: y_val data
        :return:
        """

        test_data = {
            'X_train': x_train,
            'y_train': y_train,
            'X_val': x_val,
            'y_val': y_val
        }
        return test_data

    @staticmethod
    def convert_to_binary_label(input_vector):
        """
        :param input_vector: a vector of real numbers
        :return: vector of 0,1 depending upon if the value is greater than mean value of the vector
        """
        mean_value = np.mean(input_vector)
        label_vector = np.array(input_vector > mean_value)
        return label_vector.astype(int)
