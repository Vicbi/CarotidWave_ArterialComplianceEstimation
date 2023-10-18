import pandas as pd
import numpy as np
from numpy.core.umath_tests import inner1d
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import stats

import pickle
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from keras.optimizers import SGD
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.callbacks import History
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error


def select_regression_model(X_train, X_test, y_train, model_selection, selected_batch_size, verbose):
    """
    Select and apply a regression model based on the specified criteria.

    Parameters:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Testing target.
        model_selection (str): Selected model ('LR1', 'LR2', 'ANN1', 'ANN2', 'ANN3', 'ANN4', 'ANN5').
        selected_batch_size (int): Batch size for ANN models.
        verbose (int): Verbosity mode for the ANN models.

    Returns:
        model: Trained regression model.
        y_pred: Predicted target values.
        y_test: True target values.
    """
    if model_selection in {'LR1', 'LR2'}:
        model = LinearRegression()
        y_pred = model.fit(X_train, y_train).predict(X_test)
    
    if model_selection.startswith('ANN'):
        selected_epochs = {
            'ANN1': 118, 'ANN2': 380, 'ANN3': 187, 'ANN4': 212, 'ANN5': 101
        }[model_selection]

        model, y_pred = artificial_neural_network(selected_batch_size, selected_epochs, X_train, X_test, y_train, verbose)
    
    y_pred = np.ravel(y_pred)
       
    return model, y_pred


def find_optimal_no_epochs(batch_size, data, X_train, X_val, y_train, y_val, prediction_variable, verbose):
    """
    Find the optimal number of epochs for training a neural network.

    Parameters:
        batch_size (int): Batch size for training.
        data (pandas.DataFrame): Input data.
        X_train (numpy.ndarray): Training features.
        X_val (numpy.ndarray): Validation features.
        y_train (numpy.ndarray): Training target values.
        y_val (numpy.ndarray): Validation target values.
        prediction_variable (str): Name of the prediction variable.
        verbose (int): Verbosity mode.
        norm_mode (str): Normalization mode.

    Returns:
        int: Optimal number of epochs.
    """
    numcolsX = X_train.shape[1]

    model = Sequential()
    # Adding the input layer and the first hidden layer
    model.add(Dense(32, activation='relu', input_dim=numcolsX))
    # Adding the output layer
    model.add(Dense(units=1))
    # Compiling the ANN
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True)

    model_history = model.fit(X_train, y_train,
                             batch_size = batch_size,
                             epochs = 1000,
                             verbose = verbose,
                             validation_data = (X_val, y_val),
                             callbacks = [early_stopping])

    a = np.max(data[prediction_variable]) - np.min(data[prediction_variable])
    train_loss = model_history.history['loss']
    validation_loss = model_history.history['val_loss']
    train_loss_scaled = [i * (a * a) for i in train_loss]
    validation_loss_scaled = [i * (a * a) for i in validation_loss]

    # Plot the loss function
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    ax.plot(np.sqrt(train_loss_scaled), '--', linewidth=4, color="#111111", label='Training loss')
    ax.plot(np.sqrt(validation_loss_scaled), linewidth=4, color="#111111", label='Validation loss')
    ax.set_xlabel(r'Epoch', fontsize=20)
    ax.set_ylabel(r'Loss', fontsize=20)
    ax.legend()
    ax.tick_params(labelsize=20)
    optimal_no_epochs = len(validation_loss_scaled)
    print('Epochs:', optimal_no_epochs)
    filename = 'loss_epochs_batch_size_{}.tiff'.format(batch_size)
    fig.savefig(filename, dpi=300, bbox_inches='tight')

    return optimal_no_epochs

def artificial_neural_network(selected_batch_size, selected_epochs, X_train, X_test, y_train, verbose):
    """
    Train and evaluate an Artificial Neural Network (ANN) model.

    Parameters:
        selected_batch_size (int): Batch size for training.
        manual_epochs (int): Number of training epochs.
        X_train (numpy.ndarray): Training features.
        X_test (numpy.ndarray): Test features.
        y_train (numpy.ndarray): Training target values.
        verbose (int): Verbosity mode.

    Returns:
        Tuple: Trained model and predicted values.
    """
    numcolsX = X_train.shape[1]

    model = Sequential()
    # Adding the input layer and the first hidden layer
    model.add(Dense(32, activation='relu', input_dim=numcolsX))
    # Adding the output layer
    model.add(Dense(units=1))
    # Compiling the ANN
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the ANN to the Training set
    model.fit(X_train, y_train, batch_size=selected_batch_size, epochs=selected_epochs, verbose=verbose)
    y_pred = model.predict(X_test)

    return model, y_pred


def create_features_dataframe(data,verbose,prediction_variable):
    car_waves = data.loc[:,'a2':'a101'].values
    carSBP_values = np.max(car_waves,axis=1)
    carDBP_values = np.min(car_waves,axis=1)
    carMAP_values = np.mean(car_waves,axis=1)
    carPP_values = carSBP_values-carDBP_values
    
    # Create an empty list to store values
    dicrotic_notch_pressure_values = []
    dicrotic_notch_time_values = []
    dPdtmax_values = []
    t_at_dPdtmax_values = []
    systolic_area_values = []
    diastolic_area_values = []
    upstroke_systolic_area_values = []

    for index, row in data.iterrows():
        pressure_waveform = data.loc[index,'a2':'a101'].values
        heart_rate = data.loc[index,'HR']

        max_dPdt_value, time_max_dPdt_value = find_dPdtmax(pressure_waveform,heart_rate,verbose)
        dPdtmax_values.append(max_dPdt_value)
        t_at_dPdtmax_values.append(time_max_dPdt_value)

        pressure_value, time_value, timing_index = find_dicrotic_notch(pressure_waveform,heart_rate,verbose)
        dicrotic_notch_pressure_values.append(pressure_value)
        dicrotic_notch_time_values.append(time_value)

        systolic_area, diastolic_area, upstroke_systolic_area = estimate_areas(pressure_waveform, timing_index)
        systolic_area_values.append(systolic_area)
        diastolic_area_values.append(diastolic_area)
        upstroke_systolic_area_values.append(upstroke_systolic_area)

    # return car_waves,carSBP_values,carDBP_values,carMAP_values,dicrotic_notch_pressure_values,dicrotic_notch_time_values, dPdtmax_values, t_at_dPdtmax_values, systolic_area_values,diastolic_area_values,upstroke_systolic_area_values

    # Create the dataframe including all features
    dataset = {'carSBP': carSBP_values,
               'carDBP': carDBP_values,
               'carMAP': carMAP_values,
               'carPP': carPP_values,
               'dicrotic_notch_pressure': dicrotic_notch_pressure_values,
               'dicrotic_notch_time': dicrotic_notch_time_values,
               'upstroke_systolic_area': upstroke_systolic_area_values,
               'total_systolic_area': systolic_area_values,
               'diastolic_area': diastolic_area_values,
               'dPdtmax': dPdtmax_values,
               't_at_dPdtmax': t_at_dPdtmax_values,
               'HR': data.loc[:,'HR'].values,
               'age': data.loc[:,'age'].values,
               'gender': data.loc[:,'gender'].values,
               'height': data.loc[:,'height'].values,
               'weight': data.loc[:,'weight'].values,
               'CO': data.loc[:,'CO'].values,
               'C_PPM': data.loc[:,'C_PPM'].values,
              }

    dataset = pd.DataFrame(dataset)
    return dataset

def estimate_areas(signal, timing_index):
    """
    Estimate areas under the curve of a signal.

    Args:
        signal (numpy.ndarray): The input signal.
        timing_index (int): Timing index.

    Returns:
        float: Area before the timing index.
        float: Area after the timing index.
        float: Area from the beginning to the timing index of the maximum value.
    """
    # Estimate area before timing index
    area_before = np.trapz(signal[:timing_index])

    # Estimate area after timing index
    area_after = np.trapz(signal[timing_index:])

    # Find the index of the maximum value
    max_index = np.argmax(signal)

    # Estimate area from the beginning to the timing index of the maximum value
    area_until_max = np.trapz(signal[:max_index + 1])

    return area_before, area_after, area_until_max

def find_dPdtmax(signal,heart_rate,verbose):
    """
    Find the maximum value of the first derivative of a signal and its timing index.

    Args:
        signal (numpy.ndarray): The input signal.
        heart_rate (fload): The heart rate value.
        
    Returns:
        float: Maximum value of the first derivative.
        float: Time value the maximum value of the first derivative.
    """
    # Calculate the first derivative of the signal
    dPdt = np.gradient(signal)

    # Find the index of the maximum value of the first derivative
    max_dPdt_index = np.argmax(dPdt)

    # Get the maximum value of the first derivative
    max_dPdt_value = dPdt[max_dPdt_index]

    # Create the time vector
    time = np.linspace(0,60/heart_rate,len(signal))
    
    # Get the time value of the maximum value of the first derivative
    time_max_dPdt_value = time[max_dPdt_index]
    
    if verbose:
        print(f"Maximum value of the first derivative: {max_dPdt_value}")
        print(f"Time of the maximum value of the first derivative: {time_max_dPdt_value}")
    
    return max_dPdt_value, time_max_dPdt_value


def find_dicrotic_notch(signal,heart_rate,verbose):
    """
    Find the pressure value and timing values of the dicrotic notch.

    Args:
        signal (numpy.ndarray): The pressure waveform.
        heart_rate (fload): The heart rate value.
        
    Returns:
        float: Pressure value of the dicrotic notch.
        float: Time value the dicrotic notch.
        int: Index of the dicrotic notch in the signal.
    """
    # Find the second derivative of the signal
    d2_signal = np.gradient(np.gradient(signal))

    # Find indices where the second derivative changes from positive to negative
    notch_indices = np.where((d2_signal[:-1] > 0) & (d2_signal[1:] <= 0))[0]

    # Find the index of the maximum point in the signal
    max_index = np.argmax(signal)

    # Find the index of the dicrotic notch after the maximum value
    dicrotic_notch_index = next((i for i in notch_indices if i > max_index), None)

    if dicrotic_notch_index is not None:
        # Get the pressure value of the dicrotic notch
        dicrotic_notch_pressure = signal[dicrotic_notch_index]
    else:
        dicrotic_notch_pressure = len(signal)/3

    # Create the time vector
    time = np.linspace(0,60/heart_rate,len(signal))
    
    # Get the time value of the dicrotic notch
    dicrotic_notch_time = time[dicrotic_notch_index]
    
    if verbose:
        print(f"Dicrotic notch pressure value: {pressure_value}")
        print(f"Time of the dicrotic notch: {dicrotic_notch_time}")

    return dicrotic_notch_pressure, dicrotic_notch_time, dicrotic_notch_index


def prepare_dataset(model_selection,prediction_variable,verbose,noise_mode,snr_dB):
    # Read data outside the function if feasible
    data_waves = pd.read_csv('asklepios_carotid_waves.csv')
    
    if noise_mode:
        for index, row in data_waves.iterrows():
            data_waves.loc[:, 'a2':'a101']
            data_waves.loc[index, 'a2':'a101'] = add_white_gaussian_noise(data_waves.loc[index, 'a2':'a101'], snr_dB)
        
    data_features = create_features_dataframe(data_waves,verbose,prediction_variable)

    features1 = ['carSBP','carDBP','carPP','carMAP','dicrotic_notch_pressure','dicrotic_notch_time',
                 'upstroke_systolic_area','total_systolic_area','diastolic_area','dPdtmax','t_at_dPdtmax',
                 'HR','age','gender','height','weight',prediction_variable]

    features2 = np.concatenate((['CO'],features1))

    features5 = ['carPP','carSBP','diastolic_area','total_systolic_area','weight',prediction_variable]

    if model_selection == 'LR1':
        dataset = data_features[features1]
        regressor = 'LR'

    elif model_selection == 'LR2':
        dataset = data_features[features2]
        regressor = 'LR'

    elif model_selection == 'ANN1':
        dataset = data_features[features1]
        regressor = 'ANN'

    elif model_selection == 'ANN2':
        dataset = data_features[features2]
        regressor = 'ANN'

    elif model_selection == 'ANN3':
        dataset = data_waves.loc[:, 'a2':'a101'].join(data_waves[['HR', 'age', 'gender', 'height', 'weight',prediction_variable]])
        regressor = 'ANN'

    elif model_selection == 'ANN4':
        dataset = data_waves.loc[:, 'a2':'a101'].join(data_waves[['HR',prediction_variable]])
        regressor = 'ANN'

    elif model_selection == 'ANN5':
        dataset = data_features[features5]
        regressor = 'ANN'

    else:
        raise ValueError("Invalid model selection: {}".format(model_selection))

    print('{} model was selected.'.format(model_selection))
    print('The dataset size is:', dataset.shape)

    return dataset, regressor


def add_white_gaussian_noise(signal, snr_dB):
    """
    Add white Gaussian noise to a signal.

    Args:
        signal (numpy.ndarray): The input signal.
        snr_dB (float): Signal-to-noise ratio in decibels.

    Returns:
        numpy.ndarray: Noisy signal.
    """
    # Calculate signal power
    signal_power = np.mean(np.abs(signal)**2)

    # Calculate noise power based on SNR
    snr_linear = 10**(snr_dB / 10)
    noise_power = signal_power / snr_linear

    # Generate white Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))

    # Add noise to the signal
    noisy_signal = signal + noise

    return noisy_signal


def scale_data(dataset):
    """
    Scale the input dataset using Min-Max scaling.

    Parameters:
        dataset (pd.DataFrame): The dataset to be scaled.

    Returns:
        pd.DataFrame: Scaled dataset.
    """

    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_array = min_max_scaler.fit_transform(dataset.values)
    scaled_dataset = pd.DataFrame(scaled_array, columns=dataset.columns)
     
    return scaled_dataset


def rescale_values(values, prediction_variable, dataset):
    """
    Rescale values based on the specified prediction_variable and dataset.

    Parameters:
        values (numpy.ndarray): Array to be rescaled.
        prediction_variable (str): The variable being predicted.
        dataset (pandas.DataFrame): The dataset containing the prediction variable.

    Returns:
        rescaled_values (numpy.ndarray): Rescaled values.
    """
    max_prediction_variable = np.max(dataset[prediction_variable])
    min_prediction_variable = np.min(dataset[prediction_variable])
    
    rescaled_values = min_prediction_variable + (max_prediction_variable - min_prediction_variable) * values
    
    return rescaled_values
    
    
def calculate_metrics_for_each_fold(current_fold,y_test,y_pred):
    from sklearn import metrics
    import numpy as np
    print('Fold no. (',current_fold+1,')')
    print('Mean Absolute Error:', (metrics.mean_absolute_error(y_test, y_pred)), '[unit]')  
    print('Mean Squared Error:', ((metrics.mean_squared_error(y_test, y_pred))),'[unit]')  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)),'[unit]')
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print('Normalized Root Mean Squared Error:', 100*rmse/(np.max(y_test)-np.min(y_test)),'%\n')

def split_features_target(dataset):
    """
    Split the input dataframe into features (X) and target (y).

    Parameters:
        dataset (pd.DataFrame): The input dataframe.

    Returns:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
    """
    X = np.array(dataset.iloc[:, :-1])  # All columns except the last one
    y = np.array(dataset.iloc[:, -1])   # The last column

    return X, y

def print_results(y_test, y_pred, variable_unit):
    """
    Print various regression metrics and statistics.

    Parameters:
        y_test (array-like): True values.
        y_pred (array-like): Predicted values.

    Returns:
        None
    """
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    nrmse = 100 * rmse / (np.max(y_test) - np.min(y_test))

    print('Mean Absolute Error:', np.round(mae, 2), variable_unit)
    print('Mean Squared Error:', np.round(mse, 2), variable_unit)
    print('Root Mean Squared Error:', np.round(rmse, 2), variable_unit)
    print('Normalized Root Mean Squared Error:', np.round(nrmse, 2), '%\n')

    slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_pred)
    print('Correlation:', round(r_value, 2))
    print('Slope:', round(slope, 2))
    print('Intercept:', round(intercept, 2), variable_unit)
    print('r_value:', round(r_value, 2))
    print('p_value:', round(p_value, 4))

    print('Distribution of the reference data:', round(np.mean(y_test), 1), '±', round(np.std(y_test), 1), variable_unit)
    print('Distribution of the predicted data:', round(np.mean(y_pred), 1), '±', round(np.std(y_pred), 1), variable_unit)
          

def permutation_importances(model, X, y, metric):
    baseline = metric(y, model.predict(X))
    imp = []
    for col in range(X.shape[1]):
        save = X[:, col].copy()
        X[:, col] = np.random.permutation(X[:, col])
        m = metric(y, model.predict(X))
        X[:, col] = save
        imp.append(baseline - m)
    return np.array(imp)