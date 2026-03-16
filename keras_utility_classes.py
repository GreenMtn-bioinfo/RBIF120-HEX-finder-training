import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from get_normalization_params_5 import means_path, sdevs_path, mins_path, maxes_path
from split_data_4 import load_json, write_json_pretty, all_profiles_path, IDs_file_path, split_json_path
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
import os
import sys
import shutil

#### This script/module script contains classes and wrapper functions that are useful for training & testing of Keras models


## Modified code written by others

# Defines data generator object for use in keras training (circumvent loading of full data set to RAM at once)
# This code was modififed slightly from a helpful tutorial by Afshine Amidi and Shervine Amidi
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, 
                 all_data, 
                 labels, 
                 norm_params, 
                 norm_mode, # z, minmax, z_minmax, or minmax_z
                 sample_weights = None,
                 batch_size=64, 
                 dim=(64,28,77),
                 n_classes=3, 
                 shuffle=True, 
                 workers=6, 
                 use_multiprocessing=True, 
                 max_queue_size=10):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.all_data = all_data
        self.sample_weights = sample_weights
        self.norm_mode = norm_mode
        if 'z' in norm_mode:
            self.means = norm_params['means'].transpose()
            self.sdevs = norm_params['sdevs'].transpose()
        if 'minmax' in norm_mode:
            self.mins = norm_params['mins'].transpose()
            self.maxes = norm_params['maxes'].transpose()
        super().__init__(workers=workers, use_multiprocessing=use_multiprocessing, max_queue_size=max_queue_size)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        if self.sample_weights is not None:
            X, y, sample_weights = self.__data_generation(list_IDs_temp)
            return X, y, sample_weights
        else:
            X, y = self.__data_generation(list_IDs_temp)
            return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __z_norm(self, array: np.ndarray) -> np.ndarray:
        return (array - self.means) / self.sdevs
    
    def __min_max_norm(self, array: np.ndarray, bounds: tuple = (-1, 1)) -> np.ndarray:
        return (bounds[1] - bounds[0]) * (array - self.mins) / (self.maxes - self.mins) + bounds[0]
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        
        # Generate data
        X = self.all_data[list_IDs_temp, :, :].transpose([0, 2, 1]) # TODO: This may not work with the previously fed ID lists
        # X = X.reshape((X.shape[0], *X.shape[2:]))
        match self.norm_mode:
            case 'z':
                X = self.__z_norm(X)
            case 'minmax':
                X = self.__min_max_norm(X)
            case 'z_minmax':
                X = self.__min_max_norm(self.__z_norm(X))
            case 'minmax_z':
                X = self.__z_norm(self.__min_max_norm(X))
        y = self.labels[list_IDs_temp]
        
        if self.sample_weights is not None:
            weights = self.sample_weights[list_IDs_temp]
            return X, keras.utils.to_categorical(y, num_classes=self.n_classes), weights
        else:
            return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

# Enables calculation of the per-class AUC using existing keras metrics/classes
# Written by Artem Mavrin and shared on Stack Overflow:
# https://stackoverflow.com/questions/63580476/how-to-compute-auc-for-one-class-in-multi-class-classification-in-keras
class MulticlassAUC(tf.keras.metrics.AUC):
    """AUC for a single class in a multiclass problem.
    
    Parameters
    ----------
    pos_label : int
        Label of the positive class (the one whose AUC is being computed).

    from_logits : bool, optional (default: False)
        If True, assume predictions are not standardized to be between 0 and 1.
        In this case, predictions will be squeezed into probabilities using the
        softmax function.

    sparse : bool, optional (default: True)
        If True, ground truth labels should be encoded as integer indices in the
        range [0, n_classes-1]. Otherwise, ground truth labels should be one-hot
        encoded indicator vectors (with a 1 in the true label position and 0
        elsewhere).

    curve : {'ROC', 'PR'}, optional (default: 'ROC')
        The type of curve to compute the AUC for. Use 'ROC' for Receiver
        Operating Characteristic or 'PR' for Precision-Recall.

    **kwargs : keyword arguments
        Keyword arguments for tf.keras.metrics.AUC.__init__().
        For example, `num_thresholds` or `name`.
    """

    def __init__(self, pos_label, from_logits=False, sparse=False, curve='ROC', **kwargs):
        # Pass the curve argument and any other kwargs to the parent class
        super().__init__(curve=curve, **kwargs)
        
        self.pos_label = pos_label
        self.from_logits = from_logits
        self.sparse = sparse

    def update_state(self, y_true, y_pred, **kwargs):
        """Accumulates confusion matrix statistics.

        Parameters
        ----------
        y_true : tf.Tensor
            The ground truth values. Either an integer tensor of shape
            (n_examples,) (if sparse=True) or a one-hot tensor of shape
            (n_examples, n_classes) (if sparse=False).

        y_pred : tf.Tensor
            The predicted values, a tensor of shape (n_examples, n_classes).

        **kwargs : keyword arguments
            Extra keyword arguments for tf.keras.metrics.AUC.update_state
            (e.g., sample_weight).
        """
        if self.sparse:
            y_true = tf.math.equal(y_true, self.pos_label)
            y_true = tf.squeeze(y_true)
        else:
            y_true = y_true[..., self.pos_label]

        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        y_pred = y_pred[..., self.pos_label]

        super().update_state(y_true, y_pred, **kwargs)

# This class and the function that follows were written by Google Gemini and modified as needed
class MultiThresholdFPRForClass(tf.keras.metrics.Metric):
    """
    Calculates the False Positive Rate (FPR) for a specific class ID
    across a list of decision thresholds in a multi-class setting.
    """
    def __init__(self, thresholds, class_id, name='multi_threshold_fpr_for_class', **kwargs):
        super(MultiThresholdFPRForClass, self).__init__(name=name, **kwargs)
        self.thresholds = sorted(thresholds)
        self.class_id = tf.constant(class_id, dtype=tf.int32)
        
        # Add weights to accumulate false positives and true negatives for each threshold
        self.false_positives = self.add_weight(
            name='false_positives',
            shape=(len(self.thresholds),),
            initializer='zeros',
            dtype=tf.float32
        )
        self.true_negatives = self.add_weight(
            name='true_negatives',
            shape=(len(self.thresholds),),
            initializer='zeros',
            dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Determine the number of classes from the predictions
        num_classes = tf.shape(y_pred)[-1]
        
        # Cast true labels and predictions for calculation
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, self.dtype)
        
        # Handle one-hot encoded or sparse integer true labels
        if y_true.shape.ndims > 1 and y_true.shape[-1] > 1:
            # Assume one-hot encoded and get integer labels
            y_true = tf.argmax(y_true, axis=-1, output_type=tf.int32)

        # Convert the multi-class problem to a binary one for the target class
        # `is_negative` is True for all classes except our `class_id`
        is_negative = tf.not_equal(y_true, self.class_id)
        
        # Get the probability for the target class ID
        pred_for_target_class = y_pred[..., self.class_id]
        
        # Expand dimensions to enable broadcasting over thresholds
        pred_broadcast = tf.expand_dims(pred_for_target_class, axis=-1)
        is_negative_broadcast = tf.expand_dims(is_negative, axis=-1)
        thresholds_tensor = tf.constant(self.thresholds, dtype=self.dtype)

        # Predictions are considered positive if the probability for `class_id` exceeds the threshold
        predictions_are_positive = tf.greater_equal(pred_broadcast, thresholds_tensor)

        # A false positive occurs when the true class is not `class_id`, but the model predicts it is
        false_positives_batch = tf.cast(tf.logical_and(is_negative_broadcast, predictions_are_positive), self.dtype)
        
        # A true negative occurs when the true class is not `class_id`, and the model does not predict it is
        true_negatives_batch = tf.cast(tf.logical_and(is_negative_broadcast, tf.logical_not(predictions_are_positive)), self.dtype)

        # Accumulate the counts for each threshold over batches
        self.false_positives.assign_add(tf.reduce_sum(false_positives_batch, axis=0))
        self.true_negatives.assign_add(tf.reduce_sum(true_negatives_batch, axis=0))

    def result(self):
        # Calculate FPR for each threshold
        fpr_values = self.false_positives / (self.false_positives + self.true_negatives + tf.keras.backend.epsilon())
        return fpr_values

    def reset_states(self):
        # Reset variables at the beginning of each epoch
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.true_negatives.assign(tf.zeros_like(self.true_negatives))
    
    def get_config(self):
        config = super(MultiThresholdFPRForClass, self).get_config()
        config.update({'thresholds': self.thresholds, 'class_id': self.class_id.numpy()})
        return config



## Functions/wrappers written by me:

# Checks if a training run directory exists and asks the user to decide
def check_make_dir(dir_path: str):
    if os.path.exists(dir_path):
        response = input(f'check_make_dir(): {dir_path} already exists: "overwrite", "cancel", or anything/ENTER to continue with a past run?')
        if response == "overwrite":
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
            print(f'check_make_dir(): removed {dir_path} and created a new one.')
        elif response == "cancel":
            sys.exit()
    else:
        os.mkdir(dir_path)
        print(f'check_make_dir(): made {dir_path}.')

# Plot results of a training run for chosen metric
def plot_training_metric(metrics: list, model_name: str, training_history):
    plt.figure()
    leg = []
    for metric in metrics:
        train_results = training_history.history[metric]
        validation_results = training_history.history["val_" + metric]
        plt.plot(np.arange(1, len(train_results) + 1), train_results)
        leg.append(metric)
        plt.plot(np.arange(1, len(validation_results) + 1), validation_results)
        leg.append(f"val_{metric}")
        plt.title(model_name)
        plt.ylabel("Metric", fontsize="large")
    plt.xlabel("Epoch", fontsize="large")
    plt.legend(leg, loc="best")
    plt.show()

# Breaks out the per-class results from the Keras F1 score or similar (if it was called with average=None during training)
def plot_multi_F1(model_name: str, metric_name: str, training_history):
    plt.figure()
    tr_F1 = np.array(training_history.history[metric_name])
    val_F1 = np.array(training_history.history[f'val_{metric_name}'])
    n_classes = tr_F1.shape[1]
    leg = []
    for cl in range(n_classes):
        plt.plot(np.arange(1, len(tr_F1[:,cl]) + 1), tr_F1[:,cl])
        leg.append(f'{metric_name}_{cl}')
        plt.plot(np.arange(1, len(val_F1[:,cl]) + 1), val_F1[:,cl])
        leg.append(f"val_{metric_name}_{cl}")
    plt.title(model_name)
    plt.ylabel("Metric", fontsize="large")
    plt.xlabel("Epoch", fontsize="large")
    plt.legend(leg, loc="best")
    plt.show()



## Classes written by me

# Used for running training in 
class TrainingRun():
    'Manages cross-validation runs using a Keras model and keras.utils.Sequence data generator'
    def __init__(self, 
                 class_encoding,
                 class_weights = None,
                 norm_mode = 'z_minmax',
                 shuffle = True,
                 workers = 6,
                 use_multiprocessing = False,
                 max_queue_size = 10):
        
        'Initialization'
        self.metadata_IDs_col = 0 # This is where unique identifier for the samples are stored in self.metadata
        self.metadata_Class_col = 1 # This is where the class identifiers for the samples are stored in self.metadata
        self.intron_status_col = 3
        self.__load_data_from_files()
        self.training_indices = np.argwhere(np.isin(self.metadata[:, self.metadata_IDs_col], self.partition['train'], assume_unique=True)).flatten()
        self.class_encoding = class_encoding
        self.seq_types_updated = np.array([f'{str(row[self.metadata_Class_col])}_{str(row[self.intron_status_col])}' if str(row[self.intron_status_col]) != '' else str(row[self.metadata_Class_col]) for row in self.metadata[:,] ])
        self.class_labels = np.array([self.class_encoding[str(label)] for label in self.seq_types_updated])
        self.class_weights = class_weights
        if self.class_weights is not None:
            self.sample_weights = np.array([self.class_weights[str(label)] for label in self.seq_types_updated])
        else:
            self.sample_weights = None
        # self.batch_size = batch_size
        self.input_shape = self.norm_params['means'].transpose().shape
        self.n_classes = len(set(self.class_labels))
        self.generator_params = {'labels': self.class_labels,
                                 'sample_weights': self.sample_weights,
                                 'all_data' : self.data,
                                 'norm_params': self.norm_params,
                                 'norm_mode' : norm_mode,
                                 'dim': self.input_shape,
                                 'n_classes': self.n_classes,
                                 'shuffle': shuffle,
                                 'workers' : workers, 
                                 'use_multiprocessing' : use_multiprocessing,
                                 'max_queue_size' : max_queue_size }
        self.thresholds = [0.001, 0.01, 0.025] + np.arange(0.05, 1.0, 0.05).tolist() + [0.975, 0.98, 0.985, 0.99, 0.999] # Used for things like getting the whole PR curve
    
    def __load_data_from_files(self):
        'Loads the training/validation IDs, metadata, and data.'
        print(f'TrainingRun(): Loading neccesary .json and .npy data files...')
        self.partition = load_json(split_json_path)
        self.data = np.load(all_profiles_path, mmap_mode="r")
        self.metadata = np.load(IDs_file_path, mmap_mode="r")
        self.norm_params = {'means' : np.load(means_path, allow_pickle=False),
                            'sdevs' : np.load(sdevs_path, allow_pickle=False),
                            'mins' : np.load(mins_path, allow_pickle=False),
                            'maxes' : np.load(maxes_path, allow_pickle=False)}
    
    def __compile_model(self, model, batch_size, max_epochs, start_lr=0.0002, finish_lr=0.001, decay=False):
        'Compiles the model with the desired metrics, loss, and optimizer.'
        
        # Learning rate decay scheduler
        if decay:
            lr_schedule = ExponentialDecay(initial_learning_rate = start_lr,
                                           decay_steps= max_epochs * 111969 // (batch_size // 32), # Decay over a large number of steps for a large dataset
                                           decay_rate= start_lr/finish_lr)
        
        # Pass the schedule to the optimizer
        optimizer = Adam(learning_rate=(start_lr if not decay else lr_schedule))
        
        model.compile(loss='categorical_crossentropy', 
                            optimizer=optimizer, 
                            metrics=['categorical_accuracy', 
                                      keras.metrics.KLDivergence(name="kl_divergence"),
                                      MulticlassAUC(name='AUC_ROC_Control', pos_label=self.class_encoding['control_exons'], sparse=False), 
                                      MulticlassAUC(name='AUC_ROC_Intron-Exon', pos_label=self.class_encoding['intron-exon'], sparse=False), 
                                      MulticlassAUC(name='AUC_ROC_Exon-Intron', pos_label=self.class_encoding['exon-intron'], sparse=False),
                                      MulticlassAUC(name='AUC_PR_Control', curve="PR", pos_label=self.class_encoding['control_exons'], sparse=False), 
                                      MulticlassAUC(name='AUC_PR_Intron-Exon', curve="PR", pos_label=self.class_encoding['intron-exon'], sparse=False), 
                                      MulticlassAUC(name='AUC_PR_Exon-Intron', curve="PR", pos_label=self.class_encoding['exon-intron'], sparse=False),
                                      keras.metrics.F1Score(name='F1_Score_argmax', average=None),
                                      keras.metrics.F1Score(name='F1_Score_0.9', threshold=0.9, average=None),
                                      keras.metrics.F1Score(name='F1_Score_0.8', threshold=0.8, average=None),
                                      keras.metrics.F1Score(name='F1_Score_0.7', threshold=0.7, average=None),
                                      keras.metrics.F1Score(name='F1_Score_0.6', threshold=0.6, average=None),
                                      keras.metrics.F1Score(name='F1_Score_0.5', threshold=0.5, average=None),
                                      keras.metrics.F1Score(name='F1_Score_0.4', threshold=0.4, average=None),
                                      keras.metrics.F1Score(name='F1_Score_0.3', threshold=0.3, average=None),
                                      keras.metrics.F1Score(name='F1_Score_0.2', threshold=0.2, average=None),
                                      keras.metrics.F1Score(name='F1_Score_0.1', threshold=0.1, average=None),
                                      keras.metrics.RecallAtPrecision(0.9, name='Recall_at_90P_Control', class_id=self.class_encoding['control_exons']),
                                      keras.metrics.RecallAtPrecision(0.9, name='Recall_at_90P_Intron-Exon', class_id=self.class_encoding['intron-exon']),
                                      keras.metrics.RecallAtPrecision(0.9, name='Recall_at_90P_Exon-Intron', class_id=self.class_encoding['exon-intron']),
                                      keras.metrics.Precision(name='Precision_Control', class_id=self.class_encoding['control_exons'], thresholds=self.thresholds),
                                      keras.metrics.Precision(name='Precision_Intron-Exon', class_id=self.class_encoding['intron-exon'], thresholds=self.thresholds),
                                      keras.metrics.Precision(name='Precision_Exon-Intron', class_id=self.class_encoding['exon-intron'], thresholds=self.thresholds),
                                      keras.metrics.Recall(name='Recall_Control', class_id=self.class_encoding['control_exons'], thresholds=self.thresholds),
                                      keras.metrics.Recall(name='Recall_Intron-Exon', class_id=self.class_encoding['intron-exon'], thresholds=self.thresholds),
                                      keras.metrics.Recall(name='Recall_Exon-Intron', class_id=self.class_encoding['exon-intron'], thresholds=self.thresholds),
                                      MultiThresholdFPRForClass(name='FPR_Control', class_id=self.class_encoding['control_exons'], thresholds=self.thresholds),
                                      MultiThresholdFPRForClass(name='FPR_Intron-Exon', class_id=self.class_encoding['intron-exon'], thresholds=self.thresholds),
                                      MultiThresholdFPRForClass(name='FPR_Exon-Intron', class_id=self.class_encoding['exon-intron'], thresholds=self.thresholds) ])
    
    def __make_CV_folds(self, N_folds: int, down_sample_proportion: float = None):
        'Prepares indices for an N-fold crossvalidation from training set, class representation from training sample is maintained in each fold.'
        RNG = np.random.default_rng()
        folds = [np.empty((0,), dtype=np.int64) for i in range(N_folds)]
        stratified_classes = [key for key in self.class_encoding]
        for cls in stratified_classes:
            class_indices = np.argwhere(np.isin(self.metadata[:, self.metadata_IDs_col], self.partition['train'], assume_unique=True) & (self.seq_types_updated == cls)).flatten()
            if down_sample_proportion is not None:
                class_indices = RNG.choice(class_indices, size=int(np.floor(len(class_indices) * down_sample_proportion)), replace=False)
            class_indices = RNG.permutation(class_indices)
            slice_length = len(class_indices) // N_folds
            slices = [slice(i * slice_length, (i + 1) * slice_length) if i != (N_folds - 1) else slice(i * slice_length, len(class_indices) - 1) for i in range(N_folds)]
            for i, slc in enumerate(slices):
                folds[i] = np.concatenate([folds[i], class_indices[slc]])
        return [RNG.permutation(fold) for fold in folds] # Shuffle the fold just in case downstream usage does not!
    
    def get_CV_folds(self, N_folds: int, save_dir: str = None, down_sample_proportion: float = None):
        'Returns or saves the N-fold cross validation split (useful for managing long training times).'
        splits = self.__make_CV_folds(N_folds, down_sample_proportion)
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            else:
                print(f'get_CV_folds(): {save_dir} already exists, please delete or rename to avoid loss of existing CV splits.')
                return splits
            for i, split in enumerate(splits):
                np.save(f'{save_dir}fold_{i+1}.npy', arr=split, allow_pickle=False)
        else:
            return splits
    
    def load_CV_folds(self, dir: str):
        files = os.listdir(dir)
        splits = []
        for i in range(len(files)):
            splits.append(np.load(f'{dir}fold_{i+1}.npy', allow_pickle=False))
        return splits
    
    def __make_held_out(self, validation_proportion: float, down_sample_proportion: float = None): 
        'In case only one validation split will be made from the training data, we still want it to be random each time!'
        RNG = np.random.default_rng()
        stratified_classes = [key for key in self.class_encoding]
        training = []
        validation = []
        for cls in stratified_classes:
            class_indices = np.argwhere(np.isin(self.metadata[:, self.metadata_IDs_col], self.partition['train'], assume_unique=True) & (self.seq_types_updated == cls)).flatten()
            if down_sample_proportion is not None:
                class_indices = RNG.choice(class_indices, size=int(np.floor(len(class_indices) * down_sample_proportion)), replace=False)
            val_indices = RNG.choice(class_indices, size=int(len(class_indices) * validation_proportion), replace=False).tolist()
            validation = validation + val_indices
            training = training + list(set(class_indices).difference(set(val_indices)))
        return {'train_train' : RNG.permutation(training).tolist(), 'train_validation' : RNG.permutation(validation).tolist()}
    
    # Callback wrapper: Used during training to stop after no metric improvement over a certain magnitude in over a certain number of epochs
    def __make_stopper_callback(self, stop_metric: str, mode: str, delta: float = 0.001, patience: int = 0):
        return keras.callbacks.EarlyStopping(monitor=stop_metric,
                                             min_delta=delta,
                                             patience=patience,
                                             verbose=1,
                                             mode=mode,
                                             restore_best_weights=False,
                                             start_from_epoch=0)

    # Callback wrapper: Saves weights for the model we really want based on a validation metric
    def __make_checkpoint_callback(self, save_path: str, best_metric: str, mode: str, only_save_best: bool):
        return keras.callbacks.ModelCheckpoint(save_path, 
                                               monitor=best_metric,
                                               verbose=1,
                                               mode=mode,
                                               save_best_only=only_save_best)
    
    def __summary_plots(self, model_name:str, model, history, which_plots: list = [False, True, False, True, True]):
        if which_plots[0] and 'kl_divergence' in model.metrics_names:
            plot_training_metric(['kl_divergence'], f'{model_name}: Kullback–Leibler divergence', history)
        if which_plots[1] and 'loss' in model.metrics_names:
            plot_training_metric(['loss'], f'{model_name}: loss (categorical crossentropy)', history)
        if which_plots[2] and 'AUC_ROC_Control' in model.metrics_names:
            plot_training_metric(['AUC_ROC_Control', 'AUC_ROC_Intron-Exon', 'AUC_ROC_Exon-Intron'], f'{model_name}: ROC AUC', history)
        if which_plots[3] and 'AUC_PR_Control' in model.metrics_names:
            plot_training_metric(['AUC_PR_Control', 'AUC_PR_Intron-Exon', 'AUC_PR_Exon-Intron'], f'{model_name}: PR AUC', history)
        if which_plots[4] and 'F1_Score_argmax' in model.metrics_names:
            plot_multi_F1(f'{model_name}: F1 Score (Max Probabillity)', 'F1_Score_argmax', history)
    
    def execute_training_run(self, # TODO: means, sdevs, mins, maxes are currently calculated on entire training set, so validation stats could be slightly inflated by training-validation leakage (not the same as train-test leakage!)
                             saves_base_dir: str,
                             batch_size: int, 
                             max_epochs: int, 
                             base_name: str = None,
                             model_creator = None, 
                             model_args: dict = None,
                             model_name: str = None,
                             stop_save_metric: str = 'val_loss',
                             patience: int = 1,
                             min_delta: float = 0.001,
                             run_type: str = 'held_out', 
                             val_proportion: float = None,
                             down_sample: float = None,
                             N_folds: int = None,
                             lr_decay: bool = False,
                             lr: float = 0.0002,
                             final_lr: float = None,
                             plot_results: bool = False,
                             folds_dir: str = None,
                             tuning_builder = None):
        'Runs the training and saves results along the way.'
        self.generator_params['batch_size'] = batch_size
        
        # Save some details of the class/call for later reference/context if context is lost
        params = locals()
        del params['self']
        del params['model_creator']
        del params['tuning_builder']
        params['class_weights'] = self.class_weights
        params['class_encoding'] = self.class_encoding
        params['norm_mode'] = self.generator_params['norm_mode']
        write_json_pretty(params, f'{saves_base_dir}training_call.json', verbose=True)
        
        if (run_type == 'held_out' and val_proportion is not None) or run_type == 'final_eval':
            
            model = model_creator(input_shape=self.input_shape, n_classes=self.n_classes, **model_args)
            self.__compile_model(model, batch_size=batch_size, max_epochs=max_epochs, start_lr=lr, decay=lr_decay, finish_lr=final_lr)
            
            if run_type != 'final_eval':
                print('execute_training_run(): splitting the training set into training and held-out/validation.')
                split = self.__make_held_out(val_proportion, down_sample_proportion=down_sample)
            else:
                print('execute_training_run(): final training and evaluation will soon commence!')
                self.testing_indices = np.argwhere(np.isin(self.metadata[:, self.metadata_IDs_col], self.partition['test'], assume_unique=True)).flatten()
                RNG = np.random.default_rng()
                self.testing_indices = RNG.permutation(self.testing_indices).tolist()
                self.training_indices = RNG.permutation(self.training_indices).tolist()
            training_generator = DataGenerator(list_IDs=(split['train_train'] if run_type != 'final_eval' else self.training_indices), **self.generator_params)
            validation_generator = DataGenerator(list_IDs=(split['train_validation'] if run_type != 'final_eval' else self.testing_indices), **self.generator_params)
            
            stopper = self.__make_stopper_callback(stop_save_metric, delta = min_delta, patience = patience, mode='auto')
            save_best = self.__make_checkpoint_callback(f'{saves_base_dir}{base_name}_best.keras', best_metric=stop_save_metric, mode='auto', only_save_best=True)
            log_file = f'{saves_base_dir}{base_name}_log.csv'
            log_to_csv = keras.callbacks.CSVLogger(log_file, append=False)
            
            print('execute_training_run(): training will now commence!')
            history = model.fit(x=training_generator, 
                                validation_data=validation_generator,
                                epochs=max_epochs, 
                                batch_size=batch_size, 
                                callbacks=[save_best, stopper, log_to_csv])
            
            os.rename(log_file, log_file.replace('.csv', '_complete.csv'))
            
            if plot_results:
                self.__summary_plots(model_name, model=model, history=history)
            
        elif run_type == 'cross_validation' and N_folds is not None:
            
            if folds_dir is None: # Depending on whether a pre-baked/saved cross val split has been provided or not
                print(f'execute_training_run(): splitting the training set into {N_folds} folds for cross-validation...')
                folds = self.__make_CV_folds(N_folds, down_sample_proportion=down_sample)
                start = 0
            else:
                print(f'execute_training_run(): loading cross-validation splits from files in {folds_dir}...')
                folds = self.load_CV_folds(folds_dir)
                completed_logs = [int(file.split('_')[3]) for file in os.listdir(saves_base_dir) if '_complete.csv' in file] # TODO: make regex
                if not completed_logs:
                    print(f'execute_training_run(): it appears that no folds have been completed, starting from fold 1.')
                    start = 0
                else:
                    max_completed = max(completed_logs)
                    print(f'execute_training_run(): it appears that {max_completed}/{len(folds)} have already been completed. Resuming on fold {max_completed + 1}.')
                    start = max_completed
            
            for i in range(start, len(folds)):
                
                model = model_creator(input_shape=self.input_shape, n_classes=self.n_classes, **model_args)
                self.__compile_model(model, batch_size=batch_size, max_epochs=max_epochs, start_lr=lr, decay=lr_decay, finish_lr=final_lr)
                
                this_fold = folds[i].tolist()
                other_folds = np.concatenate([fold for j, fold in enumerate(folds) if j != i]).tolist()
                training_generator = DataGenerator(list_IDs=other_folds, **self.generator_params)
                validation_generator = DataGenerator(list_IDs=this_fold, **self.generator_params)
                
                stopper = self.__make_stopper_callback(stop_save_metric, delta = min_delta, patience = patience, mode='auto')
                save_best = self.__make_checkpoint_callback(f'{saves_base_dir}{base_name}_best_fold_{i+1}.keras', best_metric=stop_save_metric, mode='auto', only_save_best=True)
                this_log = f'{saves_base_dir}{base_name}_log_fold_{i+1}.csv'
                log_to_csv = keras.callbacks.CSVLogger(this_log, append=False)
                
                print(f'execute_training_run(): fold {i+1}/{N_folds} training will now commence!')
                history = model.fit(x=training_generator, 
                                    validation_data=validation_generator,
                                    epochs=max_epochs, 
                                    batch_size=batch_size, 
                                    callbacks=[save_best, stopper, log_to_csv])
                
                os.rename(this_log, this_log.replace('.csv', '_complete.csv'))
                
                if plot_results:
                    self.__summary_plots(f'{model_name} (CV fold {i+1})', model=model, history=history)

        else:
            print('execute_training_run(): invalid run type selected and/or data splitting parameter not provided!')