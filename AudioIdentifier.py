class AudioIdentifier:
    
    
    def __init__(self, prediction_model_path=None):
        self.vgg_model = self.load_vgg_model()
        self.prediction_model = self.load_prediction_model(prediction_model_path)
        self.labels_key = {'0' : 'air_conditioner', '1' : 'car_horn', '2' : 'children_playing', 
                          '3' : 'dog_bark', '4' : 'drilling', '5' : 'engine_idling', '6' : 'gun_shot', 
                          '7' : 'jackhammer', '8' : 'siren', '9' : 'street_music'}
        self.predicted_class = None
        self.predicted_label = None
        
    
    def load_vgg_model(self):
        vgg = vgg16.VGG16(include_top=False, weights='imagenet', 
                                     input_shape=(64, 64, 3))
        output = vgg.layers[-1].output
        output = keras.layers.Flatten()(output)
        model = Model(vgg.input, output)
        model.trainable = False
        return model
    
    
    def load_prediction_model(self, model_path):
        pred_model = keras.models.load_model(model_path)
        return pred_model
    
        
    def get_sound_data(self, path, sr=22050):
        data, fsr = sf.read(path)
        data_22k = librosa.resample(data.T, fsr, sr)
        if len(data_22k.shape) > 1:
            data_22k = np.average(data_22k, axis=0)
            
        return data_22k, sr
    
    
    def windows(self, data, window_size):
        start = 0
        while start < len(data):
            yield int(start), int(start + window_size)
            start += (window_size / 2) 
            
            
    def extract_base_features(self, sound_data, bands=64, frames=64):
    
        window_size = 512 * (frames - 1)  
        log_specgrams_full = []
        log_specgrams_hp = []
        
        start, end = list(self.windows(sound_data, window_size))[0]
        
        if(len(sound_data[start:end]) == window_size):
            signal = sound_data[start:end]

            melspec_full = librosa.feature.melspectrogram(signal, n_mels = bands)
            logspec_full = librosa.amplitude_to_db(melspec_full)
            logspec_full = logspec_full.T.flatten()[:, np.newaxis].T

            y_harmonic, y_percussive = librosa.effects.hpss(signal)
            melspec_harmonic = librosa.feature.melspectrogram(y_harmonic, n_mels = bands)
            melspec_percussive = librosa.feature.melspectrogram(y_percussive, n_mels = bands)
            logspec_harmonic = librosa.amplitude_to_db(melspec_harmonic)
            logspec_percussive = librosa.amplitude_to_db(melspec_percussive)
            logspec_harmonic = logspec_harmonic.T.flatten()[:, np.newaxis].T
            logspec_percussive = logspec_percussive.T.flatten()[:, np.newaxis].T
            logspec_hp = np.average([logspec_harmonic, logspec_percussive], axis=0)

            log_specgrams_full.append(logspec_full)
            log_specgrams_hp.append(logspec_hp)

        log_specgrams_full = np.asarray(log_specgrams_full).reshape(len(log_specgrams_full), bands ,frames, 1)
        log_specgrams_hp = np.asarray(log_specgrams_hp).reshape(len(log_specgrams_hp), bands ,frames, 1)
        features = np.concatenate((log_specgrams_full, 
                                   log_specgrams_hp, 
                                   np.zeros(np.shape(log_specgrams_full))), 
                                  axis=3)

        for i in range(len(features)):
            features[i, :, :, 2] = librosa.feature.delta(features[i, :, :, 0])

        return np.array(features)
    
    
    def extract_transfer_learn_features(self, base_feature_data):
        
        base_feature_data = np.expand_dims(base_feature_data, axis=0)
        base_feature_data = preprocess_input(base_feature_data)
        model = self.vgg_model
        tl_features = model.predict(base_feature_data)
        tl_features = np.reshape(tl_features, tl_features.shape[1])
        return tl_features
    
    
    def feature_engineering(self, audio_data):
        base_features = self.extract_base_features(sound_data=audio_data)
        final_feature_map = self.extract_transfer_learn_features(base_features[0])
        return final_feature_map
    
    
    def prediction(self, feature_map):
        model = self.prediction_model
        feature_map = feature_map.reshape(1, -1)
        pred_class = model.predict_classes(feature_map, verbose=0)
        return pred_class[0]
        
    
    def prediction_pipeline(self, audio_file_path, return_class_label=True):
        
        audio_data, sr = self.get_sound_data(audio_file_path)
        feature_map = self.feature_engineering(audio_data)
        prediction_class = self.prediction(feature_map)
        self.predicted_class = prediction_class
        self.predicted_label = self.labels_key[str(self.predicted_class)]  
        if return_class_label:
            return self.predicted_label
        else:
            return self.predicted_class