from gc import collect
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk.stem
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, concatenate, multiply, average, subtract, add, maximum, minimum
from tensorflow.keras.models import Model
from scikit_learn.sklearn_svdd.svm import SVDD
import warnings

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


def train_test_split_pipeline(df_int, percent, column, f1):
    if percent == 3 or percent == 5 or percent == 7:
        df_test = np.array(df_int[df_int.fold != f1][column].to_list())
        df_fold = np.array(df_int[df_int.fold == f1][column].to_list())
        train_percent = percent / 10
        test_percent = 1 - train_percent
        df_train, discard = train_test_split(df_fold, test_size=test_percent, random_state=42)

    else:  # percent 10
        df_train = np.array(df_int[df_int.fold == f1][column].to_list())
        df_test = np.array(df_int[df_int.fold != f1][column].to_list())

    return df_train, df_test


def init_metrics():
    return {
        'precision': [],
        'recall': [],
        'f1-score': [],
        'auc_roc': [],
        'accuracy': [],
        'time': []
    }


def save_values(metrics, values):
    for key in metrics.keys():
        metrics[key].append(values[key])


def evaluation_one_class(preds_interest, preds_outliers):
    y_true = [1] * len(preds_interest) + [-1] * len(preds_outliers)
    y_pred = list(preds_interest) + list(preds_outliers)
    return classification_report(y_true, y_pred, output_dict=True)


def evaluate_model(x_train, x_test, x_outlier, model):
    one_class_classifier = model.fit(x_train)

    y_pred_interest = one_class_classifier.predict(x_test)

    y_pred_outlier = one_class_classifier.predict(x_outlier)

    score_interest = one_class_classifier.decision_function(x_test)

    score_outlier = one_class_classifier.decision_function(x_outlier)

    y_true = np.array([1] * len(x_test) + [-1] * len(x_outlier))

    fpr, tpr, _ = roc_curve(y_true, np.concatenate([score_interest, score_outlier]))

    dic = evaluation_one_class(y_pred_interest, y_pred_outlier)

    metrics = {'precision': dic['1']['precision'], 'recall': dic['1']['recall'], 'f1-score': dic['1']['f1-score'],
               'auc_roc': roc_auc_score(y_true, np.concatenate([score_interest, score_outlier])),
               'accuracy': dic['accuracy']}

    return metrics, fpr, tpr


def evaluate_models(models, representations, file_name, line_parameters, path, percent):
    for model in tqdm(models):

        lp = model + '_' + line_parameters
        fn = file_name + '_' + model.split('_')[0] + '_' + str(percent) + '.csv'
        metrics = init_metrics()

        for reps in representations:
            start = time.time()
            values, fpr, tpr = evaluate_model(reps[0], reps[1], reps[2], models[model])
            end = time.time()
            time_ = end - start
            values['time'] = time_

            save_values(metrics, values)
        write_results(metrics, fn, lp, path)


def write_results(metrics, file_name, line_parameters, path):
    if not Path(path + file_name).is_file():
        file_ = open(path + file_name, 'w')
        string = 'Parameters'

        for metric in metrics.keys():
            string += ';' + metric + '-mean;' + metric + '-std'
        string += '\n'

        file_.write(string)
        file_.close()

    file_ = open(path + file_name, 'a')
    string = line_parameters

    for metric in metrics.keys():
        string += ';' + str(np.mean(metrics[metric])) + ';' + str(np.std(metrics[metric]))

    string += '\n'
    file_.write(string)
    file_.close()


def make_density_information(cluster_list, df_train, df_test, df_outlier):
    l_x_train = []
    l_x_test = []
    l_x_outlier = []

    for cluster in cluster_list:
        kmeans = KMeans(n_clusters=cluster, random_state=0).fit(df_train)

        x_train_temp = silhouette_samples(df_train, kmeans.labels_).reshape(len(df_train), 1)
        l_x_train.append(x_train_temp)

        x_test_temp = silhouette_samples(df_test, kmeans.predict(df_test)).reshape(len(df_test), 1)
        l_x_test.append(x_test_temp)

        x_outlier_temp = silhouette_samples(df_outlier, kmeans.predict(df_outlier)).reshape(len(df_outlier), 1)
        l_x_outlier.append(x_outlier_temp)

    return np.concatenate(l_x_train, axis=1), np.concatenate(l_x_test, axis=1), np.concatenate(l_x_outlier, axis=1)


def has_numbers(input_string):
    return any(char.isdigit() for char in input_string)


class MyTokenizer:
    def __init__(self, language):
        self.wnl = WordNetLemmatizer()
        if language == 'english':
            self.STOPWORDS = nltk.corpus.stopwords.words('english')
            self.stemmer = nltk.stem.SnowballStemmer('english')

        if language == 'spanish':
            self.STOPWORDS = nltk.corpus.stopwords.words('spanish')
            self.stemmer = nltk.stem.SnowballStemmer('spanish')

        if language == 'portuguese':
            self.STOPWORDS = nltk.corpus.stopwords.words('portuguese')
            self.stemmer = nltk.stem.SnowballStemmer('portuguese')

        if language == 'multilingual':
            self.STOPWORDS = set(nltk.corpus.stopwords.words('spanish')).union(
                set(nltk.corpus.stopwords.words('portuguese'))).union(set(nltk.corpus.stopwords.words('english')))
            self.stemmer = nltk.stem.SnowballStemmer('english')

    def __call__(self, doc):
        l1 = [t for t in word_tokenize(doc)]
        l2 = []
        for token in l1:
            if token not in self.STOPWORDS and token.isnumeric() is False and len(token) > 2 and has_numbers(
                    token) is False:
                l2.append(token)
        l3 = [self.stemmer.stem(self.wnl.lemmatize(t)) for t in l2]
        return l3


def autoencoder(arq, input_length):
    encoder_inputs = Input(shape=(input_length,), name='encoder_input')

    if len(arq) == 3:
        first_dense_encoder = Dense(arq[0], activation="linear")(encoder_inputs)

        second_dense_encoder = Dense(arq[1], activation="linear")(first_dense_encoder)

        encoded = Dense(arq[2], activation="linear")(second_dense_encoder)

        first_dense_decoder = Dense(arq[1], activation="linear")(encoded)

        second_dense_decoder = Dense(arq[0], activation="linear")(first_dense_decoder)

        decoder_output = Dense(input_length, activation="linear")(second_dense_decoder)

    elif len(arq) == 2:
        first_dense_encoder = Dense(arq[0], activation="linear")(encoder_inputs)

        encoded = Dense(arq[1], activation="linear")(first_dense_encoder)

        first_dense_decoder = Dense(arq[0], activation="linear")(encoded)

        decoder_output = Dense(input_length, activation="linear")(first_dense_decoder)

    else:  # len(arq) == 1
        encoded = Dense(arq[0], activation="linear")(encoder_inputs)

        decoder_output = Dense(input_length, activation="linear")(encoded)

    encoder = Model(encoder_inputs, encoded)

    autoencoder_model = Model(encoder_inputs, decoder_output)

    autoencoder_model.compile(optimizer=tensorflow.keras.optimizers.Adam(), loss='mse')

    return autoencoder_model, encoder


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), seed=1)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, factor_multiply, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.factor_multiply = factor_multiply

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.mean_squared_error(data, reconstruction)
            )
            reconstruction_loss *= self.factor_multiply
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            return {
                "loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,
            }


def encoder_vae(arq, input_dim):
    encoder_inputs = keras.Input(shape=(input_dim,), name='encoder_input')

    if len(arq) == 3:
        first_dense = Dense(arq[0], activation="linear")(encoder_inputs)

        second_dense = Dense(arq[1], activation="linear")(first_dense)

        z_mean = layers.Dense(arq[2], name="Z_mean")(second_dense)
        z_log_var = layers.Dense(arq[2], name="Z_log_var")(second_dense)
        z = Sampling()([z_mean, z_log_var])

    elif len(arq) == 2:
        first_dense = Dense(arq[0], activation="linear")(encoder_inputs)

        z_mean = layers.Dense(arq[1], name="Z_mean")(first_dense)
        z_log_var = layers.Dense(arq[1], name="Z_log_var")(first_dense)
        z = Sampling()([z_mean, z_log_var])

    else:  # len(arq) == 1
        z_mean = layers.Dense(arq[0], name="Z_mean")(encoder_inputs)
        z_log_var = layers.Dense(arq[0], name="Z_log_var")(encoder_inputs)
        z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model([encoder_inputs], [z_mean, z_log_var, z], name="Encoder")

    return encoder


def decoder_vae(arq, output_dim):
    latent_inputs = keras.Input(shape=(arq[(len(arq) - 1)],), name='decoder_input')

    if len(arq) == 3:
        first_dense = Dense(arq[1], activation="linear")(latent_inputs)

        second_dense = Dense(arq[0], activation="linear")(first_dense)

        decoder_outputs = Dense(output_dim, activation="linear")(second_dense)

    elif len(arq) == 2:
        first_dense = Dense(arq[0], activation="linear")(latent_inputs)

        decoder_outputs = Dense(output_dim, activation="linear")(first_dense)

    else:  # len(arq) == 1
        decoder_outputs = Dense(output_dim, activation="linear")(latent_inputs)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    return decoder


def variationalautoencoder(arq, input_dim):
    encoder = encoder_vae(arq, input_dim)

    decoder = decoder_vae(arq, input_dim)

    vae = VAE(encoder, decoder, input_dim)

    vae.compile(optimizer=keras.optimizers.Adam())

    return vae, encoder, decoder


class MVAE(keras.Model):
    def __init__(self, encoder, decoder, factor_multiply_embedding, factor_multiply_density, **kwargs):
        super(MVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.factor_multiply_embedding = factor_multiply_embedding
        self.factor_multiply_density = factor_multiply_density

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder((data[0], data[1]))

            reconstruction = self.decoder(z)

            embedding_loss = tf.reduce_mean(
                keras.losses.mean_squared_error(data[0], reconstruction[0])
            )

            embedding_loss *= self.factor_multiply_embedding

            density_loss = tf.reduce_mean(
                keras.losses.mean_squared_error(data[1], reconstruction[1])
            )

            density_loss *= self.factor_multiply_density

            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = embedding_loss + density_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "tota loss": total_loss,
            "embedding loss": embedding_loss,
            "denisty loss": density_loss,
            "kl loss": kl_loss,
        }


def encoder_mvae(arq, embedding_dim, density_dim, operator):
    embedding_inputs = keras.Input(shape=(embedding_dim,), name='first_input_encoder')
    density_inputs = keras.Input(shape=(density_dim,), name='second_input_encoder')

    l1 = Dense(np.max([embedding_dim, density_dim]), activation='linear')(embedding_inputs)
    l2 = Dense(np.max([embedding_dim, density_dim]), activation='linear')(density_inputs)

    fusion = None
    if operator == 'concatenate':
        fusion = concatenate([l1, l2])
    if operator == 'multiply':
        fusion = multiply([l1, l2])
    if operator == 'average':
        fusion = average([l1, l2])
    if operator == 'subtract':
        fusion = subtract([l1, l2])
    if operator == 'add':
        fusion = add([l1, l2])
    if operator == 'max':
        fusion = maximum([l1, l2])
    if operator == 'min':
        fusion = minimum([l1, l2])

    if len(arq) == 3:
        first_dense = Dense(arq[0], activation="linear")(fusion)

        second_dense = Dense(arq[1], activation="linear")(first_dense)

        z_mean = layers.Dense(arq[2], name="Z_mean")(second_dense)
        z_log_var = layers.Dense(arq[2], name="Z_log_var")(second_dense)
        z = Sampling()([z_mean, z_log_var])

    elif len(arq) == 2:
        first_dense = Dense(arq[0], activation="linear")(fusion)

        z_mean = layers.Dense(arq[1], name="Z_mean")(first_dense)
        z_log_var = layers.Dense(arq[1], name="Z_log_var")(first_dense)
        z = Sampling()([z_mean, z_log_var])

    else:  # len(arq) == 1
        z_mean = layers.Dense(arq[0], name="Z_mean")(fusion)
        z_log_var = layers.Dense(arq[0], name="Z_log_var")(fusion)
        z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model([embedding_inputs, density_inputs], [z_mean, z_log_var, z], name="encoder")

    return encoder


def decoder_mvae(arq, embedding_dim, density_dim):
    latent_inputs = keras.Input(shape=(arq[(len(arq) - 1)],), name='input_decoder')

    if len(arq) == 3:
        first_dense = Dense(arq[1], activation="linear")(latent_inputs)

        second_dense = Dense(arq[0], activation="linear")(first_dense)

        embedding_outputs = Dense(embedding_dim, activation="linear")(second_dense)

        density_outputs = Dense(density_dim, activation="linear")(second_dense)

    elif len(arq) == 2:
        first_dense = Dense(arq[0], activation="linear")(latent_inputs)

        embedding_outputs = Dense(embedding_dim, activation="linear")(first_dense)

        density_outputs = Dense(density_dim, activation="linear")(first_dense)

    else:  # len(arq) == 1
        embedding_outputs = Dense(embedding_dim, activation="linear")(latent_inputs)

        density_outputs = Dense(density_dim, activation="linear")(latent_inputs)

    decoder = keras.Model(latent_inputs, [embedding_outputs, density_outputs], name="decoder")

    return decoder


def multimodalvae(arq, embedding_dim, density_dim, operator):
    encoder = encoder_mvae(arq, embedding_dim, density_dim, operator)

    decoder = decoder_mvae(arq, embedding_dim, density_dim)

    mvae = MVAE(encoder, decoder, embedding_dim, density_dim)

    mvae.compile(optimizer=keras.optimizers.Adam())

    return mvae, encoder, decoder


class TVAE(keras.Model):
    def __init__(self, encoder, decoder, factor_multiply_embedding, factor_multiply_density, factor_multiply_liwc,
                 **kwargs):
        super(TVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.factor_multiply_embedding = factor_multiply_embedding
        self.factor_multiply_density = factor_multiply_density
        self.factor_multiply_liwc = factor_multiply_liwc

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder((data[0], data[1], data[2]))

            reconstruction = self.decoder(z)

            embedding_loss = tf.reduce_mean(
                keras.losses.mean_squared_error(data[0], reconstruction[0])
            )

            embedding_loss *= self.factor_multiply_embedding

            density_loss = tf.reduce_mean(
                keras.losses.mean_squared_error(data[1], reconstruction[1])
            )

            density_loss *= self.factor_multiply_density

            liwc_loss = tf.reduce_mean(
                keras.losses.mean_squared_error(data[2], reconstruction[2])
            )

            liwc_loss *= self.factor_multiply_liwc

            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = embedding_loss + density_loss + liwc_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "total loss": total_loss,
            "embedding loss": embedding_loss,
            "density loss": density_loss,
            "liwc loss": liwc_loss,
            "kl loss": kl_loss
        }


def encoder_tvae(arq, embedding_dim, density_dim, liwc_dim, operator):
    embedding_inputs = keras.Input(shape=(embedding_dim,), name='first_input_encoder')
    density_inputs = keras.Input(shape=(density_dim,), name='second_input_encoder')
    liwc_inputs = keras.Input(shape=(liwc_dim,), name='third_input_encoder')

    l1 = Dense(np.max([embedding_dim, density_dim, liwc_dim]), activation='linear')(embedding_inputs)
    l2 = Dense(np.max([embedding_dim, density_dim, liwc_dim]), activation='linear')(density_inputs)
    l3 = Dense(np.max([embedding_dim, density_dim, liwc_dim]), activation='linear')(liwc_inputs)

    fusion = None
    if operator == 'concatenate':
        fusion = concatenate([l1, l2, l3])
    if operator == 'multiply':
        fusion = multiply([l1, l2, l3])
    if operator == 'average':
        fusion = average([l1, l2, l3])
    if operator == 'subtract':
        fusion = subtract([l1, l2])
        fusion = subtract([fusion, l3])
    if operator == 'add':
        fusion = add([l1, l2, l3])
    if operator == 'max':
        fusion = maximum([l1, l2])
    if operator == 'min':
        fusion = minimum([l1, l2])

    if len(arq) == 3:
        first_dense = Dense(arq[0], activation="linear")(fusion)

        second_dense = Dense(arq[1], activation="linear")(first_dense)

        z_mean = layers.Dense(arq[2], name="Z_mean")(second_dense)
        z_log_var = layers.Dense(arq[2], name="Z_log_var")(second_dense)
        z = Sampling()([z_mean, z_log_var])

    elif len(arq) == 2:
        first_dense = Dense(arq[0], activation="linear")(fusion)

        z_mean = layers.Dense(arq[1], name="Z_mean")(first_dense)
        z_log_var = layers.Dense(arq[1], name="Z_log_var")(first_dense)
        z = Sampling()([z_mean, z_log_var])

    else:  # len(arq) == 1
        z_mean = layers.Dense(arq[0], name="Z_mean")(fusion)
        z_log_var = layers.Dense(arq[0], name="Z_log_var")(fusion)
        z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model([embedding_inputs, density_inputs, liwc_inputs], [z_mean, z_log_var, z], name="encoder")

    return encoder


def decoder_tvae(arq, embedding_dim, density_dim, liwc_dim):
    latent_inputs = keras.Input(shape=(arq[(len(arq) - 1)],), name='input_decoder')

    if len(arq) == 3:
        first_dense = Dense(arq[1], activation="linear")(latent_inputs)

        second_dense = Dense(arq[0], activation="linear")(first_dense)

        embedding_outputs = Dense(embedding_dim, activation="linear")(second_dense)

        density_outputs = Dense(density_dim, activation="linear")(second_dense)

        liwc_outputs = Dense(liwc_dim, activation="linear")(second_dense)

    elif len(arq) == 2:
        first_dense = Dense(arq[0], activation="linear")(latent_inputs)

        embedding_outputs = Dense(embedding_dim, activation="linear")(first_dense)

        density_outputs = Dense(density_dim, activation="linear")(first_dense)

        liwc_outputs = Dense(liwc_dim, activation="linear")(first_dense)

    else:  # len(arq) == 1
        embedding_outputs = Dense(embedding_dim, activation="linear")(latent_inputs)

        density_outputs = Dense(density_dim, activation="linear")(latent_inputs)

        liwc_outputs = Dense(liwc_dim, activation="linear")(latent_inputs)

    decoder = keras.Model(latent_inputs, [embedding_outputs, density_outputs, liwc_outputs], name="decoder")

    return decoder


def triplevae(arq, embedding_dim, density_dim, liwc_dim, operator):
    encoder = encoder_tvae(arq, embedding_dim, density_dim, liwc_dim, operator)

    decoder = decoder_tvae(arq, embedding_dim, density_dim, liwc_dim)

    tvae = TVAE(encoder, decoder, embedding_dim, density_dim, liwc_dim)

    tvae.compile(optimizer=keras.optimizers.Adam())

    return tvae, encoder, decoder


def term_weight_type(bow_type, language_tokenizer='multilingual'):
    if bow_type == 'term-frequency':
        vectorizer = CountVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 1), min_df=1,
                                     tokenizer=MyTokenizer(language_tokenizer))
    elif bow_type == 'term-frequency-IDF':
        vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 1), min_df=1,
                                     tokenizer=MyTokenizer(language_tokenizer))
    else:  # Binary
        vectorizer = CountVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 1), min_df=1,
                                     tokenizer=MyTokenizer(language_tokenizer), binary=True)

    return vectorizer


def make_representation(prepro, df_int, df_out, folds, percent, vectorizer, cluster_list, parameter_list):
    representations = []

    for fold in folds:

        if prepro == 'BoW':
            df_train, df_test = train_test_split_pipeline(df_int, percent, 'text', fold)
            df_outlier = np.array(df_out.text.to_list())

            vectorizer.fit(df_train)

            x_train = vectorizer.transform(df_train).toarray()
            x_test = vectorizer.transform(df_test).toarray()
            x_outlier = vectorizer.transform(df_outlier).toarray()

        else:
            df_train, df_test = train_test_split_pipeline(df_int, percent, 'DistilBERT Multilingua', fold)
            df_outlier = np.array(df_out['DistilBERT Multilingua'].to_list())

            df_train_fet, df_test_fet = train_test_split_pipeline(df_int, percent, 'features_normalized', fold)
            df_train_fet = df_train_fet.astype('float32')
            df_test_fet = df_test_fet.astype('float32')
            df_outlier_fet = np.array(df_out['features_normalized'].to_list()).astype('float32')

            if prepro == 'DBERTML':
                x_train = df_train
                x_test = df_test
                x_outlier = df_outlier

            elif prepro == 'LIWC':
                x_train = df_train_fet
                x_test = df_test_fet
                x_outlier = df_outlier_fet

            elif prepro == 'Font':
                df_train_font, df_test_font = train_test_split_pipeline(df_int, percent, 'font_onehot', fold)
                df_outlier_font = np.array(df_out['font_onehot'].to_list())

                x_train = df_train_font
                x_test = df_test_font
                x_outlier = df_outlier_font

                del df_train_font
                del df_test_font
                del df_outlier_font
                collect()

            elif prepro == 'Density':
                density_train, density_test, density_outlier = make_density_information(cluster_list, df_train, df_test,
                                                                                        df_outlier)
                x_train = density_train
                x_test = density_test
                x_outlier = density_outlier

                del density_train
                del density_test
                del density_outlier
                collect()

            elif prepro == 'AE':

                epoch = parameter_list[0]
                arq = parameter_list[1]

                tf.random.set_seed(1)

                ae, encoder = autoencoder(arq, len(df_train[0]))

                ae.fit(df_train, df_train, epochs=epoch, batch_size=32, verbose=0)

                x_train = encoder.predict(df_train)
                x_test = encoder.predict(df_test)
                x_outlier = encoder.predict(df_outlier)

            elif prepro == 'VAE':

                epoch = parameter_list[0]
                arq = parameter_list[1]

                tf.random.set_seed(1)

                vae, encoder, decoder = variationalautoencoder(arq, len(df_train[0]))

                vae.fit(df_train, df_train, epochs=epoch, batch_size=32, verbose=0)

                x_train, _, _ = encoder.predict(df_train)
                x_test, _, _ = encoder.predict(df_test)
                x_outlier, _, _ = encoder.predict(df_outlier)

            elif prepro == 'Concat-DBBERT-Density':
                density_train, density_test, density_outlier = make_density_information(cluster_list, df_train, df_test,
                                                                                        df_outlier)

                x_train = np.concatenate([df_train, density_train], axis=1)
                x_test = np.concatenate([df_test, density_test], axis=1)
                x_outlier = np.concatenate([df_outlier, density_outlier], axis=1)

                del density_train
                del density_test
                del density_outlier
                collect()

            elif prepro == 'Concat-AE-Density':
                density_train, density_test, density_outlier = make_density_information(cluster_list, df_train, df_test,
                                                                                        df_outlier)

                epoch = parameter_list[0]
                arq = parameter_list[1]

                tf.random.set_seed(1)

                ae, encoder = autoencoder(arq, len(df_train[0]))

                ae.fit(df_train, df_train, epochs=epoch, batch_size=32, verbose=0)

                ae_train = encoder.predict(df_train)
                ae_test = encoder.predict(df_test)
                ae_outlier = encoder.predict(df_outlier)

                x_train = np.concatenate([ae_train, density_train], axis=1)
                x_test = np.concatenate([ae_test, density_test], axis=1)
                x_outlier = np.concatenate([ae_outlier, density_outlier], axis=1)

                del density_train
                del density_test
                del density_outlier
                del ae_train
                del ae_test
                del ae_outlier
                collect()

            elif prepro == 'Concat-VAE-Density':
                density_train, density_test, density_outlier = make_density_information(cluster_list, df_train, df_test,
                                                                                        df_outlier)

                epoch = parameter_list[0]
                arq = parameter_list[1]

                tf.random.set_seed(1)

                vae, encoder, decoder = variationalautoencoder(arq, len(df_train[0]))

                vae.fit(df_train, df_train, epochs=epoch, batch_size=32, verbose=0)

                vae_train, _, _ = encoder.predict(df_train)
                vae_test, _, _ = encoder.predict(df_test)
                vae_outlier, _, _ = encoder.predict(df_outlier)

                x_train = np.concatenate([vae_train, density_train], axis=1)
                x_test = np.concatenate([vae_test, density_test], axis=1)
                x_outlier = np.concatenate([vae_outlier, density_outlier], axis=1)

                del density_train
                del density_test
                del density_outlier
                del vae_train
                del vae_test
                del vae_outlier
                collect()

            elif prepro == 'MAE-LIWC':

                epoch = parameter_list[0]
                arq = parameter_list[1]
                operator = parameter_list[2]

                tf.random.set_seed(1)

                aem, encoder = multimodal_autoencoder(arq, len(df_train[0]), len(df_train_fet[0]), operator)

                aem.fit([df_train, df_train_fet], [df_train, df_train_fet], epochs=epoch, batch_size=32, verbose=0)

                x_train = encoder.predict([df_train, df_train_fet])
                x_test = encoder.predict([df_test, df_test_fet])
                x_outlier = encoder.predict([df_outlier, df_outlier_fet])

            elif prepro == 'MAE-FK':
                density_train, density_test, density_outlier = make_density_information(cluster_list, df_train, df_test,
                                                                                        df_outlier)

                epoch = parameter_list[0]
                arq = parameter_list[1]
                operator = parameter_list[2]

                tf.random.set_seed(1)

                aem, encoder = multimodal_autoencoder(arq, len(df_train[0]), len(density_train[0]), operator)

                aem.fit([df_train, density_train], [df_train, density_train], epochs=epoch, batch_size=32, verbose=0)

                x_train = encoder.predict([df_train, density_train])
                x_test = encoder.predict([df_test, density_test])
                x_outlier = encoder.predict([df_outlier, density_outlier])

                del density_train
                del density_test
                del density_outlier
                collect()

            elif prepro == 'MAE-Triple':

                density_train, density_test, density_outlier = make_density_information(cluster_list, df_train, df_test,
                                                                                        df_outlier)

                epoch = parameter_list[0]
                arq = parameter_list[1]
                operator = parameter_list[2]

                tf.random.set_seed(1)

                tae, encoder = triple_autoencoder(arq, len(df_train[0]), len(density_train[0]), len(df_train_fet[0]),
                                                  operator)

                tae.fit([df_train, density_train, df_train_fet], [df_train, density_train, df_train_fet], epochs=epoch,
                        batch_size=32, verbose=0)

                x_train = encoder.predict([df_train, density_train, df_train_fet])
                x_test = encoder.predict([df_test, density_test, df_test_fet])
                x_outlier = encoder.predict([df_outlier, density_outlier, df_outlier_fet])

                del density_train
                del density_test
                del density_outlier
                collect()

            elif prepro == 'MVAE-LIWC':

                epoch = parameter_list[0]
                arq = parameter_list[1]
                operator = parameter_list[2]

                tf.random.set_seed(1)

                mvae, encoder, decoder = multimodalvae(arq, len(df_train[0]), len(df_train_fet[0]), operator)

                mvae.fit([df_train, df_train_fet], [df_train, df_train_fet], epochs=epoch, batch_size=32, verbose=0)

                x_train, _, _ = encoder.predict([df_train, df_train_fet])
                x_test, _, _ = encoder.predict([df_test, df_test_fet])
                x_outlier, _, _ = encoder.predict([df_outlier, df_outlier_fet])

            elif prepro == 'MVAE-FK':
                density_train, density_test, density_outlier = make_density_information(cluster_list, df_train, df_test,
                                                                                        df_outlier)

                epoch = parameter_list[0]
                arq = parameter_list[1]
                operator = parameter_list[2]

                tf.random.set_seed(1)

                mvae, encoder, decoder = multimodalvae(arq, len(df_train[0]), len(cluster_list), operator)

                mvae.fit([df_train, density_train], [df_train, density_train], epochs=epoch, batch_size=32, verbose=0)

                x_train, _, _ = encoder.predict([df_train, density_train])
                x_test, _, _ = encoder.predict([df_test, density_test])
                x_outlier, _, _ = encoder.predict([df_outlier, density_outlier])

                del density_train
                del density_test
                del density_outlier
                collect()

            elif prepro == 'MVAE-Fonts':
                df_train_font, df_test_font = train_test_split_pipeline(df_int, percent, 'font_onehot', fold)
                df_outlier_font = np.array(df_out['font_onehot'].to_list())

                epoch = parameter_list[0]
                arq = parameter_list[1]
                operator = parameter_list[2]

                tf.random.set_seed(1)

                mvae, encoder, decoder = multimodalvae(arq, len(df_train[0]), len(df_train_font[0]), operator)

                mvae.fit([df_train, df_train_font], [df_train, df_train_font], epochs=epoch, batch_size=32, verbose=0)

                x_train, _, _ = encoder.predict([df_train, df_train_font])
                x_test, _, _ = encoder.predict([df_test, df_test_font])
                x_outlier, _, _ = encoder.predict([df_outlier, df_outlier_font])

                del df_train_font
                del df_test_font
                del df_outlier_font
                collect()

            elif prepro == 'MVAE-Triple':
                density_train, density_test, density_outlier = make_density_information(cluster_list, df_train, df_test,
                                                                                        df_outlier)

                epoch = parameter_list[0]
                arq = parameter_list[1]
                operator = parameter_list[2]

                tf.random.set_seed(1)

                tvae, encoder, decoder = triplevae(arq, len(df_train[0]), len(cluster_list), len(df_train_fet[0]),
                                                   operator)

                tvae.fit([df_train, density_train, df_train_fet],
                         [df_train, density_train, df_train_fet], epochs=epoch, batch_size=32, verbose=0)

                x_train, _, _ = encoder.predict([df_train, density_train, df_train_fet])
                x_test, _, _ = encoder.predict([df_test, density_test, df_test_fet])
                x_outlier, _, _ = encoder.predict([df_outlier, density_outlier, df_outlier_fet])

                del density_train
                del density_test
                del density_outlier
                collect()

            elif prepro == 'MVAE-LIWC-Fonts':

                df_train_font, df_test_font = train_test_split_pipeline(df_int, percent, 'font_onehot', fold)
                df_outlier_font = np.array(df_out['font_onehot'].to_list())

                epoch = parameter_list[0]
                arq = parameter_list[1]
                operator = parameter_list[2]

                tf.random.set_seed(1)

                tvae, encoder, decoder = triplevae(arq, len(df_train[0]), len(df_train_font[0]), len(df_train_fet[0]),
                                                   operator)

                tvae.fit([df_train, df_train_font, df_train_fet],
                         [df_train, df_train_font, df_train_fet], epochs=epoch, batch_size=32, verbose=0)

                x_train, _, _ = encoder.predict([df_train, df_train_font, df_train_fet])
                x_test, _, _ = encoder.predict([df_test, df_test_font, df_test_fet])
                x_outlier, _, _ = encoder.predict([df_outlier, df_outlier_font, df_outlier_fet])

                del df_train_font
                del df_test_font
                del df_outlier_font
                collect()

            elif prepro == 'MVAE-FK-Fonts':
                density_train, density_test, density_outlier = make_density_information(cluster_list, df_train, df_test,
                                                                                        df_outlier)

                df_train_font, df_test_font = train_test_split_pipeline(df_int, percent, 'font_onehot', fold)
                df_outlier_font = np.array(df_out['font_onehot'].to_list())

                epoch = parameter_list[0]
                arq = parameter_list[1]
                operator = parameter_list[2]

                tf.random.set_seed(1)

                tvae, encoder, decoder = triplevae(arq, len(df_train[0]), len(cluster_list), len(df_train_font[0]),
                                                   operator)

                tvae.fit([df_train, density_train, df_train_font],
                         [df_train, density_train, df_train_font], epochs=epoch, batch_size=32, verbose=0)

                x_train, _, _ = encoder.predict([df_train, density_train, df_train_font])
                x_test, _, _ = encoder.predict([df_test, density_test, df_test_font])
                x_outlier, _, _ = encoder.predict([df_outlier, density_outlier, df_outlier_font])

                del df_train_font
                del df_test_font
                del df_outlier_font
                del density_train
                del density_test
                del density_outlier
                collect()

            del df_train_fet
            del df_test_fet
            del df_outlier_fet
            collect()

        representations.append((x_train, x_test, x_outlier))
        del df_train
        del df_test
        del df_outlier
        del x_train
        del x_test
        del x_outlier
        collect()

    return representations


def multimodal_autoencoder(arq, first_input_len, second_input_len, operator):
    first_input = Input(shape=(first_input_len,), name='first_input_encoder')

    second_input = Input(shape=(second_input_len,), name='second_input_encoder')

    l1 = Dense(np.max([first_input_len, second_input_len]), activation='linear')(first_input)
    l2 = Dense(np.max([first_input_len, second_input_len]), activation='linear')(second_input)

    fusion = None
    if operator == 'concatenate':
        fusion = concatenate([l1, l2])
    if operator == 'multiply':
        fusion = multiply([l1, l2])
    if operator == 'average':
        fusion = average([l1, l2])
    if operator == 'subtract':
        fusion = subtract([l1, l2])
    if operator == 'add':
        fusion = add([l1, l2])
    if operator == 'max':
        fusion = maximum([l1, l2])
    if operator == 'min':
        fusion = minimum([l1, l2])

    if len(arq) == 3:
        first_dense_encoder = Dense(arq[0], activation="linear")(fusion)

        second_dense_encoder = Dense(arq[1], activation="linear")(first_dense_encoder)

        encoded = Dense(arq[2], activation="linear")(second_dense_encoder)

        first_dense_decoder = Dense(arq[1], activation="linear")(encoded)

        second_dense_decoder = Dense(arq[0], activation="linear")(first_dense_decoder)

        first_decoder_output = Dense(first_input_len, activation="linear")(second_dense_decoder)

        second_decoder_output = Dense(second_input_len, activation="linear")(second_dense_decoder)

    elif len(arq) == 2:
        first_dense_encoder = Dense(arq[0], activation="linear")(fusion)

        encoded = Dense(arq[1], activation="linear")(first_dense_encoder)

        first_dense_decoder = Dense(arq[0], activation="linear")(encoded)

        first_decoder_output = Dense(first_input_len, activation="linear")(first_dense_decoder)

        second_decoder_output = Dense(second_input_len, activation="linear")(first_dense_decoder)

    else:
        encoded = Dense(arq[0], activation="linear")(fusion)

        first_decoder_output = Dense(first_input_len, activation="linear")(encoded)

        second_decoder_output = Dense(second_input_len, activation="linear")(encoded)

    encoder = Model([first_input, second_input], encoded)

    autoencoder = Model([first_input, second_input], [first_decoder_output, second_decoder_output])

    autoencoder.compile(optimizer=tensorflow.keras.optimizers.Adam(), loss='mse')

    return autoencoder, encoder


def triple_autoencoder(arq, first_input_len, second_input_len, third_input_len, operator):
    first_input = Input(shape=(first_input_len,), name='first_input_encoder')

    second_input = Input(shape=(second_input_len,), name='second_input_encoder')

    third_input = Input(shape=(third_input_len,), name='third_input_encoder')

    l1 = Dense(np.max([first_input_len, second_input_len, third_input_len]), activation='linear')(first_input)
    l2 = Dense(np.max([first_input_len, second_input_len, third_input_len]), activation='linear')(second_input)
    l3 = Dense(np.max([first_input_len, second_input_len, third_input_len]), activation='linear')(third_input)

    fusion = None
    if operator == 'concatenate':
        fusion = concatenate([l1, l2, l3])
    if operator == 'multiply':
        fusion = multiply([l1, l2, l3])
    if operator == 'average':
        fusion = average([l1, l2, l3])
    if operator == 'subtract':
        fusion = subtract([l1, l2])
        fusion = subtract([fusion, l3])
    if operator == 'add':
        fusion = add([l1, l2, l3])
    if operator == 'max':
        fusion = maximum([l1, l2])
    if operator == 'min':
        fusion = minimum([l1, l2])

    if len(arq) == 3:
        first_dense_encoder = Dense(arq[0], activation="linear")(fusion)

        second_dense_encoder = Dense(arq[1], activation="linear")(first_dense_encoder)

        encoded = Dense(arq[2], activation="linear")(second_dense_encoder)

        first_dense_decoder = Dense(arq[1], activation="linear")(encoded)

        second_dense_decoder = Dense(arq[0], activation="linear")(first_dense_decoder)

        first_decoder_output = Dense(first_input_len, activation="linear")(second_dense_decoder)

        second_decoder_output = Dense(second_input_len, activation="linear")(second_dense_decoder)

        third_decoder_output = Dense(third_input_len, activation="linear")(second_dense_decoder)

    elif len(arq) == 2:
        first_dense_encoder = Dense(arq[0], activation="linear")(fusion)

        encoded = Dense(arq[1], activation="linear")(first_dense_encoder)

        first_dense_decoder = Dense(arq[0], activation="linear")(encoded)

        first_decoder_output = Dense(first_input_len, activation="linear")(first_dense_decoder)

        second_decoder_output = Dense(second_input_len, activation="linear")(first_dense_decoder)

        third_decoder_output = Dense(third_input_len, activation="linear")(first_dense_decoder)

    else:
        encoded = Dense(arq[0], activation="linear")(fusion)

        first_decoder_output = Dense(first_input_len, activation="linear")(encoded)

        second_decoder_output = Dense(second_input_len, activation="linear")(encoded)

        third_decoder_output = Dense(third_input_len, activation="linear")(encoded)

    encoder = Model([first_input, second_input, third_input], encoded)

    autoencoder = Model([first_input, second_input, third_input],
                        [first_decoder_output, second_decoder_output, third_decoder_output])

    autoencoder.compile(optimizer=tensorflow.keras.optimizers.Adam(), loss='mse')

    return autoencoder, encoder


def make_prepro_evaluate(models, file_name, line_parameters, path, prepro, df_int, df_out, folds, percent,
                         vectorizer=CountVectorizer(), cluster_list=(), parameter_list=()):
    representations = make_representation(prepro, df_int, df_out, folds, percent, vectorizer, cluster_list,
                                          parameter_list)

    evaluate_models(models, representations, file_name, line_parameters, path, percent)

    del representations
    collect()


def preprocessing_evaluate(datasets_dictionary, dataset, prepro, models, path_results, term_weight_list, percents,
                           cluster_matrix, epochs, arqs, operators):
    df = datasets_dictionary[dataset]

    df_int = df[df['class'] == 1]

    df_out = df[df['class'] == -1]

    folds = df_int['fold'].unique()

    file_name = dataset + '_' + prepro
    line_parameters = ''

    for percent in percents:
        if prepro == 'DBERTML' or prepro == 'LIWC' or prepro == 'Font':
            make_prepro_evaluate(models, file_name, line_parameters, path_results, prepro, df_int, df_out, folds,
                                 percent)

        elif prepro == 'BoW':
            for term_weight in term_weight_list:
                file_name = dataset + '_' + prepro + '-' + term_weight
                vectorizer = term_weight_type(term_weight)
                make_prepro_evaluate(models, file_name, line_parameters, path_results, prepro, df_int, df_out, folds,
                                     percent, vectorizer=vectorizer)

        elif prepro == 'Density' or prepro == 'Concat-DBBERT-Density':
            for cluster_list in cluster_matrix:
                line_parameters = str(cluster_list)
                make_prepro_evaluate(models, file_name, line_parameters, path_results, prepro, df_int, df_out, folds,
                                     percent, cluster_list=cluster_list)

        elif prepro == 'Concat-AE-Density' or prepro == 'Concat-VAE-Density':
            for epoch in epochs:

                for arq in arqs:
                    for cluster_list in cluster_matrix:
                        line_parameters = str(epoch) + '_' + str(arq) + '_' + str(cluster_list)

                        parameter_list = (epoch, arq)

                        make_prepro_evaluate(models, file_name, line_parameters, path_results, prepro, df_int, df_out,
                                             folds, percent, cluster_list=cluster_list,  parameter_list=parameter_list)
        else:
            for epoch in epochs:
                for arq in arqs:
                    if prepro == 'AE' or prepro == 'VAE':
                        line_parameters = str(epoch) + '_' + str(arq)
                        parameter_list = (epoch, arq)

                        make_prepro_evaluate(models, file_name, line_parameters, path_results, prepro, df_int, df_out,
                                             folds, percent, parameter_list=parameter_list)

                    else:
                        for operator in operators:
                            if prepro == 'MVAE-FK' or prepro == 'MVAE-FK-Fonts' or prepro == 'MVAE-Triple' or prepro == 'MAE-FK' or prepro == 'MAE-Triple':
                                for cluster_list in cluster_matrix:
                                    line_parameters = str(epoch) + '_' + str(arq) + '_' + str(operator) + '_' + str(
                                        cluster_list)

                                    parameter_list = (epoch, arq, operator)

                                    make_prepro_evaluate(models, file_name, line_parameters, path_results, prepro,
                                                         df_int, df_out, folds, percent, cluster_list=cluster_list,
                                                         parameter_list=parameter_list)

                            elif prepro == 'MVAE-LIWC' or prepro == 'MVAE-LIWC-Fonts' or prepro == 'MVAE-Fonts' or prepro == 'MAE-LIWC':
                                line_parameters = str(epoch) + '_' + str(arq) + '_' + str(operator)

                                parameter_list = (epoch, arq, operator)

                                make_prepro_evaluate(models, file_name, line_parameters, path_results, prepro,
                                                     df_int, df_out, folds, percent, parameter_list=parameter_list)
    del df
    del df_int
    del df_out
    collect()


def run(datasets_dictionary, models, prepros, path_results, term_weight_list, percents, cluster_matrix, epochs, arqs,
        operators):
    for dataset in tqdm(datasets_dictionary.keys()):
        for prepro in prepros:
            preprocessing_evaluate(datasets_dictionary, dataset, prepro, models, path_results, term_weight_list,
                                   percents, cluster_matrix, epochs, arqs, operators)


if __name__ == '__main__':

    warnings.filterwarnings("ignore")
    print('Start')

    path_results = '../results_2/'
    term_weight_list = ['term-frequency-IDF', 'term-frequency', 'binary']
    percents = [3, 5, 7, 10]
    cluster_matrix = [[2, 3], [2, 6], [2, 4, 5], [3, 6, 9, 12, 15], [3, 4, 6, 7, 8], [10, 11, 12, 13, 14, 15]]
    epochs = [5, 10, 15, 20, 25, 50]
    arqs = [[384, 256, 128], [256, 128, 64], [384, 256], [256, 128], [256], [64]]
    operators = ['multiply', 'concatenate', 'subtract', 'add', 'average', 'max', 'min']

    models_svdd = {
        'SVDD_RBF_0.0001_scale': SVDD(kernel='rbf', nu=0.0001, gamma='scale'),
        'SVDD_RBF_0.0005_scale': SVDD(kernel='rbf', nu=0.0005, gamma='scale'),
        'SVDD_RBF_0.001_scale': SVDD(kernel='rbf', nu=0.001, gamma='scale'),
        'SVDD_RBF_0.005_scale': SVDD(kernel='rbf', nu=0.005, gamma='scale'),
        'SVDD_RBF_0.01_scale': SVDD(kernel='rbf', nu=0.01, gamma='scale'),
        'SVDD_RBF_0.05_scale': SVDD(kernel='rbf', nu=0.05, gamma='scale'),
        'SVDD_RBF_0.1_scale': SVDD(kernel='rbf', nu=0.1, gamma='scale'),
        'SVDD_RBF_0.2_scale': SVDD(kernel='rbf', nu=0.2, gamma='scale'),
        'SVDD_RBF_0.3_scale': SVDD(kernel='rbf', nu=0.3, gamma='scale'),
        'SVDD_RBF_0.4_scale': SVDD(kernel='rbf', nu=0.4, gamma='scale'),
        'SVDD_RBF_0.5_scale': SVDD(kernel='rbf', nu=0.5, gamma='scale'),
        'SVDD_RBF_0.6_scale': SVDD(kernel='rbf', nu=0.6, gamma='scale'),
        'SVDD_RBF_0.7_scale': SVDD(kernel='rbf', nu=0.7, gamma='scale'),
        'SVDD_RBF_0.8_scale': SVDD(kernel='rbf', nu=0.8, gamma='scale'),
        'SVDD_RBF_0.9_scale': SVDD(kernel='rbf', nu=0.9, gamma='scale'),
        'SVDD_RBF_0.0001_auto': SVDD(kernel='rbf', nu=0.0001, gamma='auto'),
        'SVDD_RBF_0.0005_auto': SVDD(kernel='rbf', nu=0.0005, gamma='auto'),
        'SVDD_RBF_0.001_auto': SVDD(kernel='rbf', nu=0.001, gamma='auto'),
        'SVDD_RBF_0.005_auto': SVDD(kernel='rbf', nu=0.005, gamma='auto'),
        'SVDD_RBF_0.01_auto': SVDD(kernel='rbf', nu=0.01, gamma='auto'),
        'SVDD_RBF_0.05_auto': SVDD(kernel='rbf', nu=0.05, gamma='auto'),
        'SVDD_RBF_0.1_auto': SVDD(kernel='rbf', nu=0.1, gamma='auto'),
        'SVDD_RBF_0.2_auto': SVDD(kernel='rbf', nu=0.2, gamma='auto'),
        'SVDD_RBF_0.3_auto': SVDD(kernel='rbf', nu=0.3, gamma='auto'),
        'SVDD_RBF_0.4_auto': SVDD(kernel='rbf', nu=0.4, gamma='auto'),
        'SVDD_RBF_0.5_auto': SVDD(kernel='rbf', nu=0.5, gamma='auto'),
        'SVDD_RBF_0.6_auto': SVDD(kernel='rbf', nu=0.6, gamma='auto'),
        'SVDD_RBF_0.7_auto': SVDD(kernel='rbf', nu=0.7, gamma='auto'),
        'SVDD_RBF_0.8_auto': SVDD(kernel='rbf', nu=0.8, gamma='auto'),
        'SVDD_RBF_0.9_auto': SVDD(kernel='rbf', nu=0.9, gamma='auto'),
        'SVDD_linear_0.0001_scale': SVDD(kernel='linear', nu=0.0001, gamma='scale'),
        'SVDD_linear_0.0005_scale': SVDD(kernel='linear', nu=0.0005, gamma='scale'),
        'SVDD_linear_0.001_scale': SVDD(kernel='linear', nu=0.001, gamma='scale'),
        'SVDD_linear_0.005_scale': SVDD(kernel='linear', nu=0.005, gamma='scale'),
        'SVDD_linear_0.01_scale': SVDD(kernel='linear', nu=0.01, gamma='scale'),
        'SVDD_linear_0.05_scale': SVDD(kernel='linear', nu=0.05, gamma='scale'),
        'SVDD_linear_0.1_scale': SVDD(kernel='linear', nu=0.1, gamma='scale'),
        'SVDD_linear_0.2_scale': SVDD(kernel='linear', nu=0.2, gamma='scale'),
        'SVDD_linear_0.3_scale': SVDD(kernel='linear', nu=0.3, gamma='scale'),
        'SVDD_linear_0.4_scale': SVDD(kernel='linear', nu=0.4, gamma='scale'),
        'SVDD_linear_0.5_scale': SVDD(kernel='linear', nu=0.5, gamma='scale'),
        'SVDD_linear_0.6_scale': SVDD(kernel='linear', nu=0.6, gamma='scale'),
        'SVDD_linear_0.7_scale': SVDD(kernel='linear', nu=0.7, gamma='scale'),
        'SVDD_linear_0.8_scale': SVDD(kernel='linear', nu=0.8, gamma='scale'),
        'SVDD_linear_0.9_scale': SVDD(kernel='linear', nu=0.9, gamma='scale'),
        'SVDD_linear_0.0001_auto': SVDD(kernel='linear', nu=0.0001, gamma='auto'),
        'SVDD_linear_0.0005_auto': SVDD(kernel='linear', nu=0.0005, gamma='auto'),
        'SVDD_linear_0.001_auto': SVDD(kernel='linear', nu=0.001, gamma='auto'),
        'SVDD_linear_0.005_auto': SVDD(kernel='linear', nu=0.005, gamma='auto'),
        'SVDD_linear_0.01_auto': SVDD(kernel='linear', nu=0.01, gamma='auto'),
        'SVDD_linear_0.05_auto': SVDD(kernel='linear', nu=0.05, gamma='auto'),
        'SVDD_linear_0.1_auto': SVDD(kernel='linear', nu=0.1, gamma='auto'),
        'SVDD_linear_0.2_auto': SVDD(kernel='linear', nu=0.2, gamma='auto'),
        'SVDD_linear_0.3_auto': SVDD(kernel='linear', nu=0.3, gamma='auto'),
        'SVDD_linear_0.4_auto': SVDD(kernel='linear', nu=0.4, gamma='auto'),
        'SVDD_linear_0.5_auto': SVDD(kernel='linear', nu=0.5, gamma='auto'),
        'SVDD_linear_0.6_auto': SVDD(kernel='linear', nu=0.6, gamma='auto'),
        'SVDD_linear_0.7_auto': SVDD(kernel='linear', nu=0.7, gamma='auto'),
        'SVDD_linear_0.8_auto': SVDD(kernel='linear', nu=0.8, gamma='auto'),
        'SVDD_linear_0.9_auto': SVDD(kernel='linear', nu=0.9, gamma='auto'),
        'SVDD_sigmoid_0.0001_scale': SVDD(kernel='sigmoid', nu=0.0001, gamma='scale'),
        'SVDD_sigmoid_0.0005_scale': SVDD(kernel='sigmoid', nu=0.0005, gamma='scale'),
        'SVDD_sigmoid_0.001_scale': SVDD(kernel='sigmoid', nu=0.001, gamma='scale'),
        'SVDD_sigmoid_0.005_scale': SVDD(kernel='sigmoid', nu=0.005, gamma='scale'),
        'SVDD_sigmoid_0.01_scale': SVDD(kernel='sigmoid', nu=0.01, gamma='scale'),
        'SVDD_sigmoid_0.05_scale': SVDD(kernel='sigmoid', nu=0.05, gamma='scale'),
        'SVDD_sigmoid_0.1_scale': SVDD(kernel='sigmoid', nu=0.1, gamma='scale'),
        'SVDD_sigmoid_0.2_scale': SVDD(kernel='sigmoid', nu=0.2, gamma='scale'),
        'SVDD_sigmoid_0.3_scale': SVDD(kernel='sigmoid', nu=0.3, gamma='scale'),
        'SVDD_sigmoid_0.4_scale': SVDD(kernel='sigmoid', nu=0.4, gamma='scale'),
        'SVDD_sigmoid_0.5_scale': SVDD(kernel='sigmoid', nu=0.5, gamma='scale'),
        'SVDD_sigmoid_0.6_scale': SVDD(kernel='sigmoid', nu=0.6, gamma='scale'),
        'SVDD_sigmoid_0.7_scale': SVDD(kernel='sigmoid', nu=0.7, gamma='scale'),
        'SVDD_sigmoid_0.8_scale': SVDD(kernel='sigmoid', nu=0.8, gamma='scale'),
        'SVDD_sigmoid_0.9_scale': SVDD(kernel='sigmoid', nu=0.9, gamma='scale'),
        'SVDD_sigmoid_0.0001_auto': SVDD(kernel='sigmoid', nu=0.0001, gamma='auto'),
        'SVDD_sigmoid_0.0005_auto': SVDD(kernel='sigmoid', nu=0.0005, gamma='auto'),
        'SVDD_sigmoid_0.001_auto': SVDD(kernel='sigmoid', nu=0.001, gamma='auto'),
        'SVDD_sigmoid_0.005_auto': SVDD(kernel='sigmoid', nu=0.005, gamma='auto'),
        'SVDD_sigmoid_0.01_auto': SVDD(kernel='sigmoid', nu=0.01, gamma='auto'),
        'SVDD_sigmoid_0.05_auto': SVDD(kernel='sigmoid', nu=0.05, gamma='auto'),
        'SVDD_sigmoid_0.1_auto': SVDD(kernel='sigmoid', nu=0.1, gamma='auto'),
        'SVDD_sigmoid_0.2_auto': SVDD(kernel='sigmoid', nu=0.2, gamma='auto'),
        'SVDD_sigmoid_0.3_auto': SVDD(kernel='sigmoid', nu=0.3, gamma='auto'),
        'SVDD_sigmoid_0.4_auto': SVDD(kernel='sigmoid', nu=0.4, gamma='auto'),
        'SVDD_sigmoid_0.5_auto': SVDD(kernel='sigmoid', nu=0.5, gamma='auto'),
        'SVDD_sigmoid_0.6_auto': SVDD(kernel='sigmoid', nu=0.6, gamma='auto'),
        'SVDD_sigmoid_0.7_auto': SVDD(kernel='sigmoid', nu=0.7, gamma='auto'),
        'SVDD_sigmoid_0.8_auto': SVDD(kernel='sigmoid', nu=0.8, gamma='auto'),
        'SVDD_sigmoid_0.9_auto': SVDD(kernel='sigmoid', nu=0.9, gamma='auto'),
        'SVDD_poly_2_0.0001_scale': SVDD(kernel='poly', nu=0.0001, gamma='scale', degree=2),
        'SVDD_poly_2_0.0005_scale': SVDD(kernel='poly', nu=0.0005, gamma='scale', degree=2),
        'SVDD_poly_2_0.001_scale': SVDD(kernel='poly', nu=0.001, gamma='scale', degree=2),
        'SVDD_poly_2_0.005_scale': SVDD(kernel='poly', nu=0.005, gamma='scale', degree=2),
        'SVDD_poly_2_0.01_scale': SVDD(kernel='poly', nu=0.01, gamma='scale', degree=2),
        'SVDD_poly_2_0.05_scale': SVDD(kernel='poly', nu=0.05, gamma='scale', degree=2),
        'SVDD_poly_2_0.1_scale': SVDD(kernel='poly', nu=0.1, gamma='scale', degree=2),
        'SVDD_poly_2_0.2_scale': SVDD(kernel='poly', nu=0.2, gamma='scale', degree=2),
        'SVDD_poly_2_0.3_scale': SVDD(kernel='poly', nu=0.3, gamma='scale', degree=2),
        'SVDD_poly_2_0.4_scale': SVDD(kernel='poly', nu=0.4, gamma='scale', degree=2),
        'SVDD_poly_2_0.5_scale': SVDD(kernel='poly', nu=0.5, gamma='scale', degree=2),
        'SVDD_poly_2_0.6_scale': SVDD(kernel='poly', nu=0.6, gamma='scale', degree=2),
        'SVDD_poly_2_0.7_scale': SVDD(kernel='poly', nu=0.7, gamma='scale', degree=2),
        'SVDD_poly_2_0.8_scale': SVDD(kernel='poly', nu=0.8, gamma='scale', degree=2),
        'SVDD_poly_2_0.9_scale': SVDD(kernel='poly', nu=0.9, gamma='scale', degree=2),
        'SVDD_poly_2_0.0001_auto': SVDD(kernel='poly', nu=0.0001, gamma='auto', degree=2),
        'SVDD_poly_2_0.0005_auto': SVDD(kernel='poly', nu=0.0005, gamma='auto', degree=2),
        'SVDD_poly_2_0.001_auto': SVDD(kernel='poly', nu=0.001, gamma='auto', degree=2),
        'SVDD_poly_2_0.005_auto': SVDD(kernel='poly', nu=0.005, gamma='auto', degree=2),
        'SVDD_poly_2_0.01_auto': SVDD(kernel='poly', nu=0.01, gamma='auto', degree=2),
        'SVDD_poly_2_0.05_auto': SVDD(kernel='poly', nu=0.05, gamma='auto', degree=2),
        'SVDD_poly_2_0.1_auto': SVDD(kernel='poly', nu=0.1, gamma='auto', degree=2),
        'SVDD_poly_2_0.2_auto': SVDD(kernel='poly', nu=0.2, gamma='auto', degree=2),
        'SVDD_poly_2_0.3_auto': SVDD(kernel='poly', nu=0.3, gamma='auto', degree=2),
        'SVDD_poly_2_0.4_auto': SVDD(kernel='poly', nu=0.4, gamma='auto', degree=2),
        'SVDD_poly_2_0.5_auto': SVDD(kernel='poly', nu=0.5, gamma='auto', degree=2),
        'SVDD_poly_2_3_0.6_auto': SVDD(kernel='poly', nu=0.6, gamma='auto', degree=2),
        'SVDD_poly_2_0.7_auto': SVDD(kernel='poly', nu=0.7, gamma='auto', degree=2),
        'SVDD_poly_2_0.8_auto': SVDD(kernel='poly', nu=0.8, gamma='auto', degree=2),
        'SVDD_poly_2_0.9_auto': SVDD(kernel='poly', nu=0.9, gamma='auto', degree=2),
        'SVDD_poly_3_0.0001_scale': SVDD(kernel='poly', nu=0.0001, gamma='scale', degree=3),
        'SVDD_poly_3_0.0005_scale': SVDD(kernel='poly', nu=0.0005, gamma='scale', degree=3),
        'SVDD_poly_3_0.001_scale': SVDD(kernel='poly', nu=0.001, gamma='scale', degree=3),
        'SVDD_poly_3_0.005_scale': SVDD(kernel='poly', nu=0.005, gamma='scale', degree=3),
        'SVDD_poly_3_0.01_scale': SVDD(kernel='poly', nu=0.01, gamma='scale', degree=3),
        'SVDD_poly_3_0.05_scale': SVDD(kernel='poly', nu=0.05, gamma='scale', degree=3),
        'SVDD_poly_3_0.1_scale': SVDD(kernel='poly', nu=0.1, gamma='scale', degree=3),
        'SVDD_poly_3_0.2_scale': SVDD(kernel='poly', nu=0.2, gamma='scale', degree=3),
        'SVDD_poly_3_0.3_scale': SVDD(kernel='poly', nu=0.3, gamma='scale', degree=3),
        'SVDD_poly_3_0.4_scale': SVDD(kernel='poly', nu=0.4, gamma='scale', degree=3),
        'SVDD_poly_3_0.5_scale': SVDD(kernel='poly', nu=0.5, gamma='scale', degree=3),
        'SVDD_poly_3_0.6_scale': SVDD(kernel='poly', nu=0.6, gamma='scale', degree=3),
        'SVDD_poly_3_0.7_scale': SVDD(kernel='poly', nu=0.7, gamma='scale', degree=3),
        'SVDD_poly_3_0.8_scale': SVDD(kernel='poly', nu=0.8, gamma='scale', degree=3),
        'SVDD_poly_3_0.9_scale': SVDD(kernel='poly', nu=0.9, gamma='scale', degree=3),
        'SVDD_poly_3_0.0001_auto': SVDD(kernel='poly', nu=0.0001, gamma='auto', degree=3),
        'SVDD_poly_3_0.0005_auto': SVDD(kernel='poly', nu=0.0005, gamma='auto', degree=3),
        'SVDD_poly_3_0.001_auto': SVDD(kernel='poly', nu=0.001, gamma='auto', degree=3),
        'SVDD_poly_3_0.005_auto': SVDD(kernel='poly', nu=0.005, gamma='auto', degree=3),
        'SVDD_poly_3_0.01_auto': SVDD(kernel='poly', nu=0.01, gamma='auto', degree=3),
        'SVDD_poly_3_0.05_auto': SVDD(kernel='poly', nu=0.05, gamma='auto', degree=3),
        'SVDD_poly_3_0.1_auto': SVDD(kernel='poly', nu=0.1, gamma='auto', degree=3),
        'SVDD_poly_3_0.2_auto': SVDD(kernel='poly', nu=0.2, gamma='auto', degree=3),
        'SVDD_poly_3_0.3_auto': SVDD(kernel='poly', nu=0.3, gamma='auto', degree=3),
        'SVDD_poly_3_0.4_auto': SVDD(kernel='poly', nu=0.4, gamma='auto', degree=3),
        'SVDD_poly_3_0.5_auto': SVDD(kernel='poly', nu=0.5, gamma='auto', degree=3),
        'SVDD_poly_3_3_0.6_auto': SVDD(kernel='poly', nu=0.6, gamma='auto', degree=3),
        'SVDD_poly_3_0.7_auto': SVDD(kernel='poly', nu=0.7, gamma='auto', degree=3),
        'SVDD_poly_3_0.8_auto': SVDD(kernel='poly', nu=0.8, gamma='auto', degree=3),
        'SVDD_poly_3_0.9_auto': SVDD(kernel='poly', nu=0.9, gamma='auto', degree=3),
        'SVDD_poly_4_0.0001_scale': SVDD(kernel='poly', nu=0.0001, gamma='scale', degree=4),
        'SVDD_poly_4_0.0005_scale': SVDD(kernel='poly', nu=0.0005, gamma='scale', degree=4),
        'SVDD_poly_4_0.001_scale': SVDD(kernel='poly', nu=0.001, gamma='scale', degree=4),
        'SVDD_poly_4_0.005_scale': SVDD(kernel='poly', nu=0.005, gamma='scale', degree=4),
        'SVDD_poly_4_0.01_scale': SVDD(kernel='poly', nu=0.01, gamma='scale', degree=4),
        'SVDD_poly_4_0.05_scale': SVDD(kernel='poly', nu=0.05, gamma='scale', degree=4),
        'SVDD_poly_4_0.1_scale': SVDD(kernel='poly', nu=0.1, gamma='scale', degree=4),
        'SVDD_poly_4_0.2_scale': SVDD(kernel='poly', nu=0.2, gamma='scale', degree=4),
        'SVDD_poly_4_0.3_scale': SVDD(kernel='poly', nu=0.3, gamma='scale', degree=4),
        'SVDD_poly_4_0.4_scale': SVDD(kernel='poly', nu=0.4, gamma='scale', degree=4),
        'SVDD_poly_4_0.5_scale': SVDD(kernel='poly', nu=0.5, gamma='scale', degree=4),
        'SVDD_poly_4_0.6_scale': SVDD(kernel='poly', nu=0.6, gamma='scale', degree=4),
        'SVDD_poly_4_0.7_scale': SVDD(kernel='poly', nu=0.7, gamma='scale', degree=4),
        'SVDD_poly_4_0.8_scale': SVDD(kernel='poly', nu=0.8, gamma='scale', degree=4),
        'SVDD_poly_4_0.9_scale': SVDD(kernel='poly', nu=0.9, gamma='scale', degree=4),
        'SVDD_poly_4_0.0001_auto': SVDD(kernel='poly', nu=0.0001, gamma='auto', degree=4),
        'SVDD_poly_4_0.0005_auto': SVDD(kernel='poly', nu=0.0005, gamma='auto', degree=4),
        'SVDD_poly_4_0.001_auto': SVDD(kernel='poly', nu=0.001, gamma='auto', degree=4),
        'SVDD_poly_4_0.005_auto': SVDD(kernel='poly', nu=0.005, gamma='auto', degree=4),
        'SVDD_poly_4_0.01_auto': SVDD(kernel='poly', nu=0.01, gamma='auto', degree=4),
        'SVDD_poly_4_0.05_auto': SVDD(kernel='poly', nu=0.05, gamma='auto', degree=4),
        'SVDD_poly_4_0.1_auto': SVDD(kernel='poly', nu=0.1, gamma='auto', degree=4),
        'SVDD_poly_4_0.2_auto': SVDD(kernel='poly', nu=0.2, gamma='auto', degree=4),
        'SVDD_poly_4_0.3_auto': SVDD(kernel='poly', nu=0.3, gamma='auto', degree=4),
        'SVDD_poly_4_0.4_auto': SVDD(kernel='poly', nu=0.4, gamma='auto', degree=4),
        'SVDD_poly_4_0.5_auto': SVDD(kernel='poly', nu=0.5, gamma='auto', degree=4),
        'SVDD_poly_4_3_0.6_auto': SVDD(kernel='poly', nu=0.6, gamma='auto', degree=4),
        'SVDD_poly_4_0.7_auto': SVDD(kernel='poly', nu=0.7, gamma='auto', degree=4),
        'SVDD_poly_4_0.8_auto': SVDD(kernel='poly', nu=0.8, gamma='auto', degree=4),
        'SVDD_poly_4_0.9_auto': SVDD(kernel='poly', nu=0.9, gamma='auto', degree=4)
    }

    all_one_dataset = sys.argv[1]

    all_one_preprocessing = sys.argv[2]

    if all_one_dataset == 'All':
        datasets_dic = {
            'fcn': pd.read_pickle('../datasets/fcn.plk'),
            'fnn': pd.read_pickle('../datasets/fnn.plk'),
            'fakebr': pd.read_pickle('../datasets/fakebr.plk'),
            'fnc0': pd.read_pickle('../datasets/fnc0_font.plk'),
            'fnc1': pd.read_pickle('../datasets/fnc1_font.plk'),
            'cfn': pd.read_pickle('../datasets/cfn.plk'),
        }
    else:
        datasets_dic = {
            all_one_dataset: pd.read_pickle('../datasets/' + all_one_dataset + '.plk')
        }

    if all_one_preprocessing == 'All-Fonts':
        prepros = ['BoW', 'DBERTML', 'Font', 'Density', 'LIWC', 'AE', 'VAE',
                   'Concat-DBBERT-Density', 'Concat-AE-Density', 'Concat-VAE-Density',
                   'MAE-LIWC', 'MAE-FK', 'MAE-Triple', 'MVAE-LIWC', 'MVAE-FK', 'MVAE-Triple',
                   'MVAE-Fonts', 'MVAE-LIWC-Fonts', 'MVAE-FK-Fonts']
    elif all_one_preprocessing == 'All':
        prepros = ['BoW', 'DBERTML', 'Density', 'LIWC', 'AE', 'VAE',
                   'Concat-DBBERT-Density', 'Concat-AE-Density', 'Concat-VAE-Density',
                   'MAE-LIWC', 'MAE-FK', 'MAE-Triple', 'MVAE-LIWC', 'MVAE-FK', 'MVAE-Triple']
    else:
        prepros = [all_one_preprocessing]

    run(datasets_dic, models_svdd, prepros, path_results, term_weight_list, percents, cluster_matrix, epochs, arqs,
        operators)

    print('Done!')
