from jq import jq
import json
from random import *
import tensorflow as tf
import math
import numpy as np
import glob
from itertools import zip_longest

TRAIN_TEST_SPLIT=0.99

MAX_ITEMS_PER_CHAMP = 6
EXAMPLES_PER_CHUNK = 60000
CHAMPS_PER_GAME = 10
SPELLS_PER_CHAMP = 2
SPELLS_PER_GAME = SPELLS_PER_CHAMP * CHAMPS_PER_GAME
NUM_FEATURES = CHAMPS_PER_GAME + CHAMPS_PER_GAME * SPELLS_PER_CHAMP + CHAMPS_PER_GAME * MAX_ITEMS_PER_CHAMP

class LoadData:

    def __init__(self):
        self.dataset_test = []
        self.keys = ["team1_top_champ", "team1_mid_champ", "team1_jg_champ", "team1_adc_champ",
                     "team1_supp_champ",
                     "team2_top_champ", "team2_mid_champ", "team2_jg_champ", "team2_adc_champ", "team2_supp_champ",
                     "team1_top_spell1", "team1_mid_spell1", "team1_jg_spell1", "team1_adc_spell1",
                     "team1_supp_spell1",
                     "team1_top_spell2", "team1_mid_spell2", "team1_jg_spell2", "team1_adc_spell2",
                     "team1_supp_spell2",
                     "team2_top_spell1", "team2_mid_spell1", "team2_jg_spell1", "team2_adc_spell1",
                     "team2_supp_spell1",
                     "team2_top_spell2", "team2_mid_spell2", "team2_jg_spell2", "team2_adc_spell2",
                     "team2_supp_spell2"]
        for i in ["team1_top", "team1_mid", "team1_jg", "team1_adc", "team1_supp",
                  "team2_top", "team2_mid", "team2_jg", "team2_adc", "team2_supp"]:
            for j in ["item1", "item2", "item3", "item4", "item5", "item6"]:
                self.keys.append(i + "_" + j)

        self.train_x = np.empty((NUM_FEATURES, 0), str)
        self.train_y = np.empty((0,1), int)
        self.test_x = np.empty((NUM_FEATURES, 0), str)
        self.test_y = np.empty((0,1), int)

        try:
            self.readFromNumpyFiles()
            return
        except FileNotFoundError as error:
            repr(error)
            print("Building numpy database now. This may take a few minutes.")
            self.buildTFRecordsDB()

        try:
            self.readFromNumpyFiles()
            return
        except FileNotFoundError as error:
            repr(error)
            print("Unable to read numpy files. Is your disc full or do you not have write access to directory?")





        # self.train_x = dict(zip(self.keys, self.train_x))
        self.test_x = dict(zip(self.keys, self.test_x))
        # self.dataset_train = tf.data.Dataset.from_tensor_slices((self.train_x, self.train_y))
        # self.dataset_test = tf.data.Dataset.from_tensor_slices((self.test_x, self.test_y))

    @staticmethod
    def _uniformShuffle(l1, l2):
        assert len(l1) == len(l2)
        rng_state = np.random.get_state()
        np.random.shuffle(l1)
        np.random.set_state(rng_state)
        np.random.shuffle(l2)

    def prepareNextEpoch(self):
        self._uniformShuffle(self.train_x_filenames, self.train_y_filenames)
        self.current_training_x_y = self._getNextTrainingFile()
        self.training_counter = 0

    #This function converts the input data into a set of tfrecords files
    def buildTFRecordsDB(self):
        with open("res/final_compressed_split4") as f:
            self.data_x = json.load(f)
            # data+x has shape (n, 2, 5) where n is the number of games, 2 teams, 5 people per team
            # data_x = jq(".[] | [[.participants[:5] | .[] | [.championId]],[.participants[5:10] | .[] | [.championId]]]")\
            #    .transform(data, multiple_output=True)

        self.data_y = []
        input_slices = []
        print("Generating input & output vectors...")
        for game in self.data_x:
            team1_team_champs = np.array(game['participants'])
            team1_team_champs = team1_team_champs[:5]
            team1_team_champs = team1_team_champs[:, np.newaxis]
            team2_team_champs = np.array(game['participants'][5:])[:, np.newaxis]
            team1_team_spells = np.reshape(game['spells'][:5], (10,1))
            team2_team_spells = np.reshape(game['spells'][5:], (10,1))
            for item_snapshot in game['itemsTimeline']:
                team1_team_items_at_time_x = item_snapshot[:5]
                team1_team_items_at_time_x = [np.pad(player_items, (0, MAX_ITEMS_PER_CHAMP-len(player_items)), 'constant', constant_values=(0, 0)) for player_items in team1_team_items_at_time_x]
                team1_team_items_at_time_x = np.reshape(team1_team_items_at_time_x, (-1,1))
                team2_team_items_at_time_x = item_snapshot[5:]
                team2_team_items_at_time_x = [np.pad(player_items, (0, MAX_ITEMS_PER_CHAMP - len(player_items)), 'constant',
                                                      constant_values = (
                                                          0, 0)) for player_items in team2_team_items_at_time_x]
                team2_team_items_at_time_x = np.reshape(team2_team_items_at_time_x, (-1, 1))

                # make sure to include one negative example for each training example
                # if we don't do this the network doesn't learn as well
                # this doubles the number of training examples
                input_slices.append(np.concatenate([team1_team_champs,
                                    team2_team_champs,
                                    team1_team_spells,
                                    team2_team_spells,
                                    team1_team_items_at_time_x,
                                    team2_team_items_at_time_x], 0).astype(int).astype(str))
                input_slices.append(np.concatenate([team2_team_champs,
                                    team1_team_champs,
                                    team2_team_spells,
                                    team1_team_spells,
                                    team2_team_items_at_time_x,
                                    team1_team_items_at_time_x], 0).astype(int).astype(str))
                self.data_y.append([1])
                self.data_y.append([0])

        self.data_x = input_slices
        print("Shuffling...")
        self._uniformShuffle(self.data_x, self.data_y)

        self.size = len(input_slices)

        #data_x now has shape (n, NUM_FEATURES)

        # split data into train and test
        splitpoint = math.floor(TRAIN_TEST_SPLIT * self.size)
        self.train_x = self.data_x[:splitpoint]
        self.test_x = self.data_x[splitpoint:]

        self.train_y = self.data_y[:splitpoint]
        self.test_y = self.data_y[splitpoint:]

        # transform data into right format
        self.train_x = np.reshape(np.concatenate(self.train_x, 1), (NUM_FEATURES, -1))
        self.test_x = np.reshape(np.concatenate(self.test_x, 1), (NUM_FEATURES, -1))


        # now train_x and test_x have shape (NUM_FEATURES,n)
        # [
        #    [], #feature1
        #    [], #feature2
        #    ...
        #    [] #feature130
        # ]


        # self.writeToTFRecords()
        print("Writing to disk...")
        self.writeToNumpyFile()

    def getKeys(self):
        return self.keys

    def champ2id(self, champname):
        try:
            return str(self.lookup_table[champname])
        except AttributeError:
            with open("res/champ2id") as f:
                self.lookup_table = json.load(f)
            try:
                return str(self.lookup_table[champname])
            except KeyError as k:
                print("Key {} not found".format(k))



    def train_input_fn(self, batch_size):

        # def _parse_function(example_proto):
        #     keys_to_features = dict.fromkeys(self.getKeys(), tf.FixedLenFeature([], tf.int64))
        #     keys_to_features['label'] = tf.FixedLenFeature(0, tf.int64)
        #     features = tf.parse_single_example(example_proto, keys_to_features)
        #
        #     labels = features.pop('label', None)
        #
        #     return features, labels
        #
        # self.dataset_train = tf.data.TFRecordDataset(self.train_filenames)
        #
        # self.dataset_train = self.dataset_train.map(_parse_function)
        # self.dataset_train = self.dataset_train.flat_map(lambda self,x: tf.data.Dataset().from_tensor_slices(x))

        # Shuffle, repeat, and batch the examples.

        self.train_x, self.train_y = self.nextTrainingDataset()

        self.train_x = dict(zip(self.keys, self.train_x))
        self.dataset_train = tf.data.Dataset.from_tensor_slices((self.train_x, self.train_y))
        self.dataset_train = self.dataset_train.shuffle(EXAMPLES_PER_CHUNK).batch(batch_size)
        self.training_counter += 1

        # Return the dataset.
        return self.dataset_train

    def eval_input_fn(self, batch_size):

        # self.dataset_test = tf.data.TFRecordDataset(self.test_filenames)
        # self.dataset_test = self.dataset_test.map(self.parse_function)
        # Batch the examples
        assert batch_size is not None, "batch_size must not be None"

        self.dataset_test = tf.data.Dataset.from_tensor_slices((self.test_x, self.test_y))
        self.dataset_test = self.dataset_test.shuffle(EXAMPLES_PER_CHUNK).batch(batch_size)

        # Return the dataset.
        return self.dataset_test

    def predict_input_fn(self, features, labels, batch_size):
        """An input function for evaluation or prediction"""
        if labels is None:
            # No labels, use only features.
            inputs = features
        else:
            inputs = (features, labels)

        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices(inputs)

        # Batch the examples
        assert batch_size is not None, "batch_size must not be None"
        dataset = dataset.batch(batch_size)

        # Return the dataset.
        return dataset


    def readFromTFRecords(self):

        # Creates a dataset that reads all of the examples from filenames.

        self.train_filenames = glob.glob('res/tfrecords/*_train.tfrecords')
        self.test_filenames = glob.glob('res/tfrecords/*_test.tfrecords')
        if not self.train_filenames or not self.test_filenames:
            raise FileNotFoundError("No train or test tfrecords files in that location")


    def _getNextTrainingFile(self):
        for x,y in zip(self.train_x_filenames, self.train_y_filenames):
            yield np.load(x), np.load(y)

    def nextTrainingDataset(self):
        return next(self.current_training_x_y)

    def readFromNumpyFiles(self):

        # Creates a dataset that reads all of the examples from filenames.

        self.train_x_filenames = sorted(glob.glob('res/input_numpys/*_train_x.npy'))
        self.test_x_filenames = sorted(glob.glob('res/input_numpys/*_test_x.npy'))
        self.train_y_filenames = sorted(glob.glob('res/input_numpys/*_train_y.npy'))
        self.test_y_filenames = sorted(glob.glob('res/input_numpys/*_test_y.npy'))

        if not self.train_x_filenames or not self.test_x_filenames or \
                not self.train_y_filenames or not self.test_y_filenames :
            raise FileNotFoundError("No train or test numpy files in that location")

        for i in self.test_x_filenames:
            data = np.load(i)
            self.test_x = np.append(self.test_x, data, 1)

        for i in self.test_y_filenames:
            data = np.load(i)
            self.test_y = np.append(self.test_y, data, 0)

        self.test_x = dict(zip(self.keys, self.test_x))

    def writeToNumpyFile(self):
        def _chunks(l, n):
            n = max(1, n)
            return [l[i:i + n] for i in range(0, len(l), n)]

        for train_test in [(self.train_x, self.train_y, 'train'), (self.test_x, self.test_y, 'test')]:
            counter = 62
            inputs, labels, mode = train_test
            _, inputlen = inputs.shape
            print("Now writing {} numpy files to disk".format(mode))

            for i in zip(_chunks(inputs.transpose(), EXAMPLES_PER_CHUNK), _chunks(labels, EXAMPLES_PER_CHUNK)):
                input_by_feature = i[0].transpose()

                # Writing the serialized example.
                with open('res/input_numpys/'+str(counter)+'_'+mode+'_x.npy', "wb") as writer:
                    np.save(writer, input_by_feature)
                with open('res/input_numpys/'+str(counter)+'_'+mode+'_y.npy', "wb") as writer:
                    np.save(writer, i[1])

                counter += 1
                print("{}% complete".format(int(min(100, 100*(counter*EXAMPLES_PER_CHUNK/inputlen)))))

    # def writeToTFRecords(self):
    #
    #     def _chunks(l, n):
    #         n = max(1, n)
    #         return [l[i:i + n] for i in range(0, len(l), n)]
    #
    #     # break into multiple smaller tfrecord files
    #     chunksize = 1024
    #
    #     for train_test in [(self.train_x, self.train_y, 'train'), (self.test_x, self.test_y, 'test')]:
    #
    #
    #         counter = 0
    #         inputs, labels, mode = train_test
    #         print("Now writing {} tfrecords to disk".format(mode))
    #         labels = [i[0] for i in labels]
    #         inputs = np.append(inputs, [labels], 0)
    #         _, inputlen = inputs.shape
    #
    #         for i in _chunks(inputs.transpose(), chunksize):
    #             input_by_feature = i.transpose()
    #             feature = dict()
    #             feature['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=input_by_feature[-1]))
    #
    #             for key, f in zip(self.getKeys(), input_by_feature[:-1]):
    #                 feature[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=f))
    #
    #             # Create an example protocol buffer
    #             example = tf.train.Example(features=tf.train.Features(feature=feature))
    #             # Writing the serialized example.
    #             with tf.python_io.TFRecordWriter('res/tfrecords/chunk'+str(counter)+'_'+mode+'.tfrecords') as writer:
    #                 writer.write(example.SerializeToString())
    #
    #             counter += 1
    #             print("{}% complete".format(int(min(100, 100*(counter*chunksize/inputlen)))))
