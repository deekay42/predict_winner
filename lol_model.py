import argparse
import tensorflow as tf
import lol_loader
import json
import random
import numpy as np
from tensorflow.python import debug as tf_debug
import sys


# gridsearch = {
#     "hidden_units": [[10],[64],[128], [256], [512], [1024],
#                      [10,10], [128,64], [256,128], [512,256],
#                      [10, 10, 10], [128, 64, 32], [256, 128, 64], [512, 256, 128],
#                      [10, 10, 10, 10], [64,32,16,8], [128,64,32,16], [256,128,64,32],
#                      [10,10,10,10,10], [128,64,32,16,8], [256,128,64,32,16]],
#
#     "lr": [0.1, 0.01, 0.001, 0.0001],
#     "optimizer": [tf.train.AdagradOptimizer, tf.train.AdamOptimizer, tf.train.GradientDescentOptimizer,
#                tf.train.AdadeltaOptimizer, tf.train.RMSPropOptimizer],
#     "batch_size": [10, 100, 1000],
#     "epochs": [10],
#     "dropout": [0.0, .1, .4],
#     "l2reg": [0.0, .0001, .001, .01]
# }

gridsearch = {
    "hidden_units": [[256, 128, 64, 32], [512,256,128,64], [1024,512,256,128],
                     [256, 128, 64, 32, 16], [512,256,128,64,32], [1024,512,256,128,64]],

    "lr": [0.01, 0.001],
    "optimizer": [tf.train.AdamOptimizer],
    "batch_size": [1000, 10000],
    "epochs": [3],
    "dropout": [0.0],
    "l2reg": [0.0]
}
#input dim is ~13000
#output dim is ~1000
NUM_CHAMPS = 144
NUM_ITEMS = 202
NUM_SPELLS = 9

#total input layer dimension= 140*10 + 18*20 + 6*10*210 = 14360

parser = argparse.ArgumentParser()

def my_model(features, labels, mode, params):

    net = tf.feature_column.input_layer(features, params['feature_columns'])

    for units in params['hidden_units']:
        net = tf.layers.dropout(tf.layers.dense(net, units=units, activation=tf.nn.sigmoid,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(params['l2reg'])),
                                rate=params['dropout'] if mode == tf.estimator.ModeKeys.TRAIN else 0)

    #output layer
    sigmoid_activations = tf.layers.dense(net, params['n_classes'], activation=tf.sigmoid)

    #sigmoid_activations = tf.Print(tmp, [tmp])
    # Compute predictions.
    predicted_classes = tf.round(sigmoid_activations)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': sigmoid_activations,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.log_loss(labels, sigmoid_activations)# + tf.losses.get_regularization_loss()

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.Print(loss, [predicted_classes, labels], summarize=100)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = params['optimizer'](learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def classify_next_item(num_champs, num_items, num_labels, learning_rate):
    champs_one_hot = []
    for _ in range(champs_per_game):
        champs_one_hot += input_data(shape=[None, num_champs], name='input')

    champs_emb = []
    champs_emb[0] = embedding(champs_one_hot[0], input_dim=num_champs, output_dim=champ_emb_dim, reuse=False, scope="champ_scope")
    for i in range(1, champs_per_game):
        champs_emb[i] = embedding(champs_one_hot[0], input_dim=num_champs, output_dim=champ_emb_dim, reuse=True, scope="champ_scope")
    champs = merge(champs_emb, mode='concat', axis=0)


    items_one_hot = []
    for _ in range(champs_per_game*items_per_champ):
        items_one_hot += input_data(shape=[None, num_items], name='input')

    items_emb = []
    items_emb[0] = embedding(items_one_hot[0], input_dim=num_champs, output_dim=item_emb_dim, reuse=False, scope="item_scope")
    for i in range(1, champs_per_game * items_per_champ):
        items_emb[i] = embedding(items_one_hot[0], input_dim=num_champs, output_dim=item_emb_dim, reuse=True, scope="item_scope")

    champ_items = []

    for i in range(num_champs):
        champ_items[i] = merge(items_emb[i*items_per_champ:(i+1)*items_per_champ], mode='sum', axis=1)


    spells = input_data(shape=[None, spells_input_size], name='input')

    final_input_layer = merge([champs,champ_items,spells], mode='concat', axis=0)

    # dim = champs_per_game * champ_emb_dim + champs_per_game * items_per_champ * item_emb_dim + spells_input_size
    # input dim = 410 for standard LoL match

    net = relu(batch_normalization(fully_connected(net, 512, bias=False, activation=None, regularizer="L2")))
    #dropout
    net = relu(batch_normalization(fully_connected(net, 512, bias=False, activation=None, regularizer="L2")))
    net = fully_connected(net, num_labels, activation='sigmoid')

    return regression(net, optimizer='adam', learning_rate=learning_rate,
                      loss='binary_crossentropy', name='target')


predict_winner_champ_select_config = {
        "num_champs": NUM_CHAMPS,
        "champ_ids": "res/champ_ids"
    }

predict_winner_itemization_config = {
        "num_champs": NUM_CHAMPS,
        "champ_ids": "res/champ_ids",
        "num_spells": NUM_SPELLS,
        "spell_ids": "res/summoner_spells_ids",
        "num_items": NUM_ITEMS,
        "item_ids": "res/item_ids"
    }

def main(argv):

    config = predict_winner_itemization_config
    dataloader = lol_loader.LoadData()
    # Fetch the data
    keys = dataloader.getKeys()

    # Feature columns describe how to use the input.
    my_feature_columns = []

    #10 input features for champ select guessing(10 champs)

    for key_index in range(lol_loader.CHAMPS_PER_GAME):
        my_feature_columns.append(tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_file(
                key=keys[key_index],
                vocabulary_file=config['champ_ids'],
                vocabulary_size=config['num_champs'])))

    # 120 additional input features for itemization guessing(10 champs + 20 spells + 10*10 items)
    if config == predict_winner_itemization_config:
        for key_index in range(lol_loader.CHAMPS_PER_GAME, lol_loader.CHAMPS_PER_GAME+lol_loader.SPELLS_PER_GAME):
            my_feature_columns.append(tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_file(
                    key=keys[key_index],
                    vocabulary_file=config['spell_ids'],
                    vocabulary_size=config['num_spells'])))
        for key_index in range(lol_loader.CHAMPS_PER_GAME+lol_loader.SPELLS_PER_GAME,lol_loader.NUM_FEATURES):
            my_feature_columns.append(tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_file(
                    key=keys[key_index],
                    vocabulary_file=config['item_ids'],
                    vocabulary_size=config['num_items'])))


    # config = tf.contrib.learn.RunConfig(master=None, num_cores=0,
    #                                     log_device_placement=False,
    #                                     gpu_memory_fraction=1,
    #                                     tf_random_seed=None,
    #                                     save_summary_steps=1,
    #                                     save_checkpoints_secs=None,
    #                                     save_checkpoints_steps=1,
    #                                     keep_checkpoint_max=5,
    #                                     keep_checkpoint_every_n_hours=10000,
    #                                     log_step_count_steps=1000,
    #                                     evaluation_master='',
    #                                     model_dir=None,
    #                                     session_config=None)

    with open("accuracies", "w") as f:
        historical_accuracy = []

        while True:
            # current_config = dict(
            #     net= random.choice(gridsearch['hidden_units']),
            #     lr= random.choice(gridsearch['lr']),
            #     optimizer= random.choice(gridsearch['optimizer']),
            #     batchsize= random.choice(gridsearch['batch_size']),
            #     epochs= random.choice(gridsearch['epochs']),
            #     dropout= random.choice(gridsearch['dropout']),
            #     l2reg= random.choice(gridsearch['l2reg'])\
            # )

            current_config = dict(
                net= [512,256,128,64,32],
                lr= 0.001,
                optimizer= tf.train.AdamOptimizer,
                batchsize= 10000,
                epochs= 20,
                dropout= 0.0,
                l2reg= 0.0\
            )

            tf.reset_default_graph()

            classifier = tf.estimator.Estimator(
                model_fn=my_model,
                params={
                    'feature_columns': my_feature_columns,
                    'n_classes': 1,
                    'hidden_units': current_config['net'],
                    'learning_rate': current_config['lr'],
                    'optimizer': current_config['optimizer'],
                    'dropout': current_config['dropout'],
                    'l2reg': current_config['l2reg']
                })

            serialized_config = dict(Accuracy='', Configuration=current_config)
            serialized_config['Configuration']['optimizer'] = current_config['optimizer'].__name__

            print('Commencing training')
            # Train the Model.
            epoch_counter = 0
            for i in range(current_config['epochs']):
                print('Starting epoch {}'.format(epoch_counter))
                epoch_counter += 1
                dataloader.prepareNextEpoch()

                #Run through all training examples
                while True:
                    try:
                        classifier.train(
                            input_fn=lambda: dataloader.train_input_fn(current_config['batchsize']))
                    except StopIteration:
                        print("Epoch complete")
                        break
                    print("Epoch {}% complete".format(int(100*(dataloader.training_counter/len(dataloader.train_x_filenames)))))

                #Evaluate once after each epoch
                print('Training epoch complete.\nCommencing evaluation')
                # Evaluate the model.
                eval_result = classifier.evaluate(
                    input_fn=lambda:dataloader.eval_input_fn(current_config['batchsize']))

                serialized_config['Accuracy'] = dict(accuracy=np.float(eval_result['accuracy']),
                                                     loss=np.float(eval_result['loss']),
                                                     global_step=np.int(eval_result['global_step']))
                historical_accuracy.append(serialized_config.copy())
                f.seek(0)
                f.truncate()
                json.dump(historical_accuracy, f)
                f.flush()
                print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))



    # # Generate predictions from the model
    # expected = [[1],[1],[1], [0], [0], [0]]
    # # predict_x = {
    # #     'champ0': ["Maokai",     "Sion",  "Vayne"     ],
    # #     'champ1': ["Sejuani",    "Gragas",  "Xin Zhao"  ],
    # #     'champ2': ["Zoe",        "Ryze",  "Yasuo"     ],
    # #     'champ3': ["Tristana",   "Xayah",  "Jinx"      ],
    # #     'champ4': ["Shen",       "Rakan", "Talon"     ],
    # #     'champ5': ["Garen",     "Jayce",     "Camille"   ],
    # #     'champ6': ["Master Yi", "Olaf",      "Rumble"    ],
    # #     'champ7': ["Zed",       "Fizz",      "Galio"     ],
    # #     'champ8': ["Ashe",      "Kalista",   "Caitlyn"   ],
    # #     'champ9': ["Zyra",      "Soraka",   "Morgana"   ],
    # # }
    #
    # predict_x = {
    #     'champ0': ['14',  '114',    '126', '59', '80', '17'],
    #     'champ1': ['13',  '42',     '38', '112', '4', '163'],
    #     'champ2': ['421', '113',    '421', '79', '79', '154'],
    #     'champ3': ['429', '119',    '81' , '18', '81', '236'],
    #     'champ4': ['12',  '25',     '12', '223', '201', '412'],
    #     'champ5': ['59',  '80',     '17', '14', '114', '126'],
    #     'champ6': ['112', '4',      '163', '13', '42', '38'],
    #     'champ7': ['79',  '79',     '154', '421', '113', '421'],
    #     'champ8': ['18',  '81',     '236', '429', '119', '81'],
    #     'champ9': ['223', '201',    '412', '12', '25', '12'],
    # }
    #
    # #predict_x = {k:[dataloader.champ2id(champ) for champ in v] for k,v in predict_x.items()}
    #
    # predictions = classifier.predict(
    #     input_fn=lambda:dataloader.eval_input_fn(predict_x,
    #                                             labels=None,
    #                                             batch_size=BATCH_SIZE))
    #
    # for pred_dict, expec in zip(predictions, expected):
    #     template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
    #
    #     class_id = pred_dict['class_ids'][0]
    #     probability = pred_dict['probabilities'][0]
    #
    #     print(template.format(class_id,
    #                           100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
