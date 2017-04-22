## Predicting Protein Binding Affinity With Word Embeddings and Recurrent Neural Networks

** Biorxiv link to paper**: http://biorxiv.org/content/early/2017/04/18/128223.article-metrics

To recreate the results reported, download this repo, navigate to the main directory and run `bash project_results_embedding.sh` and `bash project_results_rnn.sh` . The data is already contained in the /data folder, and the results should pop up on the /results directory. Feel free do delete its current contents if you'd like to re-create them yourself.
The bash commands will run a variety of models/model parameters and will store each run in the results folder. For more info on the experiments ran, please refer to the paper submission. Then, run `python analyze_results` to create the visualizations and csv summaries.

NOTE: running the above commands will take LONG (~36 hours). I'll post a script soon to reproduce just the best performing models soon.  


### Creating models and predictions

The main module responsible for the computations is mhcPreds_tflearn_cmd_line.py. It can be run as a standalone command line python program and it accepts a variety of different options:

    mhcPreds_tflearn_cmd_line.py [-h] [-cmd CMD] [-b BATCH_SIZE]
                                 [-bn BATCH_NORM] [-ls LAYER_SIZE]
                                 [-nl NUM_LAYERS] [-d EMBEDDING_SIZE]
                                 [-a ALLELE] [-m MODEL] [-c DATA_ENCODING]
                                 [-r LEARNING_RATE] [-e EPOCHS] [-n NAME]
                                 [-l LEN] [-s SAVE] [--data-dir DATA_DIR]
                                 [--cell-size CELL_SIZE]
                                 [--tensorboard-verbose TENSORBOARD_VERBOSE]
                                 [--from-file FROM_FILE] [--run-id RUN_ID]


For example: `mhcPreds_tflearn_cmd_line.py -cmd 'train_test_eval' -e 15 -bn 1 -nl 3 -c 'kmer_embedding' -a 'A0101' -m 'embedding_rnn' -r 0.001`

Will run the train, test, and evaluation protocol with 15 epochs, 1 round of batch normalization, learning rate being 0.001. It will run on the a subset of the train data set comprised of peptides binding to the HLA-A0101 allele and will transform each kmer in the data set into a 9-mer. Other default parameters can be seen by
Results will be stored to the `/mhcPreds/results/run_id` folder, where run_id is either specified by the user or a randomly selected number between 0 and 10000.


optional arguments:

      -h, --help            show this help message and exit
      -cmd CMD              command
      -b BATCH_SIZE, --batch-size BATCH_SIZE
      -bn BATCH_NORM, --batch-norm BATCH_NORM
                            Perform batch normalization either: only after LSTM (1), after and before (2)
      -ls LAYER_SIZE, --layer-size LAYER_SIZE
                            Size of inner layeres of RNN
      -nl NUM_LAYERS, --num-layers NUM_LAYERS
                            Number of LSTM layers
      -d EMBEDDING_SIZE, --embedding-size EMBEDDING_SIZE
                            Embedding layer output dimension
      -a ALLELE, --allele ALLELE
                            Allele to use for prediction. None predicts for all alleles.
      -m MODEL, --model MODEL
                            RNN model. Basic LSTM, Birectional LSTM or simple RNN
      -c DATA_ENCODING, --data-encoding DATA_ENCODING
                            Embedding layer output dimension
      -r LEARNING_RATE, --learning-rate LEARNING_RATE
                            learning rate (default 0.001)
      -e EPOCHS, --epochs EPOCHS
                            number of trainig epochs
      -n NAME, --name NAME  name of model, used when generating default weights filenames
      -l LEN, --len LEN     size of k-mer to predict on
      -s SAVE, --save SAVE  Save model to --data-dir
      --data-dir DATA_DIR   directory to use for saving models
      --cell-size CELL_SIZE
                            size of RNN cell to use (default 32)
      --tensorboard-verbose TENSORBOARD_VERBOSE
                            tensorboard verbosity level (default 0)
      --run-id RUN_ID       Name of run to be displayed in tensorboard and results folder

### NOTES-1:

Here's a list of possible options for some of the parameters.
        
    POSSIBLE_ALLELES = ['A3101', 'B1509', 'B2703', 'B1517', 'B1801', 'B1501', 'B4002', 'B3901', 'B5701', 'A6801',
                        'B5301', 'A2301', 'A2902', 'B0802', 'A3001', 'A0301', 'A0202', 'A0101', 'B4001', 'B5101',
                        'A1101', 'B4402', 'B0803', 'B5801', 'A2601', 'A0203', 'A3002', 'B4601', 'A3301', 'A6802',
                        'B3801', 'A3201', 'B3501', 'A2603', 'B0702', 'A6901', 'B0801', 'B4501', 'A0206', 'A0201',
                        'B1503', 'A2602', 'A8001', 'A2402', 'B2705', 'B4403', 'A2501', 'B5401']
    
    TRAIN_DEFAULTS = ['A0201', 'A0301', 'A0203', 'A1101', 'A0206', 'A3101']
    AVAILABLE_MODELS = ['deep_rnn', 'embedding_rnn', 'bi_rnn']
    DATA_ENCODINGS = ['one_hot', 'kmer_embedding']

### NOTES-2:

1. `embedding_rnn` requires not parameters referring to a RNN since it's been found that using an embedding layer + hidden layers is sufficient to obtain good accuracy. Adding recurrent layers for the most hurts performance. 
2. Similarly, 'embedding_rnn' requires 'kmer_embedding' as argument, and can not be used with `one_hot` data encoding. 
3. `one_hot` encoding allows the user to specifiy a variety of different architectures, including:
    - bi-directional rnn with a user-definer layer size
    - deep LSTM with a user-defined number of LSTM layers
    - simple rnn with a user-definer layer size
4. One Hot encoding usually leads to slower training due to increased feature dimension







