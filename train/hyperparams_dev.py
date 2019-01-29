class Hyperparams():
    #data
    SAVE_DATA_FOLDER = 'dev/'
    MAX_VOCAB = 1000000
    MIN_FREQ = 0
    NUMP_OPS_BPE = 1000
    MAX_NB_PHRASES = 100 #number of pair of phrases

    #model
    SHARE_WEIGHTS = True
    NB_LAYERS = 1
    NB_HEADS = 2
    D_MODEL = 32
    NB_NEURONS = 128
    WARMUP_STEPS = 500
    DROPOUT = 0.1

    #constants for training
    TRAIN_SPLIT = 0.8
    NB_EPOCH = 50
    BATCH_SIZE = 10
    MAX_SEQ = 100


    # PREDICT_BATCH_SIZE = 10