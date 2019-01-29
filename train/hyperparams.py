class Hyperparams():
    #data
    SAVE_DATA_FOLDER = 'full/'
    MAX_VOCAB = float('inf')
    MIN_FREQ = 0
    NUMP_OPS_BPE = float('inf')
    MAX_NB_PHRASES = float('inf') #number of pair of phrases

    #model
    SHARE_WEIGHTS = True
    NB_LAYERS = 6
    NB_HEADS = 8
    D_MODEL = 512
    NB_NEURONS = 2048
    WARMUP_STEPS = 16000

    #constants for training
    TRAIN_SPLIT = 0.8
    TRAINING_STEPS = 100000
    BATCH_SIZE = 120
    MAX_SEQ = 100
    DROPOUT = 0.1

    EVAL_EVERY_TIMESTEPS = 10
    # PREDICT_BATCH_SIZE = 160