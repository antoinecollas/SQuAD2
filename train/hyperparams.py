class Hyperparams():
    #data
    SAVE_DATA_FOLDER = 'full/'
    MAX_VOCAB = 1000000
    MIN_FREQ = 0
    NUMP_OPS_BPE = 30000
    MAX_NB_PHRASES = 100000 #number of pair of phrases

    #model
    SHARE_WEIGHTS = True
    NB_LAYERS = 2
    NB_HEADS = 2
    D_MODEL = 256
    NB_NEURONS = 1024
    WARMUP_STEPS = 16000

    #constants for training
    TRAIN_SPLIT = 0.8
    NB_EPOCH = 1000
    BATCH_SIZE = 120
    MAX_SEQ = 100

    #constants for evaluation
    EVAL_EVERY_EPOCH = 5
    # PREDICT_BATCH_SIZE = 160

    # BLEU TRAIN
    # 0.91
    # BLEU TEST
    # 0.55