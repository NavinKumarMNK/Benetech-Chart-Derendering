from pathlib import Path

PROMPT_TOKEN = "<|BOS|>"
X_START = "<x_start>"
X_END = "<x_end>"
Y_START = "<y_start>"
Y_END = "<y_end>"

SEPARATOR_TOKENS = [
    PROMPT_TOKEN,
    X_START,
    X_END,
    Y_START,
    Y_END,
]

LINE_TOKEN =  "<line>" 
VERTICAL_BAR_TOKEN = "<vertical_bar>"
HORIZONTAL_BAR_TOKEN = "<horizontal_bar>"
SCATTER_TOKEN = "<scatter>"
DOT_TOKEN = "<dot>"

CHART_TYPE_TOKENS = [
    LINE_TOKEN,
    VERTICAL_BAR_TOKEN,
    HORIZONTAL_BAR_TOKEN,
    SCATTER_TOKEN,
    DOT_TOKEN,
]

NEW_TOKENS = SEPARATOR_TOKENS + CHART_TYPE_TOKENS

class Config:
    # General
    debug = False
    num_proc = 2
    num_workers = 2
    gpus = 2

    # Data
    data_dir = Path('/kaggle/input/benetech-making-graphs-accessible/train')
    images_path = data_dir/'images'
    train_json_files = list(data_dir.glob('annotations/*.json'))

    # Training
    epochs = 5
    val_check_interval = 1.0
    check_val_every_n_epoch = 1
    gradient_clip_val = 2.0
    lr = 2e-5
    lr_scheduler_type = "cosine"
    num_warmup_steps = 100
    seed = 42
    output_path = "output"
    log_steps = 200
    batch_size = 2
    use_wandb = True
    
    image_height = 512
    image_width  = 512
    max_length = 1024
