import argparse
import os

from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-base",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HuggingFace Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Root directory for checkpoints/logs. Defaults to model-specific directory when unset.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "dit"],
        help="Backbone model type to train.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://huggingface.co/papers/2303.09556.",
    )
    parser.add_argument(
        "--dream_training",
        action="store_true",
        help=(
            "Use the DREAM training method, which makes training more efficient and accurate at the "
            "expense of doing an extra forward pass. See: https://huggingface.co/papers/2312.00210"
        ),
    )
    parser.add_argument(
        "--dream_detail_preservation",
        type=float,
        default=1.0,
        help="Dream detail preservation factor p (should be greater than 0; default=1.0, as suggested in the paper)",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--text_enable",
        action="store_true",
        help=(
            "Whether to use text encoder. If set, the text encoder will be used for training. If not set, a custom"
            " condition encoder will be used."
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--offload_ema", action="store_true", help="Offload EMA model to CPU during training step.")
    parser.add_argument("--foreach_ema", action="store_true", help="Use faster foreach implementation of EMAModel.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--test_mode",
        type=str,
        default="test",
    )
    parser.add_argument(
        "--use_text_encoder",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained_encoder_path",
        type=str,
        default=None,
        help="Path to pretrained encoder. If provided, the encoder will be loaded from this path and frozen during training.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=None,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="mosaic",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--image_interpolation_mode",
        type=str,
        default="lanczos",
        choices=[
            f.lower() for f in dir(transforms.InterpolationMode) if not f.startswith("__") and not f.endswith("__")
        ],
        help="The image interpolation method to use for resizing images.",
    )
    parser.add_argument(
        "--tracker_run_name",
        type=str,
        default=None,
        help="The run name for this experiment (used by wandb, etc)."
    )
    parser.add_argument(
        "--drop_rate",
        type=float,
        default=0.1,
        help="Dropout rate for the condition encoder. Default is 0.0 (no dropout).",
    )
    parser.add_argument(
        "--test_accuracy",
        action="store_true",
    )
    parser.add_argument(
        "--id_condition",
        action="store_true",
        help="Whether to not use a condition encoder. If set, the model will not use a condition encoder.",
    )
    parser.add_argument(
        "--class_embeddings_concat",
        action="store_true",
        help="Whether to concatenate class embeddings."
    )
    parser.add_argument("--presaved_path", type=str, default="./presaved_latents", help="Path to the presaved model weights.")
    parser.add_argument("--val_num_samples", type=int, default=5, help="Number of samples to generate for validation.")
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Whether to not save the model checkpoints."
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Stop training once the best validation accuracy has reached --early_stopping_threshold "
             "and has not improved for --early_stopping_patience_steps validations (or _epochs).",
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=None,
        help="Best validation accuracy that must be reached before early stopping can trigger. None = no threshold.",
    )
    parser.add_argument(
        "--early_stopping_patience_steps",
        type=int,
        default=30,
        help="Number of step-based validations without improvement before stopping.",
    )
    parser.add_argument(
        "--early_stopping_patience_epochs",
        type=int,
        default=3,
        help="Number of epoch-based validations without improvement before stopping.",
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=1,
        help="Number of tokens to use for conditioning."
    )
    parser.add_argument(
        "--encoder_learning_rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--switch_training",
        type=int,
        default=None,
        help="Whether to switch training to the condition encoder after a certain number of epochs."
    )
    parser.add_argument(
        "--id_condition_encoder",
        action="store_true",
        help="Whether to use an ID condition encoder. If set, the model will use an ID condition encoder."
    )
    parser.add_argument(
        "--debug_condition",
        action="store_true",
        help="Whether to use a debug condition encoder. If set, the model will use a debug condition encoder."
    )
    parser.add_argument(
        "--off_encoder",
        action="store_true",
        help="Whether to use an off condition encoder. If set, the model will use an off condition encoder."
    )
    parser.add_argument(
        "--condition_encoder_drop_rate",
        type=float,
        default=0.1,
        help="Dropout rate for the condition encoder. Default is 0.1.",
    )
    parser.add_argument(
        "--image_save",
        action="store_true",
        help="Whether to save generated images."
    )
    parser.add_argument(
        "--classification_loss",
        action="store_true",
        help="Whether to use classification learning. If set, the model will use classification learning."
    )
    parser.add_argument(
        "--classification_loss_ce",
        action="store_true",
        help="Whether to use classification learning. If set, the model will use classification learning."
    )
    parser.add_argument(
        "--classification_loss_weight",
        type=float,
        default=1.0,
        help="Weight for the classification loss. Default is 1.0."
    )
    parser.add_argument(
        "--no_image",
        action="store_true",
        help="Whether to not log images during validation."
    )
    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=None,
        help="Batch size for validation."
    )
    parser.add_argument(
        "--timestep_sampling",
        type=float,
        default=None,
        help="Whether to use timestep sampling."
    )
    parser.add_argument(
        "--noise_oversampling",
        action="store_true",
        help="Whether to use noise oversampling at 0.7."
    )
    parser.add_argument(
        "--noise_oversampling2",
        action="store_true",
        help="Whether to use timestep oversampling at 0.2 and 0.5.",
    )
    parser.add_argument(
        "--validation_scheduler",
        type=str,
        default="EulerDiscreteScheduler",
        choices=["DPMSolverMultistepScheduler", "EulerDiscreteScheduler", "DDPMScheduler"],
        help="Select which scheduler to use for validation. DDPMScheduler is recommended for DeepFloyd IF.",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
        help="Weighting scheme for flow matching loss (DiT mode).",
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="Mean to use for 'logit_normal' weighting scheme (DiT mode).",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="Std to use for 'logit_normal' weighting scheme (DiT mode).",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale parameter for 'mode' weighting scheme (DiT mode).",
    )
    parser.add_argument(
        "--precondition_outputs",
        type=int,
        default=1,
        help="Whether to precondition model outputs in DiT flow matching.",
    )

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args
