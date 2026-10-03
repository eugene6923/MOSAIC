import argparse

from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Train a conditional UNet / DiT on a MOSAIC task.")

    # task / run
    parser.add_argument("--test_mode", type=str, required=True,
                        help="<task>[_complex|_composition<N>]_<size>, e.g. count_100000, position_complex_10000, attribute_composition1_10000.")
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "dit"], help="Backbone model type to train.")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Root directory for checkpoints/logs. Defaults to dit_weights/ or unet_weights/.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=128, help="Images are resized to this resolution.")
    parser.add_argument("--image_interpolation_mode", type=str, default="lanczos",
                        choices=[f.lower() for f in dir(transforms.InterpolationMode) if not f.startswith("__") and not f.endswith("__")],
                        help="The image interpolation method to use for resizing images.")
    parser.add_argument("--presaved_path", type=str, default="./presaved_latents", help="Root folder of the VAE latent cache.")

    # pretrained VAE / UNet scheduler
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-2-base",
                        help="Hub id or path providing the VAE and (for unet) the noise/validation schedulers.")
    parser.add_argument("--revision", type=str, default=None, help="Revision of the pretrained model.")
    parser.add_argument("--variant", type=str, default=None, help="Variant of the pretrained model files, e.g. fp16.")

    # optimisation
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help='One of ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]')
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--allow_tf32", action="store_true", help="Allow TF32 matmul on Ampere GPUs.")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    # conditioning
    parser.add_argument("--drop_rate", type=float, default=0.1,
                        help="Probability of zeroing the condition (classifier-free guidance dropout).")
    parser.add_argument("--condition_encoder_drop_rate", type=float, default=0.1,
                        help="nn.Dropout rate inside the condition encoder.")

    # DiT flow-matching loss
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"])
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--precondition_outputs", type=int, default=1, help="Whether to precondition model outputs in DiT flow matching.")

    # checkpointing
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save a resumable checkpoint every X updates.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="Max number of checkpoints to store.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help='A checkpoint-* folder, or "latest" to resume from the most recent one in the run directory.')
    parser.add_argument("--no_save", action="store_true", help="Do not save any model checkpoints.")

    # validation
    parser.add_argument("--validation_steps", type=int, default=500, help="Run validation every X steps.")
    parser.add_argument("--validation_epochs", type=int, default=None, help="Run validation every X epochs (disables validation_steps).")
    parser.add_argument("--validation_batch_size", type=int, default=None, help="Batch size for validation generation.")
    parser.add_argument("--validation_scheduler", type=str, default="EulerDiscreteScheduler",
                        choices=["DPMSolverMultistepScheduler", "EulerDiscreteScheduler", "DDPMScheduler"],
                        help="Sampler used for validation of unet runs (DiT always uses FlowMatchEuler).")
    parser.add_argument("--val_num_samples", type=int, default=5, help="Number of images generated per validation prompt.")
    parser.add_argument("--test_accuracy", action="store_true", help="Score validation images with the pretrained classifiers.")
    parser.add_argument("--image_save", action="store_true", help="Save validation images to <run>/validation_images.")
    parser.add_argument("--no_image", action="store_true", help="Do not log validation images to the tracker.")
    parser.add_argument("--early_stopping", action="store_true",
                        help="Stop once the best validation accuracy has reached --early_stopping_threshold "
                             "and has not improved for --early_stopping_patience_steps validations (or _epochs).")
    parser.add_argument("--early_stopping_threshold", type=float, default=None)
    parser.add_argument("--early_stopping_patience_steps", type=int, default=30)
    parser.add_argument("--early_stopping_patience_epochs", type=int, default=3)

    # logging
    parser.add_argument("--report_to", type=str, default="wandb", help='"tensorboard", "wandb", "comet_ml" or "all".')
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--tracker_project_name", type=str, default="mosaic")
    parser.add_argument("--tracker_run_name", type=str, default=None, help="Run name for the tracker (defaults to <test_mode>_<param_string>).")

    return parser.parse_args()
