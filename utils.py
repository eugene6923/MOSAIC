import torch
import torch.nn.functional as F
from torchvision import transforms as T


# ---------------------------------------------------------------------------
# test_mode parsing
#
# A test_mode looks like "<task>[_complex|_composition<N>]_<size>", e.g.
#   count_100000, position_complex_10000, attribute_composition1_10000
# NOTE: "composition" contains the substring "position", so never test
# `"position" in test_mode` directly; use task_of() instead.
# ---------------------------------------------------------------------------
TASKS = ("count", "position", "attribute")


def task_of(test_mode):
    """The task is the leading token of the test_mode ("count_composition1_10000" -> "count").
    Substring tests would misfire because "composition" contains "position"."""
    head = test_mode.split("_")[0]
    if head in TASKS:
        return head
    raise ValueError(f"Unknown test mode {test_mode!r}; it must start with one of {TASKS}.")


def is_composition(test_mode):
    return "composition" in test_mode


def is_complex(test_mode):
    return "complex" in test_mode


def uses_two_token_condition(test_mode):
    """attribute (color1, color2) and composition (color, position) runs feed two one-hot tokens."""
    return task_of(test_mode) == "attribute" or is_composition(test_mode)


def uses_secondary_classifier(test_mode):
    """attribute runs (except complex) also score the shape; position/count-composition runs also score the colour."""
    if task_of(test_mode) == "attribute":
        return not is_complex(test_mode)
    return is_composition(test_mode)


def best_model_metric(test_mode):
    """Metric used to pick best_model: joint accuracy for attribute-composition, primary accuracy otherwise."""
    if task_of(test_mode) == "attribute" and is_composition(test_mode) and not is_complex(test_mode):
        return "joint_accuracy"
    return "accuracy"


# ---------------------------------------------------------------------------
# pretrained classifiers used to score generated images
# ---------------------------------------------------------------------------
CLASSIFIER_DIR = "./classifier_weights"


def load_classifiers(test_mode, device, resolution=128):
    """Returns (primary, secondary) classifiers for a test_mode; secondary is None when unused.

    primary:   count (20 classes, idx = count-1), position (10, idx = K-1), attribute (100, idx = c1*10+c2)
    secondary: colour (10 classes) for composition runs, shape (6 classes, 1 = ball) for attribute runs
    """
    from model import Classifier, PretrainedClassifier  # local import: model.py pulls in torchvision models

    if resolution != 128:
        raise ValueError("Classifier weights only exist for resolution 128.")

    def load(model, name, **kw):
        model.load_state_dict(torch.load(f"{CLASSIFIER_DIR}/{name}", map_location="cpu", **kw))
        return model.to(device).eval().requires_grad_(False)

    task = task_of(test_mode)
    secondary = None
    if task == "attribute":
        primary = load(PretrainedClassifier(num_classes=100), "best_classifier_attribute_pretrained.pth", weights_only=False)
        if not is_complex(test_mode):
            secondary = load(PretrainedClassifier(num_classes=6), "best_classifier_attribute_shape_pretrained.pth", weights_only=False)
    elif task == "position":
        if is_complex(test_mode):
            primary = load(Classifier(num_classes=10), "best_classifier_position_complex.pth")
        else:
            primary = load(PretrainedClassifier(num_classes=10), "best_classifier_position_pretrained.pth")
        if is_composition(test_mode):
            secondary = load(PretrainedClassifier(num_classes=10), "best_classifier_position_color_pretrained.pth")
    else:
        primary = load(Classifier(num_classes=20), "best_classifier_count.pth")
        if is_composition(test_mode):
            secondary = load(Classifier(num_classes=10), "best_classifier_count_color.pth")
    return primary, secondary


# ---------------------------------------------------------------------------
# validation accuracy
# ---------------------------------------------------------------------------
def _primary_targets(test_mode, test_labels, train_dataset, device):
    task = task_of(test_mode)
    if task == "attribute":
        # label "COLOR1_COLOR2" -> index into the 10x10 classifier output
        values = [train_dataset.classifier_color_cond_dict[(l.split("_")[0], l.split("_")[1])] for l in test_labels]
    elif task == "position":
        # label "positionK" or "COLOR_positionK" -> K-1
        values = [int(l.split("_")[-1].replace("position", "")) - 1 for l in test_labels]
    else:
        # label K (int) or "COLOR_K" -> K-1
        values = [int(str(l).split("_")[-1]) - 1 for l in test_labels]
    return torch.tensor(values, device=device)


def test_accuracy(images, validation_key, args, train_dataset, accelerator, Tester, Tester2=None):
    """Score generated images with the pretrained classifiers.

    Returns a dict with 'accuracy', 'confidence', 'distance' and, when Tester2 is given,
    'shape_accuracy'/'confidence_shape' (attribute) or 'color_accuracy'/'confidence_color'
    (position-composition) plus 'joint_accuracy'.
    """
    test_images = []
    for image in images:
        img = image.convert("RGB")
        img = T.Resize((args.resolution, args.resolution))(img)
        test_images.append(T.ToTensor()(img))
    test_labels = list(validation_key[: len(images)])

    results = {}
    task = task_of(args.test_mode)

    with torch.no_grad():
        test_images_batch = torch.stack(test_images).to(accelerator.device)
        Tester.eval()
        output = Tester(test_images_batch)
        predicted_labels = torch.argmax(output, dim=1)
        adjusted_actual = _primary_targets(args.test_mode, test_labels, train_dataset, accelerator.device)

        if Tester2 is not None:
            Tester2.eval()
            output2 = Tester2(test_images_batch)
            predicted2 = torch.argmax(output2, dim=1)
            if task == "attribute":
                # shape classifier: class 1 is the ball
                actual2 = torch.ones_like(predicted2)
                results["shape_accuracy"] = (predicted2 == actual2).float().mean()
                results["confidence_shape"] = F.softmax(output2, dim=1).max(dim=1).values.mean()
            else:
                # colour classifier for position/count-composition, label "COLOR_positionK" / "COLOR_K"
                color_names = list(train_dataset.color_dict.keys())
                actual2 = torch.tensor([color_names.index(l.split("_")[0]) for l in test_labels], device=accelerator.device)
                results["color_accuracy"] = (predicted2 == actual2).float().mean()
                results["confidence_color"] = F.softmax(output2, dim=1).max(dim=1).values.mean()
            results["joint_accuracy"] = ((predicted_labels == adjusted_actual) & (predicted2 == actual2)).float().mean()

        results["accuracy"] = (predicted_labels == adjusted_actual).float().mean()
        results["confidence"] = F.softmax(output, dim=1).max(dim=1).values.mean()

        if task == "position":
            distance_raw = torch.abs(predicted_labels - adjusted_actual)
            results["distance"] = torch.minimum(distance_raw, 10 - distance_raw).float().mean().item()
        else:
            results["distance"] = torch.norm((predicted_labels - adjusted_actual).float(), p=1).item()

    return results


def log_accuracy_results(results, args, accelerator, global_step):
    """Log the metrics from test_accuracy to the tracker and print them."""
    if not accelerator.is_main_process:
        return

    task = task_of(args.test_mode)
    name = {"count": "Count", "position": "Position", "attribute": "Attribute"}[task]

    log_dict = {
        f"validation_accuracy ({name})": results["accuracy"].item(),
        f"validation_confidence ({name})": results["confidence"].item(),
    }
    if task != "attribute":
        log_dict[f"validation_distance ({name})"] = results["distance"]
    print(f"Validation Accuracy ({name}): {results['accuracy'].item() * 100:.2f}%")

    if "shape_accuracy" in results:
        log_dict["validation_accuracy (Shape)"] = results["shape_accuracy"].item()
        log_dict["validation_confidence (Shape)"] = results["confidence_shape"].item()
        print(f"Validation Accuracy (Shape): {results['shape_accuracy'].item() * 100:.2f}%")
    if "color_accuracy" in results:
        log_dict["validation_accuracy (Color)"] = results["color_accuracy"].item()
        log_dict["validation_confidence (Color)"] = results["confidence_color"].item()
        print(f"Validation Accuracy (Color): {results['color_accuracy'].item() * 100:.2f}%")
    if "joint_accuracy" in results:
        log_dict["joint_accuracy"] = results["joint_accuracy"].item()
        print(f"Validation Joint Accuracy: {results['joint_accuracy'].item() * 100:.2f}%")

    accelerator.log(log_dict, step=global_step + 1)
    return results
