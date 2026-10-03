import torch
import torch.nn.functional as F
from torchvision import transforms as T

import numpy as np

def test_accuracy(images, validation_key, args, train_dataset, accelerator, Tester, Tester2=None):
    """
    Test accuracy of generated images using pre-trained classifiers.
    
    Args:
        images: List of PIL images to test
        validation_key: List of ground truth labels
        args: Training arguments
        train_dataset: Training dataset (for color_dict access)
        accelerator: Accelerator object
        Tester: Primary classifier model
        Tester2: Secondary classifier model (for OOD tasks)
    
    Returns:
        Dictionary containing accuracy metrics
    """
    test_images = []
    test_labels = []
    
    # Prepare images and labels
    for idx, image in enumerate(images):
        img = image.convert("RGB")
        img = T.Resize((args.resolution, args.resolution))(img)
        img = T.ToTensor()(img)
        test_images.append(img)
        label = validation_key[idx]
        test_labels.append(label)
    
    results = {}
    
    with torch.no_grad():
        test_images_batch = torch.stack(test_images).to(accelerator.device)
        Tester.eval()
        
        # Run primary classifier inference
        output = Tester(test_images_batch)
        predicted_labels = torch.argmax(output, dim=1)
        
        # Calculate adjusted actual labels based on test mode
        if "ood" not in args.test_mode:
            if "position" not in args.test_mode and "attribute" not in args.test_mode:
                adjusted_actual = torch.tensor(
                    [l-1 if l != 0 else 0 for l in test_labels], 
                    device=accelerator.device
                )
            
            elif "attribute" in args.test_mode:
                adjusted_actual = torch.tensor(
                    [train_dataset.classifier_color_cond_dict[(l.split("_")[0]), (l.split("_")[1])] for l in test_labels], 
                    device=accelerator.device
                )

            else:
                adjusted_actual = torch.tensor(
                    [int(l.replace("position",""))-1 for l in test_labels], 
                    device=accelerator.device
                )
            
            if Tester2 is not None:
                Tester2.eval()
                output_shape = Tester2(test_images_batch)
                predicted_shapes = torch.argmax(output_shape, dim=1)

                adjusted_actual_shape = torch.tensor(
                    [1 for l in test_labels], 
                    device=accelerator.device
                )

                results['shape_accuracy'] = (predicted_shapes == adjusted_actual_shape).float().mean()
                results['confidence_shape'] = F.softmax(output_shape, dim=1).max(dim=1).values.mean()
                results['joint_accuracy'] = ((predicted_labels == adjusted_actual) & (predicted_shapes == adjusted_actual_shape)).float().mean()

        else:
            if "attribute" in args.test_mode:
                adjusted_actual = torch.tensor(
                    [train_dataset.classifier_color_cond_dict[(l.split("_")[0]), (l.split("_")[1])] for l in test_labels], 
                    device=accelerator.device
                )
            elif "position" in args.test_mode:
                adjusted_actual = torch.tensor(
                    [int(l.split("_")[1].replace("position",""))-1 for l in test_labels], 
                    device=accelerator.device
                )
            elif "object" in args.test_mode:
                adjusted_actual = torch.tensor(
                    [train_dataset.classifier_object_cond_dict[train_dataset.parse_objects_from_filename(l)] for l in test_labels], 
                    device=accelerator.device
                )
            else:
                adjusted_actual = torch.tensor(
                    [int(l.split("_")[-1])-1 for l in test_labels], 
                    device=accelerator.device
                )
            
            # Run secondary classifier for OOD tasks
            if Tester2 is not None:
                Tester2.eval()
                
                if "attribute" not in args.test_mode:
                    output_color = Tester2(test_images_batch)
                    predicted_colors = torch.argmax(output_color, dim=1)
                    
                    adjusted_actual_color = torch.tensor(
                        [list(train_dataset.color_dict.keys()).index(l.split("_")[0]) for l in test_labels], 
                        device=accelerator.device
                    )
                    
                    results['color_accuracy'] = (predicted_colors == adjusted_actual_color).float().mean()
                    results['confidence_color'] = F.softmax(output_color, dim=1).max(dim=1).values.mean()
                    results['joint_accuracy'] = ((predicted_labels == adjusted_actual) & (predicted_colors == adjusted_actual_color)).float().mean()
                
                else:
                    output_shape = Tester2(test_images_batch)
                    predicted_shapes = torch.argmax(output_shape, dim=1)
                    adjusted_actual_shape = torch.tensor(
                        [1 for l in test_labels], 
                        device=accelerator.device
                    )

                    results['shape_accuracy'] = (predicted_shapes == adjusted_actual_shape).float().mean()
                    results['confidence_shape'] = F.softmax(output_shape, dim=1).max(dim=1).values.mean()
                    results['joint_accuracy'] = ((predicted_labels == adjusted_actual) & (predicted_shapes == adjusted_actual_shape)).float().mean()

        # Calculate primary metrics
        results['accuracy'] = (predicted_labels == adjusted_actual).float().mean()
        results['confidence'] = F.softmax(output, dim=1).max(dim=1).values.mean()
        
        # Calculate distance metric
        if "position" in args.test_mode:
            distance_raw = torch.abs(predicted_labels - adjusted_actual)
            distance_circular = 10 - distance_raw
            results['distance'] = torch.minimum(distance_raw, distance_circular).float().mean().item()
        else:
            results['distance'] = torch.norm((predicted_labels - adjusted_actual).float(), p=1).item()
        
    
    # Clean up tensors
    del test_images_batch, test_images, test_labels
    if 'output' in locals():
        del output
    if 'output_color' in locals():
        del output_color
    if 'predicted_labels' in locals():
        del predicted_labels
    if 'predicted_colors' in locals():
        del predicted_colors
    if 'adjusted_actual' in locals():
        del adjusted_actual
    if 'adjusted_actual_color' in locals():
        del adjusted_actual_color
    
    return results

def log_accuracy_results(results, args, accelerator, global_step):
    """
    Log accuracy results to tracking system and print to console.
    
    Args:
        results: Dictionary of accuracy metrics from test_accuracy function
        args: Training arguments
        accelerator: Accelerator object
        global_step: Current training step
    """
    if not accelerator.is_main_process:
        return
    
    log_dict = {}
    
    if "position" in args.test_mode:
        if "ood" not in args.test_mode:
            log_dict.update({
                "validation_accuracy (Position)": results['accuracy'].item(),
                "validation_distance (Position)": results['distance'],
                "validation_confidence (Position)": results['confidence'].item()
            })
            print("Validation Accuracy (Position): {:.2f}%".format(results['accuracy'].item() * 100))
        else:
            log_dict.update({
                "validation_accuracy (Position)": results['accuracy'].item(),
                "validation_distance (Position)": results['distance'],
                "validation_accuracy (Color)": results['color_accuracy'].item(),
                "joint_accuracy": results['joint_accuracy'].item(),
                "validation_confidence (Color)": results['confidence_color'].item(),
                "validation_confidence (Position)": results['confidence'].item()
            })
            print("Validation Accuracy (Position): {:.2f}%".format(results['accuracy'].item() * 100))
            print("Validation Accuracy (Color): {:.2f}%".format(results['color_accuracy'].item() * 100))
            print("Validation Joint Accuracy: {:.2f}%".format(results['joint_accuracy'].item() * 100))
    

    elif "attribute" in args.test_mode:
        if "multi" in args.test_mode or "disturber" in args.test_mode:
            log_dict.update({
                "validation_accuracy (Attribute)": results['accuracy'].item(),
                "validation_confidence (Attribute)": results['confidence'].item()
            })
            print("Validation Accuracy (Attribute): {:.2f}%".format(results['accuracy'].item() * 100))

        else:
            log_dict.update({
                "validation_accuracy (Attribute)": results['accuracy'].item(),
                "validation_accuracy (Shape)": results['shape_accuracy'].item(),
                "joint_accuracy": results['joint_accuracy'].item(),
                "validation_confidence (Shape)": results['confidence_shape'].item(),
                "validation_confidence (Attribute)": results['confidence'].item()
            })
            print("Validation Accuracy (Attribute): {:.2f}%".format(results['accuracy'].item() * 100))
            print("Validation Accuracy (Shape): {:.2f}%".format(results['shape_accuracy'].item() * 100))
            print("Validation Joint Accuracy: {:.2f}%".format(results['joint_accuracy'].item() * 100))
    
    else:
        if "ood" not in args.test_mode:
            log_dict.update({
                "validation_accuracy (Count)": results['accuracy'].item(),
                "validation_distance (Count)": results['distance'],
                "validation_confidence (Count)": results['confidence'].item()
            })
            print("Validation Accuracy (Count): {:.2f}%".format(results['accuracy'].item() * 100))
        else:
            if "object" in args.test_mode:
                log_dict.update({
                "validation_accuracy (Object)": results['accuracy'].item(),
                "validation_distance (Object)": results['distance'],
                "validation_confidence (Object)": results['confidence'].item()
            })
                print("Validation Accuracy (Object): {:.2f}%".format(results['accuracy'].item() * 100))
                
        
            else:
                log_dict.update({
                    "validation_accuracy (Count)": results['accuracy'].item(),
                    "validation_distance (Count)": results['distance'],
                    "validation_accuracy (Color)": results['color_accuracy'].item(),
                    "joint_accuracy": results['joint_accuracy'].item(),
                    "validation_confidence (Color)": results['confidence_color'].item(),
                    "validation_confidence (Count)": results['confidence'].item()
                })
                print("Validation Accuracy (Count): {:.2f}%".format(results['accuracy'].item() * 100))
                print("Validation Accuracy (Color): {:.2f}%".format(results['color_accuracy'].item() * 100))
                print("Validation Joint Accuracy: {:.2f}%".format(results['joint_accuracy'].item() * 100))
                
            
    
    accelerator.log(log_dict, step=global_step+1)
    
    return results
