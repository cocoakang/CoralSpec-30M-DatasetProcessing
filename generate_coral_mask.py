import numpy as np
import os
import shutil
import json
from coral_obj_classifier_net import Boost_Classifier
from utils import get_bounding_box,remove_reflected_light_area
import re
import cv2
import torch
import tqdm
import argparse

from torch.utils.tensorboard import SummaryWriter 

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

from transformers import SamProcessor
from transformers import SamModel

from skimage import morphology

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate coral masks for spectral dataset using classifier and SAM.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset root directory")
    args = parser.parse_args()

    PUB_DATA_ROOT=args.data_path

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim=16
    num_classes=3

    spectrum_min = 400.0
    spectrum_max = 700.0
    coral_color = (np.array([0.0, 1.0, 0.0])*255.0).astype(np.uint8)  # Cyan color for coral

    dataset_wavelengths = np.fromfile(PUB_DATA_ROOT+"entry_0000/raw_data/wavelengths.bin",dtype=np.float32)
    wavelength_indices = np.where((dataset_wavelengths>=spectrum_min) & (dataset_wavelengths<=spectrum_max))[0]
    main_spectra = dataset_wavelengths[wavelength_indices]

    #---- tensorboard logger
    tensorboard_log_dir = PUB_DATA_ROOT + "tensorboard_logs/"
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir, exist_ok=True)

    for file_name in os.listdir(tensorboard_log_dir):
        file_path = os.path.join(tensorboard_log_dir, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
    tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)

    #---- build network model
    classifier = Boost_Classifier(latent_dim, num_classes,main_spectra)
    classifier.to(torch_device)

    pretrained_model_path = PUB_DATA_ROOT+"network_models/classifier.pth"
    checkpoint = torch.load(pretrained_model_path, map_location=torch_device,)

    classifier.load_state_dict(checkpoint["classifier"])
    classifier.eval()
    classifier.to(torch_device)
    for param in classifier.parameters():
        param.requires_grad = False
    print("classifier model loaded.")


    print("creating sam model...")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam = SamModel.from_pretrained("facebook/sam-vit-base")
    sam_model_path = PUB_DATA_ROOT+"network_models/sam_finetuned.pth"
    pretrained_model = torch.load(sam_model_path)
    sam.load_state_dict(pretrained_model["model"])
    sam.eval()
    sam.to(torch_device)
    # sam_origin = SamModel.from_pretrained("facebook/sam-vit-base")
    # sam_origin.to(device=torch_device)
    print("SAM model created.")

    #---- process each data entry
    entry_folders = sorted([folder_name for folder_name in os.listdir(PUB_DATA_ROOT) if re.match(r"entry_\d{4}",folder_name)])

    for which_entry in tqdm.tqdm(range(len(entry_folders)), desc="generating masks for entries"):
        entry_folder_root = PUB_DATA_ROOT+entry_folders[which_entry]+"/"
        raw_data_path = entry_folder_root+"raw_data/"
        processed_data_path = entry_folder_root+"processed_data/"

        #--- load json meta info
        meta_info = json.load(open(entry_folder_root+"meta_data.json","r"))

        if meta_info["illumination"] != "white":
            continue

        img_height,img_width, num_spectral_channels_origin = meta_info["image_height"], meta_info["image_width"], meta_info["num_channels"]

        #--- load reflectance data
        reflectance_cube = np.fromfile(processed_data_path+"reflectance_cube.bin",dtype=np.float32).reshape((img_height*img_width,num_spectral_channels_origin))[:,wavelength_indices]
        reflectance_cube_torch = torch.from_numpy(reflectance_cube).float().to(torch_device)

        #--- load rgb image
        rgb_img = cv2.imread(processed_data_path + "reflectance_rgb.jpg")
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        grey_bar = np.ones((rgb_img.shape[0], 10, 3), dtype=np.uint8) * 128

        #--- predict by classifier
        with torch.no_grad():
            input_data = reflectance_cube_torch.reshape((1,img_height,img_width,main_spectra.shape[0])).permute(0,3,1,2)
            logits = classifier(input_data).reshape((img_height, img_width, num_classes)) # (H, W, num_classes)
            confidences = F.softmax(logits, dim=2).cpu().numpy()
            predicted_classes = np.argmax(confidences, axis=2)
            predicted_classes = predicted_classes.reshape((img_height, img_width))

        #--- save results
        class_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        class_mask[predicted_classes==0] = np.array([255, 0, 0]) 
        class_mask[predicted_classes==1] = np.array([0, 255, 0]) 
        class_mask[predicted_classes==2] = np.array([255, 255, 0]) 
        cv2.imwrite(processed_data_path+"class_mask.png", class_mask[:,:,::-1])
        with open(processed_data_path+"class_mask.bin","wb") as pf:
            predicted_classes.astype(np.uint8).tofile(pf)

        if meta_info["health_status"] == "healthy":
            coral_mask_by_spectrum = predicted_classes == 0

            #--remove reflected light area
            coral_mask_by_spectrum_filtered_tmp = remove_reflected_light_area(coral_mask_by_spectrum)

            coral_mask_by_spectrum_filtered = morphology.remove_small_objects(coral_mask_by_spectrum_filtered_tmp.astype(bool), min_size=50)
            coral_mask_by_spectrum_filtered = np.repeat(coral_mask_by_spectrum_filtered.astype(np.uint8)[:,:,None], 3, axis=2)*255
            cv2.imwrite(processed_data_path+"coral_mask_by_spectrum.png", (coral_mask_by_spectrum_filtered))

        mask_img = np.ones((img_height, img_width, 3), dtype=np.uint8)*0.5
        mask_img = np.where(coral_mask_by_spectrum_filtered[:,:,[0]], coral_color[None,None,:] , mask_img)
        alpha = 0.3
        coral_detect_by_spectrum_mask = (1 - alpha) * rgb_img + alpha * mask_img
        coral_detect_by_spectrum_mask = np.clip(coral_detect_by_spectrum_mask, 0, 255).astype(np.uint8)

        final_result = np.concatenate([
            rgb_img,grey_bar,
            class_mask,grey_bar,
            np.repeat(coral_mask_by_spectrum.astype(np.uint8)[:,:,None],3,axis=2)*255,grey_bar,
            np.repeat(coral_mask_by_spectrum_filtered_tmp.astype(np.uint8)[:,:,None],3,axis=2)*255 ,grey_bar,
            coral_mask_by_spectrum_filtered,grey_bar,
            coral_detect_by_spectrum_mask
        ],axis=1)

        cv2.imwrite(processed_data_path + "detection_log_spectrum.png", final_result[:,:,::-1])
        tb_writer.add_image(f"Entry_{which_entry:04d}/detection_log_spectrum", final_result, global_step=0, dataformats='HWC')

        #--- prepare data for sam
        is_coral_confidence_healthy = confidences[:,:,[0]]
        is_coral_confidence_sick = confidences[:,:,[1]]
        is_others_confidence = confidences[:,:,[2]]
        is_coral_mask_pred_healthy = predicted_classes == 0
        is_coral_mask_pred_sick = predicted_classes == 1
        is_others = predicted_classes == 2  # Other classes (not healthy or sick coral)
        is_coral_mask_pred = np.logical_or( is_coral_mask_pred_healthy, is_coral_mask_pred_sick)  # Healthy coral (0) or Sick coral (1)

        #---------------------------------------
        #--- generate sam masks
        #---------------------------------------
        #--step.1 create bounding box prompt            
        is_coral_mask_pred = is_coral_mask_pred.reshape((img_height, img_width, 1)).astype(np.uint8)*255  # (H, W, 1)
        is_coral_mask_pred = np.repeat(is_coral_mask_pred, 3, axis=2)  # (H, W, 3)
        for i in range(10):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            is_coral_mask_pred = cv2.morphologyEx(is_coral_mask_pred, cv2.MORPH_OPEN, kernel)
        bounding_box = get_bounding_box(is_coral_mask_pred[:,:,0], with_perturbation = False)  # (x1, y1, x2, y2)

        #--step.2 create points prompt
        selected_coords_collector = []
        selected_pixel_label_collector = []
        for confidence_map, label in [
            (is_coral_confidence_healthy, 1),  # Healthy coral
            (is_coral_confidence_sick, 0),  # Sick coral
        ]:
            pred_coral_coords = np.stack(np.where(
                confidence_map[:,:,0] > 0.98
            ), axis=1)  # (N, 2 coordinates in (y,x) format)
            selected_num = min(50, pred_coral_coords.shape[0])
            selected_coord_indices = np.random.choice(np.arange(pred_coral_coords.shape[0]), selected_num, replace=False)
            selected_coords = pred_coral_coords[selected_coord_indices]  # (N, 2) coordinates in (y,x) format
            selected_coords_collector.append(selected_coords)
            selected_pixel_label_collector.append(np.ones((selected_coords.shape[0],), dtype=np.int32) * label)  # All points are labeled as healthy coral (1) or sick
        input_point = np.concatenate(selected_coords_collector,axis=0)[:,::-1].reshape((-1,2))  # (N,1, 2) points in (x,y) format
        input_label = np.concatenate(selected_pixel_label_collector,axis=0).reshape((-1,)).astype(np.int32)  # All points are labeled as healthy coral (1)
        
        if meta_info["health_status"] == "healthy":#if it is healthy fragments
            inputs = processor(
                rgb_img, 
                input_boxes=[[bounding_box]], 
                input_points = [input_point.tolist()],
                input_labels = [input_label.tolist()],
                return_tensors="pt"
            )
        else:
            inputs = processor(
                rgb_img, 
                input_boxes=[[bounding_box]], 
                return_tensors="pt"
            )

        #--step.3 run sam prediction
        inputs = {k: v.to(torch_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = sam(**inputs, multimask_output=False)
        
        #--step.4 process sam results
        medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg_prob = cv2.resize(medsam_seg_prob, (img_width, img_height), interpolation=cv2.INTER_NEAREST)  # Resize to match the image size

        mask_sam_final = (medsam_seg_prob > 0.5).astype(np.uint8) * 255#(H, W) np.uint8
        mask_sam_final_morphology = mask_sam_final.copy()
        mask_sam_final_morphology = morphology.remove_small_objects(mask_sam_final_morphology.astype(bool), min_size=300).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_sam_final_morphology = cv2.morphologyEx(mask_sam_final_morphology, cv2.MORPH_CLOSE, kernel)
        mask_sam_final_morphology = morphology.remove_small_holes(mask_sam_final_morphology.astype(bool), area_threshold=1000).astype(np.uint8) * 255

        # Find all contours in the mask
        contours, _ = cv2.findContours(mask_sam_final_morphology, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(mask_sam_final_morphology)

        for i, cnt in enumerate(contours):
            # Create a mask for the current contour
            contour_mask = np.zeros_like(mask_sam_final_morphology)
            cv2.drawContours(contour_mask, [cnt], -1, color=255, thickness=-1)

            # Get the pixels inside the contour
            inside_mask = (contour_mask > 0)
            # Check predicted classes inside the contour
            coral_pixels = np.logical_or(is_coral_mask_pred_healthy, is_coral_mask_pred_sick)  # (H, W) boolean mask for coral pixels
            inside_mask_coral = np.logical_and(inside_mask, coral_pixels)
            num_coral = np.sum(inside_mask_coral)
            num_inside = np.sum(inside_mask)
            ratio = num_coral / num_inside if num_inside > 0 else 0

            # Keep contour if enough coral pixels inside (e.g., at least 50%)
            if ratio > 0.5:
                filtered_mask = cv2.drawContours(filtered_mask, [cnt], -1, color=255, thickness=-1)

        mask_sam_final_morphology = filtered_mask

        #--step.5 save final mask
        cv2.imwrite(processed_data_path+"coral_mask_by_sam.png", mask_sam_final_morphology)

        #show detecting process
        prompt_input = rgb_img.copy()
        cv2.rectangle(prompt_input, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (255, 0, 0), 2)
        for point, label in zip(input_point, input_label):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)  # Green for healthy coral, Red for sick coral
            cv2.circle(prompt_input, (point[0], point[1]), 5, color, -1)

        mask_img = np.ones((img_height, img_width, 3), dtype=np.uint8)*0.5
        mask_img = np.where(mask_sam_final_morphology[:,:,None], coral_color[None,None,:] ,mask_img)
        alpha = 0.3
        coral_detect_by_final_mask = (1 - alpha) * rgb_img + alpha * mask_img
        coral_detect_by_final_mask = np.clip(coral_detect_by_final_mask, 0, 255).astype(np.uint8)
        
        mask_sam_final_log = np.repeat(mask_sam_final[:,:,None],3,axis=-1)
        mask_sam_final_morphology_log = np.repeat(mask_sam_final_morphology[:,:,None],3,axis=-1)

        final_result = np.concatenate([
            rgb_img,grey_bar,
            prompt_input,grey_bar,
            mask_sam_final_log,grey_bar,
            mask_sam_final_morphology_log,grey_bar,
            coral_detect_by_final_mask
        ],axis=1)

        cv2.imwrite(processed_data_path + "detection_log_sam.png", final_result[:,:,::-1])
        tb_writer.add_image(f"Entry_{which_entry:04d}/detection_log_sam", final_result, global_step=0, dataformats='HWC')

        #---------------------------------------
        #--- final
        #---------------------------------------
        if meta_info["health_status"] == "healthy" and meta_info["size"] == "small":#if it is healthy fragments
            final_coral_mask = coral_mask_by_spectrum_filtered
        else:
            final_coral_mask = np.repeat(mask_sam_final_morphology[:,:,None],3,axis=-1)
        
        cv2.imwrite(processed_data_path+"coral_mask.png", final_coral_mask)

        mask_img = np.ones((img_height, img_width, 3), dtype=np.uint8)*0.5
        mask_img = np.where(final_coral_mask[:,:,[0]], coral_color[None,None,:] ,mask_img)
        alpha = 0.3
        coral_detect_by_final_mask = (1 - alpha) * rgb_img + alpha * mask_img
        coral_detect_by_final_mask = np.clip(coral_detect_by_final_mask, 0, 255).astype(np.uint8)
        final_result = np.concatenate([
            rgb_img,grey_bar,
            final_coral_mask,grey_bar,
            coral_detect_by_final_mask
        ],axis=1)
        cv2.imwrite(processed_data_path + "detection_log_final.png", final_result[:,:,::-1])
        tb_writer.add_image(f"Entry_{which_entry:04d}/detection_log_final", final_result, global_step=0, dataformats='HWC')
