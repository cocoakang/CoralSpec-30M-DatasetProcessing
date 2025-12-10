import numpy as np
import os 
import cv2
import tqdm

species_abbr_2_name={
    "PVER":"Pocillopra Verrucosa",
    "SPIS":"Stylophora Pistillata",
    "PNOD":"Porites Nodifera",
    "MDAN":"Montipora Danae",
    "LBOT":"Leptastrea Bottae",
    "ASQU":"Acropora Squarrosa",
    "APHA":"Acropora Pharaonis",
    "ADIG":"Acropora Digitifera",
    "ACYT":"Acropora Cytherea",
    "AARA":"Acropora Arabensis"
}

species_name_2_abbr={
    "Pocillopra Verrucosa":"PVER",
    "Stylophora Pistillata":"SPIS",
    "Porites Nodifera":"PNOD",
    "Montipora Danae":"MDAN",
    "Leptastrea Bottae":"LBOT",
    "Acropora Squarrosa":"ASQU",
    "Acropora Pharaonis":"APHA",
    "Acropora Digitifera":"ADIG",
    "Acropora Cytherea":"ACYT",
    "Acropora Arabensis":"AARA"
}

def get_species_data_entries(
        data_holder_root,species, 
    ):
    '''
        species: PEVR
    '''
    with open(data_holder_root+"dn_data_{}.txt".format(species),"r") as f:
        meta_data_description = [a.split(":")[0] for a in f.readline()[1:].strip("\n").split(",")]
        meta_data_dtype = f.readline()[1:].strip("\n").split(",")
        data_collector = []
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                continue

            tmp_meta_data = {}
            line_split = line.split(",")
            for i in range(len(meta_data_description)):
                if meta_data_dtype[i] == "int":
                    tmp_meta_data[meta_data_description[i]] = int(line_split[i])
                elif meta_data_dtype[i] == "float":
                    tmp_meta_data[meta_data_description[i]] = float(line_split[i])
                else:
                    tmp_meta_data[meta_data_description[i]] = line_split[i]
            tmp_meta_data["species"] = species
            
            data_collector.append(tmp_meta_data)
    print("Available {} data entries {}".format(species,len(data_collector)))

    return data_collector

def get_labeled_pixels(data_holder_root,species,binary=False):
    if os.path.exists(data_holder_root+"dn_data_{}_labeled_pixels.txt".format(species)) == False:
        print("No labeled pixels for {}".format(species))
        return []
    with open(data_holder_root+"dn_data_{}_labeled_pixels.txt".format(species),"r") as pf:
        pf.readline()
        pf.readline()
        label_ids = pf.readline().strip("\n")
        label_ids = label_ids[len("label_id:")+1:].split(";")
        label_collector = []
        for i in range(len(label_ids)):
            tmp_record = label_ids[i].split("-")
            tmp_label_id, tmp_label_descript = int(tmp_record[0]),tmp_record[1]

            cur_label = {}
            cur_label["label_id"] = tmp_label_id
            cur_label["label_descript"] = tmp_label_descript
            cur_label["label_indices"] = []
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap('tab10')  # Use a dedicated color map
            cur_label['color'] = np.array(cmap(i % cmap.N)[:3],np.float32)  # Assign color based on label index
            label_collector.append(cur_label)

        pf.readline()

        remaining_lines = pf.readlines()
        assert len(remaining_lines) % 2 == 0
        data_entry_num = len(remaining_lines)//2

        for i in tqdm.tqdm(range(data_entry_num)):
            data_folder,data_name = remaining_lines[i*2+0].strip("\n").split(",")

            if remaining_lines[i*2+1] == "\n":
                continue
            tmp_labels = remaining_lines[i*2+1].strip("\n").split(";")
            for j in range(len(tmp_labels)):
                tmp_label = tmp_labels[j].split(",")
                pixel_coords = np.array([i,int(tmp_label[0]),int(tmp_label[1])],dtype=np.int32)
                label_id = int(tmp_label[2])            
                label_collector[label_id]["label_indices"].append(pixel_coords)
                if label_id > 0:
                    label_collector[1]["label_indices"].append(pixel_coords)

        for i in range(len(label_collector)):
            if len(label_collector[i]["label_indices"]) == 0:
                label_collector[i]["label_indices"] = np.zeros((0,3),dtype=np.int32)
            else:
                label_collector[i]["label_indices"] = np.stack(label_collector[i]["label_indices"],axis=0)

        if binary:
            label_collector_binary = []
            label_collector_binary.append(label_collector[0])
            cur_label = {}
            cur_label["label_id"] = 1
            cur_label["label_descript"] = "others"
            cur_label["label_indices"] = []
            cur_label["color"] = label_collector[1]["color"]
            for i in range(1, len(label_collector)):
                cur_label["label_indices"].append(label_collector[i]["label_indices"])
            cur_label["label_indices"] = np.concatenate(cur_label["label_indices"])
            label_collector_binary.append(cur_label)
            label_collector = label_collector_binary
        return label_collector  

#-----------spec2rgb related
def load_cmf_tab(tab_path):
    tab_raw = np.loadtxt(tab_path,delimiter=',')
    tabs_bands = tab_raw[:,0]
    tabs_values = tab_raw[:,1:]
    
    y_values = tabs_values[:,1]
    y_values_mid = 0.5 * (y_values[1:] + y_values[:-1])
    CIE_Y_integral = np.sum(np.diff(tabs_bands) * y_values_mid)
    tab = {
        'bands': tabs_bands,
        'values': tabs_values,
        'CIE_Y_integral' : CIE_Y_integral
    }
    return tab

def spec2xyz_tab(w, spec, tab):
    '''
        input w: (wave_num,)
        spec: (batch_size, wave_num)
        tab:{
                'bands': (tab_record_num,)
                'values': (tab_record_num, 3)
            }
    '''
    tab_min_band = tab['bands'][0]
    tab_max_band = tab['bands'][-1]

    valid_w_idx = np.logical_and(w >= tab_min_band, w <= tab_max_band)
    w = w[valid_w_idx]
    spec = spec[:, valid_w_idx]

    diff_matrix = np.abs(tab['bands'][None,:] - w[:, None])#(effective_wave_num, tab_record_num)
    closest_indices = np.argmin(diff_matrix, axis=1)#(effective_wave_num,)
    response_value = tab['values'][closest_indices]#(effective_wave_num, 3)

    widths = np.diff(w)#(effective_wave_num-1,)

    xyz = spec[:,:,None] * response_value[None,:,:]#(batch_size, effective_wave_num, 3)
    xyz_middle = 0.5 * (xyz[:, 1:, :] + xyz[:, :-1, :])#(batch_size, effective_wave_num-1, 3)
    xyz = np.sum(widths[None,:,None] * xyz_middle, axis=1)#(batch_size, 3)

    xyz = xyz / tab['CIE_Y_integral']

    return xyz

def xyz2rgb(xyz):
    '''
        input xyz: (record_num, 3)
    '''
    m = np.array([[3.2404542, -1.5371385, -0.4985314],
                    [-0.9692660, 1.8760108, 0.0415560],
                    [0.0556434, -0.2040259, 1.0572252]],np.float32)
    rgb = np.matmul(m, xyz.T).T#(recordr_num, 3)
    # rgb = np.clip(rgb, 0, 1)
    return rgb
    
def linear_float_to_srgb_float(data):
    data = np.where(data <= 0.0031308, data * 12.92, 1.055 * np.power(data, 1.0/2.4) - 0.055)
    return data

#---sam
def get_bounding_box(ground_truth_map, with_perturbation=True):
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    if with_perturbation:
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    return bbox

def remove_reflected_light_area(coral_mask):
    coral_mask = coral_mask.astype(np.uint8)
    assert coral_mask.ndim == 2
    contours, _ = cv2.findContours(coral_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(coral_mask)

    for i, cnt in enumerate(contours):
        # Create a mask for the current contour
        contour_mask = np.zeros_like(coral_mask)
        cv2.drawContours(contour_mask, [cnt], -1, color=255, thickness=-1)

        attached_to_upper_area = (contour_mask[:5].reshape((-1,)) > 0).any(axis=0)

        if not attached_to_upper_area:
            filtered_mask = cv2.drawContours(filtered_mask, [cnt], -1, color=255, thickness=-1)
    
    return filtered_mask