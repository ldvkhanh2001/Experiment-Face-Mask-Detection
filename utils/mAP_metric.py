import os
import shutil
class mAP_score():
  def voc_ap(self,rec, prec):
      rec.insert(0, 0.0) # insert 0.0 at begining of list
      rec.append(1.0) # insert 1.0 at end of list
      mrec = rec[:]
      prec.insert(0, 0.0) # insert 0.0 at begining of list
      prec.append(0.0) # insert 0.0 at end of list
      mpre = prec[:]

      for i in range(len(mpre)-2, -1, -1):
          mpre[i] = max(mpre[i], mpre[i+1])
     
      i_list = []
      for i in range(1, len(mrec)):
          if mrec[i] != mrec[i-1]:
              i_list.append(i) # if it was matlab would be i + 1
     
      ap = 0.0
      for i in i_list:
          ap += ((mrec[i]-mrec[i-1])*mpre[i])
      return ap, mrec, mpre  

  def evaluate(self,pred,target,  dataset, model, default_iou=0.5, specified_iou=None,class_ignored=None,class_map=None):
    #pred: python dictionary has key which is file_id, value which is np.array(no_boxoes,6)
    #target : python dictionary has  key which is  file_id, value which is np.array(no_boxes,5)
    #default_iou: default is 0.5
    #specified_iou python dictionary has key is a class_id and value is corresponding iou
    #class_ignored: list class_id should be ignored
    #class_map: python dictionary has key is a class_id, value is name of  class
    #bounding box format mặc định : xmin, ymin, xmax, ymax absolute value
    if class_ignored is None:
      class_ignored = []

    specific_iou_flagged=False
    if specified_iou is not None:
      specific_iou_flagged=True
      specific_iou_classes= list(specified_iou.keys())

    gt_counter_per_class = {}
    counter_images_per_class = {}
    gt_files = {}
    if dataset == 'MAFA':
        kind_of_face = {'1':0, '2':0, '3':0}
    else:
        kind_of_face = {'0':0, '1':0, '2':0}
    for file_id in target:
      bounding_boxes = []
      already_seen_classes = []
      
      for line in range(0,target[file_id].shape[0]):
        class_id, left, top, right, bottom, kind = target[file_id][line]
        if class_id in class_ignored:
          continue
        bbox = str(left) + " " + str(top) + " " + str(right) + " " +str(bottom)
        bounding_boxes.append({"class_id":class_id, "bbox":bbox, "used":False, 'kind':str(int(kind))})
        if class_id in gt_counter_per_class:
            gt_counter_per_class[class_id] += 1
        else:
            gt_counter_per_class[class_id] = 1
        if class_id not in already_seen_classes:
            if class_id in counter_images_per_class:
                counter_images_per_class[class_id] += 1
            else:
                # if class didn't exist yet
                counter_images_per_class[class_id] = 1
            already_seen_classes.append(class_id)        
      gt_files[file_id] = bounding_boxes

    

    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)
    #print(gt_classes)
    n_classes = len(gt_classes)
    
    dr_files = {}
    for class_index, class_id in enumerate(gt_classes):
        bounding_boxes = []
 
        for file_id in pred:

            
            for line in range(0,pred[file_id].shape[0]):
                
                    tmp_class_id, confidence, left, top, right, bottom = pred[file_id][line][0],pred[file_id][line][1],pred[file_id][line][2],pred[file_id][line][3],pred[file_id][line][4],pred[file_id][line][5]
                
                    if tmp_class_id == class_id:
                        
                        bbox = str(left) + " " + str(top) + " " + str(right) + " " +str(bottom)
                        bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})
                        
        
        bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
        dr_files[class_id] = bounding_boxes
    #print(dr_files)
    sum_AP = 0.0
    ap_dictionary = {}    

    output_file = ""

    output_file+="# AP and precision/recall per class\n"
    count_true_positives = {}
    for class_index, class_id in enumerate(gt_classes):
        count_true_positives[class_id] = 0
        
        
        dr_data = dr_files[class_id]

        nd = len(dr_data)
        tp = [0] * nd # creates an array of zeros of size nd
        fp = [0] * nd
        for idx, detection in enumerate(dr_data):
            file_id = detection["file_id"]
            
            
            
            ground_truth_data = gt_files[file_id]
            
            ovmax = -1
            gt_match = -1
            # load detected object bounding-box
            bb = [ float(x) for x in detection["bbox"].split() ]

            #print(ground_truth_data)
            for obj in ground_truth_data:
                # look for a class_name match
                #print('kind of face', type(kind))
                if obj["class_id"] == class_id:
                    # print(bb)
                    # print(obj)
                    # return 99999999
                    bbgt = [ float(x) for x in obj["bbox"].split() ]
                    bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                        + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        
                        if ov > ovmax:
                            kind = obj['kind']
                            ovmax = ov
                            gt_match = obj

            
            
            # set minimum overlap
            min_overlap = default_iou
            if specific_iou_flagged:
                if class_id in specific_iou_classes:
                    
                    min_overlap = float(specified_iou[class_id])
            if ovmax >= min_overlap:
                        if not bool(gt_match["used"]):
                            # true positive
                            tp[idx] = 1
                            gt_match["used"] = True
                            count_true_positives[class_id] += 1
                            kind_of_face[kind] +=1
                            
                            
                        else:
                            # false positive (multiple detection)
                            fp[idx] = 1

            else:
                # false positive
                fp[idx] = 1
                
                    


        #print(tp)
        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        #print(tp)
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_id]
        #print(rec)
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        #print(prec)

        ap, mrec, mprec = self.voc_ap(rec[:], prec[:])
        sum_AP += ap
        text = "{0:.2f}%".format(ap*100) + " = " + str(class_map[class_id]) + " AP " #class_id + " AP = {0:.2f}%".format(ap*100)
        """
          Write to output.txt
        """
        rounded_pre = [ round(elem,2) for elem in prec ]
        rounded_rec = [ round(elem,2) for elem in rec ]
        output_file+=text + "\n Precision: " + str(rounded_pre) + "\n Recall :" + str(rounded_rec) + "\n\n"
       
        ap_dictionary[class_id] = ap

        n_images = counter_images_per_class[class_id]
      

    output_file+="\n# mAP of all classes\n"
    mAP = sum_AP / n_classes
    text = "mAP = {0:.2f}%".format(mAP*100)
    output_file+=text + "\n"
    


    det_counter_per_class = {}
    for file_id in pred:
        # get lines to list
        
        for line in range(0,pred[file_id].shape[0]):
            class_id = pred[file_id][line][0]
            # check if class is in the ignore list, if yes skip
            if class_id in class_ignored:
                continue
            # count that object
            if class_id in det_counter_per_class:
                det_counter_per_class[class_id] += 1
            else:
                # if class didn't exist yet
                det_counter_per_class[class_id] = 1
    
    dr_classes = list(det_counter_per_class.keys())
    output_file+="\n# Number of ground-truth objects per class\n"
    for class_id in sorted(gt_counter_per_class):
      output_file+=str(class_map[class_id]) + ": " + str(gt_counter_per_class[class_id]) + "\n"
    for class_id in dr_classes:
        # if class exists in detection-result but not in ground-truth then there are no true positives in that class
        if class_id not in gt_classes:
            count_true_positives[class_id] = 0

    output_file+="\n# Number of detected objects per class\n"
    for class_id in sorted(dr_classes):
        n_det = det_counter_per_class[class_id]
        text = str(class_map[class_id]) + ": " + str(n_det)
        text += " (tp:" + str(count_true_positives[class_id]) + ""
        text += ", fp:" + str(n_det - count_true_positives[class_id]) + ")\n"
        output_file+=text
        output_file+= str(kind_of_face)
    path_output = os.path.join(os.getcwd(), 'result', dataset, model, 'output')
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    else:
        shutil.rmtree(path_output)
        os.makedirs(path_output)
    with open(os.path.join(path_output,'output.txt'), 'w+') as file:
        file.writelines(output_file)
    return rounded_pre, rounded_rec, mAP








  

  