import scipy.io
import os
import xml.etree.ElementTree as ET
def convert_pwmfd(path, path_convert): # convert original ground truth file of PWMFD to new format
  #labels = []
  kind_of_face = {'without_mask':0, 'with_mask':1, 'incorrect_mask':2}
  bboxs = []
  try:
    tree = ET.parse(path)
  except:
    return False
  root = tree.getroot()
  for child in root: 
    if child.tag == 'filename':
      file_name = child.text.split('.')[0] + '.txt'
    if child.tag == 'object':
      #labels.append(child[0].text)
      label = child[0].text 
      label_id = kind_of_face[label]
      x = float(child[1][0].text)
      y  = float(child[1][1].text)
      x_max = float(child[1][2].text)
      y_max = float(child[1][3].text)
      bboxs.append([x , y, x_max, y_max, label_id])
  try:
    with open(os.path.join(path_convert, file_name ), 'a+') as file:
      for bbox in bboxs:
        file.writelines('1 {} {} {} {} {}\n'.format(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]))
  except:
    return False
  return True
def convert_mafa(path, path_convert): # convert original ground truth file of MAFA to new format
  labels = scipy.io.loadmat(path)
  labels = labels['LabelTest'][0]
  for label in labels:
    file_name = label[0][0].split('.')[0] + '.txt'
    with open(os.path.join(path_convert, file_name), 'a+') as file:
      for gr_bbox in label[1]:
        xmin = float(gr_bbox[0])
        ymin = float(gr_bbox[1])
        xmax = float(gr_bbox[2]) + xmin
        ymax = float(gr_bbox[3]) + ymin
        type_of_face =  gr_bbox[4]
        file.writelines('1 {} {} {} {} {}\n'.format(xmin, ymin, xmax, ymax, type_of_face))
  return True
if __name__ == '__main__':
  PATH_MAFA = os.path.join(os.getcwd(), 'data', 'MAFA', 'label') # path original label
  PATH_CONVERT_MAFA = os.path.join(os.getcwd(), 'data', 'MAFA', 'label_converted') # path new label
  PATH_PWMFD = os.path.join(os.getcwd(), 'data', 'PWMFD', 'label') # path original label
  PATH_CONVERT_PWMFD = os.path.join(os.getcwd(), 'data', 'PWMFD', 'label_converted') # path new label
  for file in os.listdir(PATH_PWMFD):
    path_annot = os.path.join(PATH_PWMFD, file)
    convert_pwmfd(path_annot,PATH_CONVERT_PWMFD)
  for file in os.listdir(PATH_MAFA):
    path_annot = os.path.join(PATH_MAFA, file)    
    convert_mafa(path_annot, PATH_CONVERT_MAFA)

