import cv2
import os

def center_crop(img, dim):
  """Returns center cropped image

  Args:
  img: image to be center cropped
  dim: dimensions (width, height) to be cropped from center
  """

  width, height = img.shape[1], img.shape[0]
  #process crop width and height for max available dimension
  crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
  crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0]

  mid_x, mid_y = int(width/2), int(height/2)
  cw2, ch2 = int(crop_width/2), int(crop_height/2)
  crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
  return crop_img

if __name__ == "__main__":
    input_folder = "/mnt/新增磁碟區/RGB_IR資料集/VisDrone_crop/train/IR"
    output_folder = "/mnt/新增磁碟區/RGB_IR資料集/VisDrone_crop/train/IR_512/"

    for filename in os.listdir(input_folder):
        print("processing: {}".format(filename))
        img = cv2.imread(os.path.join(input_folder, filename))
        # resized_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
        resized_img = center_crop(img, (512, 512))

        cv2.imwrite(output_folder + filename, resized_img)