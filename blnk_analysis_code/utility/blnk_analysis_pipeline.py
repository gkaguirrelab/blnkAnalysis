# Import public libraries libraries 
import dill
import scipy
import os
import numpy as np
import sys 
import subprocess
import shlex
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import custom libraries
sys.path.append(os.path.dirname(__file__))
import extract_eye_features
import video_io

"""Predict eye features (pupil and eyelid) for a given BLNK video"""
def predict_eye_features(filepath: str,
                         output_folder_path: str, 
                         crop_box: tuple[int]=(140, 275, 190, 425), 
                         target_size: tuple[int]=(480, 640),   
                         temp_dir_path: str="./BLNK_temp", 
                         threshold_value: int=225,
                         visualize_results: bool=False,
                         overrwrite_existing: bool=False
                        ) -> None:

    # Define the portion of the video to crop out 
    t, b, l, r = crop_box

    # Extract the name of this video
    video_name: str = os.path.splitext(os.path.basename(filepath.rstrip('/')))[0]

    # Find the FPS of the video 
    video_fps: float = video_io.inspect_video_FPS(filepath)

    # Videos are small, so we can load them entirely in from memory 
    video_as_arr: np.ndarray = video_io.destruct_video(filepath, is_grayscale=True)

    # Note: A given BLNK video is composed of 2 independent 
    # videos of L and R eyes joined at the middle. They are 
    # the same size originally image wise, so first, we must split them 
    width_midpoint: int = video_as_arr.shape[2] // 2
    L_video: np.ndarary = video_as_arr[:, :, :width_midpoint]
    R_video: np.ndarray = video_as_arr[:, :, width_midpoint:]
    assert L_video.shape == R_video.shape, f"Shape of L video: {L_video.shape} not equal to R video: {R_video.shape}"

    for eye_label, eye_video in zip("LR", (L_video, R_video)):
      # Define the eye video name (replacing dual with what eye this is)
      assert "dual" in video_name, f"'dual' not in video name: {video_name}"
      eye_video_name: str = video_name.replace("dual", eye_label)  

      visualization_output_path: str = os.path.join(output_folder_path, f"{eye_video_name}_eye_features_visualized.avi")
      output_path: str = os.path.join(output_folder_path, f"{eye_video_name}_eyeFeatures.mat")
      if(os.path.exists(output_path) and overrwrite_existing is False):
          continue 

      # Crop out only the eye from the video
      # and set the rest of the frame to black 
      eye_video_pixel_modified: np.ndarray = eye_video[:, t:b, l:r].copy()
      white_pixels: np.ndarray = np.mean(eye_video_pixel_modified, axis=(0,)) > threshold_value
      eye_video_pixel_modified[:, white_pixels] = 0

      # Resize the video to not a small resolution 
      y_offset = (target_size[0] - eye_video_pixel_modified.shape[1]) // 2
      x_offset = (target_size[1] - eye_video_pixel_modified.shape[2]) // 2

      video_resized: np.ndarray = np.zeros((len(eye_video), *target_size), dtype=np.uint8)
      video_resized[:, y_offset:y_offset + eye_video_pixel_modified.shape[1], x_offset:x_offset + eye_video_pixel_modified.shape[2]] = eye_video_pixel_modified

      # Display the transfomred image if desired
         # If we want to visualize, we will display what the thresholding and cropping looks like 
      fig, axes = None, None
      if(visualize_results is True):
          fig, axes = plt.subplots(1, 3, figsize=(16, 14))
          axes = axes.flatten()

          axes[0].set_title(f"Original image")
          axes[0].imshow(video_as_arr[0], cmap='gray')

          axes[1].set_title(f"{eye_label} | Before Transformations")
          axes[1].imshow(eye_video[0], cmap='gray')
          crop_box_rect: patches.Rectangle = patches.Rectangle((l, t), r-l, b-t, linewidth=2, edgecolor='red', facecolor='none', fill=False)
          axes[1].add_patch(crop_box_rect)

          axes[2].set_title(f"{eye_label} | After Pixel Brightness + Spatial Transformation")
          axes[2].imshow(video_resized[0], cmap='gray')

          # Show the figure
          plt.show()

      # Generate a temp video from this cropped video 
      if(not os.path.exists(temp_dir_path)):
          os.mkdir(temp_dir_path)

      temp_video_path: str = os.path.join(temp_dir_path, f"temp_{eye_video_name}.avi")
      video_io.frames_to_video(video_resized, temp_video_path, video_fps)

      # Extract the eye features for this video
      eye_features, perimeter_info_dict = extract_eye_features.extract_eye_features(temp_video_path, 
                                                                                    is_grayscale=True, 
                                                                                    visualize_results=visualize_results,
                                                                                    visualization_output_filepath=visualization_output_path,
                                                                                    pupil_feature_method='pylids', 
                                                                                    safe_execution=True
                                                                                   )

                                                                                   
      # Repackage the features along with their metadata
      eye_features_dict: dict = {}
      eye_features_dict["data"] = eye_features
      eye_features_dict["metadata"] = {'threshold_value': {'v': threshold_value, 
                                                      'desc': "binary mask constructed from avg cropped frame. Pixels above this value=0. Done to remove light around the eye"
                                                        },
                                      'crop_box': {'v': crop_box, 
                                                  'desc': "box cropped out from original video to isolate the eye (t, b, l, r)"
                                                  },
                                      'target_size': {'v': target_size,
                                                      'desc': "target size after crop + threshold. Eye video is centered via padding to reach this size"},
                                      'model_names': {'v': ('pytorch-pupil-v1', 'pytorch-eyelid-v1'), 
                                                      'desc': "models used for pupil/eyelid fitting"
                                                    }
                                    }
      
      # Output the features
      scipy.io.savemat(os.path.join(output_folder_path, output_path), 
                       {"eyeFeatures": eye_features_dict}
                      )
      
      # Remvove the temp avi video 
      os.remove(temp_video_path)

    return 

def main():
    pass 

if(__name__ == "__main__"):
    main()