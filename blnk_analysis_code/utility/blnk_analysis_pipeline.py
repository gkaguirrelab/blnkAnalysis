# Import public libraries libraries 
import dill
import scipy
import os
import numpy as np
import sys 
import subprocess
import shlex
import matplotlib.pyplot as plt

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
                         threshold_value: int=225
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

      # Crop out only the eye from the video
      # and set the rest of the frame to black 
      eye_video: np.ndarray = eye_video[:, t:b, l:r].copy()
      white_pixels: np.ndarray = np.mean(eye_video, axis=(0,))  > threshold_value
      eye_video[:, white_pixels] = 0

      # Resize the video to not a small resolution 
      y_offset = (target_size[0] - eye_video.shape[1]) // 2
      x_offset = (target_size[1] - eye_video.shape[2]) // 2

      video_resized: np.ndarray = np.zeros((len(eye_video), *target_size), dtype=np.uint8)
      video_resized[:, y_offset:y_offset + eye_video.shape[1], x_offset:x_offset + eye_video.shape[2]] = eye_video

      # Generate a temp video from this cropped video 
      if(not os.path.exists(temp_dir_path)):
          os.mkdir(temp_dir_path)

      temp_video_path: str = os.path.join(temp_dir_path, f"temp_{eye_video_name}.avi")
      video_io.frames_to_video(video_resized, temp_video_path, video_fps)

      # Extract the eye features for this video
      eye_features: list[dict] = extract_eye_features.extract_eye_features(temp_video_path, 
                                                                           is_grayscale=True, 
                                                                           visualize_results=False, 
                                                                           pupil_feature_method='pylids', 
                                                                           safe_execution=True
                                                                          )

      # Repackage the features along with their metadata
      eye_features_dict: dict = {}
      eye_features_dict["eye_features"] = eye_features
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
      scipy.io.savemat(os.path.join(output_folder_path, f"{eye_video_name}_eye_features.mat"), 
                       {"eye_features": eye_features_dict}
                      )
      
      # Remvove the temp avi video 
      os.remove(temp_video_path)

    return 

def main():
    pass 

if(__name__ == "__main__"):
    main()