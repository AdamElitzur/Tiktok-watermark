from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os
from moviepy.editor import VideoFileClip
import tempfile
import uuid
from dotenv import load_dotenv
import glob
import time
import threading
import atexit

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

PROCESSED_FOLDER = os.getenv("PROCESSED_FOLDER", "processed")
TEMPLATE_FOLDER = os.getenv("TEMPLATE_FOLDER", "templates")
TEMPLATE_FILE = os.path.join(TEMPLATE_FOLDER, "tiktok_watermark.png")
# Max number of videos to keep in processed folder - reduced to prevent disk space issues
MAX_VIDEOS_TO_KEEP = 2  # Keeping fewer videos to ensure cleanup works properly

# Create directories if they don't exist
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(TEMPLATE_FOLDER, exist_ok=True)

# Check if template exists, create a default one if not
if not os.path.exists(TEMPLATE_FILE):
    # Create a more realistic TikTok-like template
    template = np.zeros((163, 194, 3), dtype=np.uint8)
    # Add a semi-transparent dark background
    cv2.rectangle(template, (0, 0), (194, 163), (40, 40, 40), -1)
    # Add TikTok-like text
    cv2.putText(template, "TikTok", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 
                1, (255, 255, 255), 2)
    cv2.putText(template, "@user", (20, 100), cv2.FONT_HERSHEY_COMPLEX, 
                1, (255, 255, 255), 2)
    cv2.imwrite(TEMPLATE_FILE, template)

# Load the template once at startup for efficiency
TEMPLATE_IMAGE = cv2.imread(TEMPLATE_FILE)
if TEMPLATE_IMAGE is None:
    # Create a fallback template in memory
    TEMPLATE_IMAGE = np.zeros((163, 194, 3), dtype=np.uint8)
    cv2.rectangle(TEMPLATE_IMAGE, (0, 0), (194, 163), (40, 40, 40), -1)
    cv2.putText(TEMPLATE_IMAGE, "TikTok", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 
                1, (255, 255, 255), 2)

# Custom exceptions for better error handling
class VideoProcessingError(Exception):
    """Base exception for video processing errors"""
    pass

class WatermarkDetectionError(VideoProcessingError):
    """Exception raised for errors during watermark detection"""
    pass

class VideoCleanupError(VideoProcessingError):
    """Exception raised for errors during video cleanup"""
    pass

# TikTok watermark coordinates
TIKTOK_COORDINATES = {
    "before_4s": {"x": 0, "y": 875, "width": 194, "height": 163},  # 1038 - 875
    "after_4s": {
        "x": 884,
        "y": 1456,
        "width": 196,  # 1080 - 884
        "height": 163,  # 1619 - 1456
    },
}

def cleanup_old_videos(force=False):
    """
    Clean up old videos from the processed folder, but keep the folders.
    Only keeps the MAX_VIDEOS_TO_KEEP most recent videos.
    
    Args:
        force: If True, will clean up even if we have fewer than MAX_VIDEOS_TO_KEEP videos
        
    Raises:
        VideoCleanupError: If there is an error during cleanup
    """
    try:
        # Make sure the directory exists
        if not os.path.exists(PROCESSED_FOLDER):
            os.makedirs(PROCESSED_FOLDER, exist_ok=True)
            return
            
        # Get all mp4 files in the processed folder
        video_files = glob.glob(os.path.join(PROCESSED_FOLDER, "*.mp4"))
        
        # If we have more than MAX_VIDEOS_TO_KEEP, or if force=True, delete the oldest ones
        if len(video_files) > MAX_VIDEOS_TO_KEEP or force:
            # Sort files by modification time (oldest first)
            video_files.sort(key=os.path.getmtime)
            
            # How many files to delete
            if force and len(video_files) <= MAX_VIDEOS_TO_KEEP:
                # If forced cleanup but we have fewer than max, delete all but the most recent
                num_to_delete = max(0, len(video_files) - 1)
            else:
                # Normal case: delete all except MAX_VIDEOS_TO_KEEP most recent ones
                num_to_delete = max(0, len(video_files) - MAX_VIDEOS_TO_KEEP)
            
            # Remove oldest files, keeping only MAX_VIDEOS_TO_KEEP most recent
            for file_path in video_files[:num_to_delete]:
                try:
                    # Double check the file exists and is in the correct folder
                    if os.path.exists(file_path) and PROCESSED_FOLDER in os.path.abspath(file_path):
                        os.remove(file_path)
                except Exception as e:
                    # Continue with next file instead of stopping completely
                    pass
    except Exception as e:
        error_msg = f"Error during video cleanup: {str(e)}"
        raise VideoCleanupError(error_msg) from e

def generate_mask(frame_shape, bbox):
    """Create a binary mask for inpainting"""
    mask = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
    x, y, w, h = bbox
    mask[y : y + h, x : x + w] = 255
    return mask

def check_watermark_region(frame, x, y, width, height):
    """
    Check if a region of an image likely contains a watermark using multiple detection methods:
    1. Template matching with TikTok watermark template
    2. Analysis of variance, brightness, and edge density
    
    Args:
        frame: Video frame to check
        x, y, width, height: Region coordinates
        
    Returns:
        bool: True if watermark likely exists, False otherwise
        
    Raises:
        WatermarkDetectionError: If there is an error during watermark detection
    """
    # Make sure coordinates are valid
    frame_height, frame_width = frame.shape[:2]
    
    # Check if coordinates are within frame boundaries
    if y >= frame_height or x >= frame_width:
        return False
    
    # Adjust dimensions to fit within frame
    if y + height > frame_height:
        height = frame_height - y
    if x + width > frame_width:
        width = frame_width - x
        
    # If region is too small, it's not valid
    if height <= 10 or width <= 10:
        return False
    
    # Extract region of interest
    roi = frame[y:y+height, x:x+width]
    
    # Convert to grayscale for analysis
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Template matching with TikTok watermark template
    template_match_score = 0
    try:
        # Resize template to match ROI size if needed
        if TEMPLATE_IMAGE.shape[0] != height or TEMPLATE_IMAGE.shape[1] != width:
            template_resized = cv2.resize(TEMPLATE_IMAGE, (width, height))
        else:
            template_resized = TEMPLATE_IMAGE
            
        # Convert template to grayscale for matching
        template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)
        
        # Perform template matching
        result = cv2.matchTemplate(roi_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        template_match_score = max_val
    except Exception as e:
        error_msg = f"Template matching error: {str(e)}"
        raise WatermarkDetectionError(error_msg) from e
    
    # Method 2: Calculate statistical metrics
    
    # 1. Variance - high variance might indicate detail/content
    variance = np.var(roi_gray)
    
    # 2. Mean - if too dark or too bright, probably not a watermark
    mean = np.mean(roi_gray)
    
    # 3. Edge detection - watermarks usually have edges
    edges = cv2.Canny(roi_gray, 100, 200)
    edge_count = np.count_nonzero(edges)
    edge_density = edge_count / (width * height)
    
    # Combination of criteria for watermark detection
    has_sufficient_variance = variance > 150  # Adjusted threshold
    has_reasonable_brightness = 20 < mean < 235  # Not too dark, not too bright
    has_edges = edge_density > 0.02  # At least some edges
    template_match_good = template_match_score > 0.3  # Template match threshold
    
    # Return True if either template matching is good OR other criteria are met
    # This provides a more robust detection that works even if template is not exact
    return template_match_good or (has_sufficient_variance and has_reasonable_brightness and has_edges)


def process_video(input_path, use_inpainting=False):
    """
    Process video to detect and remove TikTok watermark.
    Checks if watermark exists before processing.
    
    Args:
        input_path: Path to input video
        use_inpainting: Whether to use inpainting (True) or blur (False)
        
    Returns:
        tuple: (Watermark detected flag, Output video path)
        
    Raises:
        VideoProcessingError: If there is an error during video processing
    """
    cap = None
    temp_output = None
    out = None
    original_video = None
    processed_video = None
    
    try:
        # Open video and get properties
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise VideoProcessingError("Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # First, check if a watermark is present
        watermark_locations = {"before_4s": False, "after_4s": False}
        
        # Check first frame for watermark at first position
        ret, first_frame = cap.read()
        if not ret:
            raise VideoProcessingError("Could not read first frame of video")
            
        # Check for watermark at first position
        coords = TIKTOK_COORDINATES["before_4s"]
        watermark_locations["before_4s"] = check_watermark_region(
            first_frame, coords["x"], coords["y"], coords["width"], coords["height"]
        )
        
        # Check for watermark at second position (move to frame at 5 seconds)
        frame_at_5s = None
        target_frame = int(5 * fps)
        if target_frame < cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame_at_5s = cap.read()
            if ret:
                coords = TIKTOK_COORDINATES["after_4s"]
                watermark_locations["after_4s"] = check_watermark_region(
                    frame_at_5s, coords["x"], coords["y"], coords["width"], coords["height"]
                )
        
        # Reset to start of video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # If no watermark is detected, return the original video
        watermark_detected = watermark_locations["before_4s"] or watermark_locations["after_4s"]
        if not watermark_detected:
            if cap:
                cap.release()
            return False, input_path
        
        # Create temporary output file
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_output, fourcc, fps, (frame_width, frame_height))

        # Process all frames based on detected watermark locations
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_count / fps
            
            # Only process frames if a watermark was detected at that position
            if current_time <= 3.47 and watermark_locations["before_4s"]:
                # First position (0-3.46 seconds)
                coords = TIKTOK_COORDINATES["before_4s"]
                x, y = coords["x"], coords["y"]
                width, height = coords["width"], coords["height"]
                
                # Apply effect (blur or inpainting)
                if use_inpainting:
                    # Create mask for inpainting
                    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                    mask[y:y+height, x:x+width] = 255
                    frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
                else:
                    # Apply blur effect
                    roi = frame[y:y+height, x:x+width]
                    blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
                    frame[y:y+height, x:x+width] = blurred_roi
                    
            elif current_time > 3.47 and watermark_locations["after_4s"]:
                # Second position (after 3.46 seconds)
                coords = TIKTOK_COORDINATES["after_4s"]
                x, y = coords["x"], coords["y"]
                width, height = coords["width"], coords["height"]
                
                # Apply effect (blur or inpainting)
                if use_inpainting:
                    # Create mask for inpainting
                    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                    mask[y:y+height, x:x+width] = 255
                    frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
                else:
                    # Apply blur effect
                    roi = frame[y:y+height, x:x+width]
                    blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
                    frame[y:y+height, x:x+width] = blurred_roi

            # Write the processed frame
            out.write(frame)
            frame_count += 1

        # Release video objects
        if cap:
            cap.release()
        if out:
            out.release()

        # Add audio to processed video
        try:
            original_video = VideoFileClip(input_path)
            processed_video = VideoFileClip(temp_output)
            final_video = processed_video.set_audio(original_video.audio)

            output_filename = f"processed_{uuid.uuid4()}.mp4"
            output_path = os.path.join(PROCESSED_FOLDER, output_filename)
            
            final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)

            # Clean up
            if original_video:
                original_video.close()
            if processed_video:
                processed_video.close()
            if temp_output and os.path.exists(temp_output):
                os.unlink(temp_output)

            return True, output_path

        except Exception as e:
            # Fallback to video without audio if there's an error
            output_filename = f"processed_{uuid.uuid4()}.mp4"
            output_path = os.path.join(PROCESSED_FOLDER, output_filename)
            os.rename(temp_output, output_path)
            return True, output_path

    except Exception as e:
        error_msg = f"Error in process_video: {str(e)}"
        raise VideoProcessingError(error_msg) from e
    finally:
        # Ensure resources are released
        if cap:
            cap.release()
        if out:
            out.release()
        if original_video:
            original_video.close()
        if processed_video:
            processed_video.close()
        if temp_output and os.path.exists(temp_output):
            try:
                os.unlink(temp_output)
            except Exception:
                pass


@app.route("/process-video", methods=["POST"])
def process_video_route():
    """Handle video processing requests."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    temp_video = None
    
    try:
        # Clean up old processed videos before processing a new one
        cleanup_old_videos()

        video_file = request.files["video"]
        use_inpainting = request.form.get("use_inpainting", "false").lower() == "true"
        
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video_file.save(temp_video.name)

        # Process the video (watermark detection and removal)
        watermark_detected, processed_path = process_video(temp_video.name, use_inpainting)
        
        # If no watermark detected, return early with null processedVideoUri
        if not watermark_detected:
            return jsonify({
                "watermarkDetected": False,
                "processedVideoUri": None,
                "message": "No watermark detected in video"
            })
        
        # Only prepare download URL if watermark was detected and processed
        if os.path.exists(processed_path):
            os.chmod(processed_path, 0o644)
            host = request.host_url.rstrip("/")
            video_url = f"{host}/processed/{os.path.basename(processed_path)}"
            
            # Ensure we're not accumulating too many videos
            cleanup_old_videos()
            
            return jsonify({
                "watermarkDetected": True,
                "processedVideoUri": video_url,
            })
        else:
            error_msg = "Failed to process video - output not found"
            return jsonify({"error": error_msg}), 500

    except VideoProcessingError as e:
        error_msg = f"Video processing error: {str(e)}"
        return jsonify({"error": error_msg}), 500
    except VideoCleanupError as e:
        error_msg = f"Video cleanup error: {str(e)}"
        return jsonify({"error": error_msg}), 500
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        return jsonify({"error": error_msg}), 500
    finally:
        if temp_video and os.path.exists(temp_video.name):
            try:
                os.unlink(temp_video.name)
            except Exception:
                pass

@app.route("/processed/<filename>")
def serve_processed_video(filename):
    """Serve processed video files."""
    absolute_path = os.path.abspath(PROCESSED_FOLDER)
    response = send_from_directory(absolute_path, filename)
    response.headers["Content-Type"] = "video/mp4"
    return response

@app.route("/maintenance/cleanup", methods=["GET"])
def maintenance_cleanup():
    """Admin route to force cleanup of old processed videos"""
    try:
        # Force cleanup of all but the most recent video
        before_count = len(glob.glob(os.path.join(PROCESSED_FOLDER, "*.mp4")))
        cleanup_old_videos(force=True)
        after_count = len(glob.glob(os.path.join(PROCESSED_FOLDER, "*.mp4")))
        
        return jsonify({
            "success": True,
            "message": f"Cleaned up processed videos folder. Before: {before_count}, After: {after_count}",
            "deleted": before_count - after_count
        })
    except Exception as e:
        error_msg = f"Maintenance cleanup error: {str(e)}"
        return jsonify({
            "success": False,
            "error": error_msg
        }), 500

# Run cleanup on exit to ensure we don't leave orphaned files
atexit.register(lambda: cleanup_old_videos(force=True))

# Run an initial cleanup when the server starts
try:
    cleanup_old_videos(force=True)
except Exception:
    pass

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    app.run(host=host, port=port)
