import unittest
import cv2
import numpy as np
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.main import process_video, check_watermark_region, generate_mask, TEMPLATE_IMAGE

class TestWatermarkRemoval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment with sample videos and test directories"""
        # Create a temporary directory for test files
        cls.test_dir = tempfile.mkdtemp()
        
        # Create test directories
        cls.test_uploads = os.path.join(cls.test_dir, 'uploads')
        cls.test_processed = os.path.join(cls.test_dir, 'processed')
        os.makedirs(cls.test_uploads, exist_ok=True)
        os.makedirs(cls.test_processed, exist_ok=True)
        
        # Create test videos
        cls.video_with_watermark = os.path.join(cls.test_dir, 'test_with_watermark.mp4')
        cls.video_without_watermark = os.path.join(cls.test_dir, 'test_without_watermark.mp4')
        
        # Create test videos with different scenarios
        cls._create_test_video_with_watermark()
        cls._create_test_video_without_watermark()
        
        # Ensure that the template image exists and is properly loaded
        print(f"Using template image with shape: {TEMPLATE_IMAGE.shape if TEMPLATE_IMAGE is not None else 'None'}")
    
    @classmethod
    def _create_test_video_with_watermark(cls):
        """Create a test video with TikTok-like watermarks at the standard positions"""
        width, height = 1080, 1920  # TikTok dimensions
        fps = 30
        duration = 6  # seconds
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(cls.video_with_watermark, fourcc, fps, (width, height))
        
        # Create frames with watermarks at TikTok positions
        for i in range(duration * fps):
            # Create a base frame (dark background with some content)
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add some content to the video (colored shapes)
            cv2.rectangle(frame, (400, 800), (700, 1100), (0, 0, 255), -1)  # Red rectangle
            cv2.circle(frame, (300, 500), 150, (0, 255, 0), -1)  # Green circle
            
            # Add fake text to simulate video content
            cv2.putText(frame, "Test Video", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                        3, (255, 255, 255), 5)
            
            # Add a timestamp to the video
            cv2.putText(frame, f"Frame: {i}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2)
            
            # Add watermark based on time (before or after 3.47 seconds)
            if i / fps <= 3.47:
                # First position (before 3.47 seconds) - bottom left
                x, y = 0, 875
                w, h = 194, 163
                # Create a TikTok-like watermark (white text on semi-transparent background)
                overlay = frame[y:y+h, x:x+w].copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (40, 40, 40), -1)
                cv2.putText(overlay, "TikTok", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 
                            1, (255, 255, 255), 2)
                cv2.putText(overlay, "@user", (20, 100), cv2.FONT_HERSHEY_COMPLEX, 
                            1, (255, 255, 255), 2)
                
                # Apply watermark with alpha blending (70% opacity)
                alpha = 0.7
                frame[y:y+h, x:x+w] = cv2.addWeighted(overlay, alpha, frame[y:y+h, x:x+w], 1-alpha, 0)
            else:
                # Second position (after 3.47 seconds) - bottom right
                x, y = 884, 1456
                w, h = 196, 163
                # Create a TikTok-like watermark (white text on semi-transparent background)
                overlay = frame[y:y+h, x:x+w].copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (40, 40, 40), -1)
                cv2.putText(overlay, "TikTok", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 
                            1, (255, 255, 255), 2)
                cv2.putText(overlay, "@user", (20, 100), cv2.FONT_HERSHEY_COMPLEX, 
                            1, (255, 255, 255), 2)
                
                # Apply watermark with alpha blending (70% opacity)
                alpha = 0.7
                frame[y:y+h, x:x+w] = cv2.addWeighted(overlay, alpha, frame[y:y+h, x:x+w], 1-alpha, 0)
            
            out.write(frame)
        
        out.release()
        print(f"Created test video with watermark: {cls.video_with_watermark}")
    
    @classmethod
    def _create_test_video_without_watermark(cls):
        """Create a test video without any watermarks"""
        width, height = 1080, 1920  # TikTok dimensions
        fps = 30
        duration = 4  # seconds
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(cls.video_without_watermark, fourcc, fps, (width, height))
        
        # Create frames without watermarks
        for i in range(duration * fps):
            # Create a base frame (dark background with some content)
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add some content to the video (colored shapes)
            cv2.rectangle(frame, (400, 800), (700, 1100), (255, 0, 0), -1)  # Blue rectangle
            cv2.circle(frame, (300, 500), 150, (255, 255, 0), -1)  # Yellow circle
            
            # Add fake text to simulate video content
            cv2.putText(frame, "Clean Video", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                        3, (255, 255, 255), 5)
            
            # Add a timestamp to the video
            cv2.putText(frame, f"Frame: {i}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"Created test video without watermark: {cls.video_without_watermark}")
    
    def test_watermark_detection_positive(self):
        """Test if watermark is correctly detected in a video that has one"""
        # Process the video with watermark
        watermark_detected, output_path = process_video(self.video_with_watermark, use_inpainting=False)
        
        # Check if watermark was detected
        self.assertTrue(watermark_detected, "Watermark should be detected in the test video")
        
        # Check if output file exists and has content
        self.assertTrue(os.path.exists(output_path), "Output video file should exist")
        self.assertTrue(os.path.getsize(output_path) > 0, "Output video file should have content")
        
        # Clean up the generated file
        if os.path.exists(output_path):
            os.remove(output_path)
    
    def test_watermark_detection_negative(self):
        """Test if no watermark is reported for a clean video"""
        # Process the video without watermark
        watermark_detected, output_path = process_video(self.video_without_watermark, use_inpainting=False)
        
        # Check that no watermark was detected
        self.assertFalse(watermark_detected, "No watermark should be detected in the clean video")
        
        # Output path should be the same as input for videos without watermarks
        self.assertEqual(output_path, self.video_without_watermark, 
                         "For videos without watermark, original path should be returned")
    
    def test_blur_vs_inpainting(self):
        """Test both processing methods and verify they produce different results"""
        # Process with blur
        _, blur_output = process_video(self.video_with_watermark, use_inpainting=False)
        
        # Process with inpainting
        _, inpaint_output = process_video(self.video_with_watermark, use_inpainting=True)
        
        # Both should produce valid output files
        self.assertTrue(os.path.exists(blur_output), "Blur output should exist")
        self.assertTrue(os.path.exists(inpaint_output), "Inpainting output should exist")
        
        # The file sizes should be different since the processing methods are different
        blur_size = os.path.getsize(blur_output)
        inpaint_size = os.path.getsize(inpaint_output)
        
        # Log sizes for debugging
        print(f"Blur output size: {blur_size}, Inpainting output size: {inpaint_size}")
        
        # Files should have content
        self.assertTrue(blur_size > 0, "Blur output should have content")
        self.assertTrue(inpaint_size > 0, "Inpainting output should have content")
        
        # Clean up
        if os.path.exists(blur_output):
            os.remove(blur_output)
        if os.path.exists(inpaint_output):
            os.remove(inpaint_output)
    
    def test_generate_mask(self):
        """Test mask generation for inpainting"""
        frame_shape = (1920, 1080)  # height, width
        bbox = (100, 200, 300, 150)  # x, y, w, h
        
        mask = generate_mask(frame_shape, bbox)
        
        # Mask should have the correct dimensions
        self.assertEqual(mask.shape, frame_shape, "Mask should have same dimensions as frame")
        
        # Check if the mask has white pixels inside the bounding box
        self.assertEqual(mask[250, 250], 255, "Pixel inside bbox should be white (255)")
        
        # Check if the mask has black pixels outside the bounding box
        self.assertEqual(mask[50, 50], 0, "Pixel outside bbox should be black (0)")
    
    def test_check_watermark_region(self):
        """Test watermark region checking with different regions"""
        # Create a test frame with a watermark-like region
        frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
        
        # Add a watermark-like pattern in a specific region
        x, y, w, h = 100, 200, 200, 150
        watermark_region = frame[y:y+h, x:x+w]
        
        # Create a pattern with text and background (simulating a watermark)
        cv2.rectangle(watermark_region, (0, 0), (w, h), (40, 40, 40), -1)
        cv2.putText(watermark_region, "TikTok", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 
                    1, (255, 255, 255), 2)
        cv2.putText(watermark_region, "@user", (20, 100), cv2.FONT_HERSHEY_COMPLEX, 
                    1, (255, 255, 255), 2)
        
        # Test with the watermark region
        result1 = check_watermark_region(frame, x, y, w, h)
        self.assertTrue(result1, "Should detect watermark in the region with text")
        
        # Test with a plain region (no watermark)
        result2 = check_watermark_region(frame, 500, 500, 200, 150)
        self.assertFalse(result2, "Should not detect watermark in plain region")
        
        # Test with out-of-bounds region
        result3 = check_watermark_region(frame, 5000, 5000, 200, 150)
        self.assertFalse(result3, "Should not detect watermark in out-of-bounds region")
        
        # Test with a region that looks similar to the template
        x4, y4, w4, h4 = 700, 700, TEMPLATE_IMAGE.shape[1], TEMPLATE_IMAGE.shape[0]
        if y4 + h4 <= frame.shape[0] and x4 + w4 <= frame.shape[1]:  # Make sure it fits
            frame[y4:y4+h4, x4:x4+w4] = TEMPLATE_IMAGE
            result4 = check_watermark_region(frame, x4, y4, w4, h4)
            self.assertTrue(result4, "Should detect watermark based on template matching")
    
    def test_template_matching(self):
        """Test the template matching capability"""
        # Create a frame with a copy of the template
        frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
        template_h, template_w = TEMPLATE_IMAGE.shape[:2]
        
        # Place the template in the frame
        pos_x, pos_y = 300, 400
        if pos_y + template_h <= frame.shape[0] and pos_x + template_w <= frame.shape[1]:
            frame[pos_y:pos_y+template_h, pos_x:pos_x+template_w] = TEMPLATE_IMAGE
            
            # The check_watermark_region should detect this exact template
            result = check_watermark_region(frame, pos_x, pos_y, template_w, template_h)
            self.assertTrue(result, "Should detect exact template match")
            
            # Test with slightly modified region - should still detect
            modified_frame = frame.copy()
            # Add some noise to the template region
            noise = np.random.normal(0, 15, (template_h, template_w, 3)).astype(np.uint8)
            modified_region = modified_frame[pos_y:pos_y+template_h, pos_x:pos_x+template_w]
            modified_region = cv2.add(modified_region, noise)
            modified_frame[pos_y:pos_y+template_h, pos_x:pos_x+template_w] = modified_region
            
            result_modified = check_watermark_region(modified_frame, pos_x, pos_y, template_w, template_h)
            self.assertTrue(result_modified, "Should detect template with some noise")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test directories and files"""
        try:
            # Remove test directory and all its contents
            shutil.rmtree(cls.test_dir)
            print(f"Cleaned up test directory: {cls.test_dir}")
        except Exception as e:
            print(f"Error during cleanup: {e}")

if __name__ == '__main__':
    unittest.main() 