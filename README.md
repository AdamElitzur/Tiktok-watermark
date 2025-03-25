# TikTok Watermark Remover

A React Native application that automatically detects and removes TikTok watermarks from videos using computer vision techniques.

## Features

- Upload videos using Expo Image Picker
- Automatically detect TikTok watermarks using multiple detection methods:
  - Template matching with TikTok watermark template
  - Statistical analysis (variance, brightness, edge density)
- Remove watermarks using either blurring or inpainting techniques
- Visual feedback with loading spinner during processing
- Save processed videos to your device gallery
- Automatic cleanup of old processed videos
- Clean UI with toggle for selecting removal method

## Project Structure

```
.
├── frontend/           # React Native frontend with Expo
│   └── src/
│       └── app/        # Main application code
└── backend/           # Python Flask backend
    ├── app/
    │   └── main.py    # Main backend application
    ├── tests/         # Unit tests for the backend
    │   └── test_watermark_removal.py  # Tests for watermark detection and removal
    └── requirements.txt
```

## Prerequisites

- Node.js and npm
- Python 3.8+
- Expo CLI

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:

   ```bash
   cd backend
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the backend server:
   ```bash
   python app/main.py
   ```

The backend server will start at `http://localhost:5000`

### Frontend Setup

1. Navigate to the frontend directory:

   ```bash
   cd frontend
   ```

2. Install dependencies:

   ```bash
   npm install
   ```

3. Update the backend URL in `.env` file if needed:

   ```
   BACKEND_URL=http://your-local-ip:5000
   ```

4. Start the Expo development server:
   ```bash
   npx expo start
   ```

## Usage

1. Launch the app on your device/emulator
2. Toggle between blur (faster) and inpainting (higher quality) methods
3. Tap "Pick a Video" to select a video from your device
4. The app will:
   - Display a loading spinner while processing
   - Detect if the video contains a TikTok watermark
   - Process the video if watermarks are found
   - Display the processed video with watermarks removed
   - Allow you to save the processed video to your gallery

## Running Tests

The application includes unit tests for watermark detection and removal functionality:

```bash
cd backend
python -m unittest tests/test_watermark_removal.py
```

The tests cover the following scenarios:

- Watermark detection in videos with TikTok watermarks
- Correctly reporting absence of watermarks in clean videos
- Comparing blur vs. inpainting methods
- Testing mask generation for inpainting
- Testing the template matching capability
- Testing watermark region detection with various inputs

## Test Cases

When testing the application, verify these key functionalities:

- ✅ **Upload an MP4**  
  Can you select and upload a video from your phone?

- ✅ **Detect watermark**  
  If you upload a video with a TikTok watermark, is it detected?  
  If you upload a video without a watermark, does it correctly report no watermark?

- ✅ **Blur/inpaint**  
  Is the watermark visibly blurred or removed correctly?  
  Does the inpainting option produce a cleaner result than blurring?

- ✅ **Show detection result**  
  Does the frontend show "Watermark detected and removed successfully!" when detected?  
  Does it show "No watermark detected in this video" when none is found?

- ✅ **Play processed video**  
  Can the processed video be played in the app?  
  Do the play/pause controls work correctly?

- ✅ **Save to gallery**  
  Can you save the processed video to your device gallery?  
  Is there appropriate feedback after saving?

- ✅ **Video Cleanup**  
  Does the app properly clean up old processed videos?  
  Can you access the maintenance endpoint to force cleanup?

## Technical Details

### Frontend

- Built with React Native and Expo
- Uses expo-image-picker for video selection
- Expo Video for video playback
- Includes automatic local file cleanup to prevent storage issues
- ActivityIndicator for loading states

### Backend

- Flask server for handling video processing
- OpenCV for watermark detection and processing
- Multiple watermark detection methods for increased accuracy
- MoviePy for video processing
- Automatic cleanup of old processed videos
- Error handling with specific exception types

## Watermark Detection Algorithm

The application uses a hybrid approach to detect TikTok watermarks:

1. **Template Matching**: Uses OpenCV template matching to compare regions against a TikTok watermark template
2. **Statistical Analysis**:
   - Variance: Checks if the region has enough detail
   - Brightness: Ensures the brightness is within a typical watermark range
   - Edge density: Measures edges that are characteristic of text/graphics
3. **Position-based checks**: Checks regions where TikTok typically places watermarks:
   - Bottom left (0-3.47 seconds)
   - Bottom right (after 3.47 seconds)

By combining multiple detection methods, the app can accurately detect watermarks even if they vary slightly from the template.

## Processing Methods

1. **Blur**: Applies Gaussian blur to the watermark region
   - Faster processing
   - Less visible artifacts
2. **Inpainting**: Reconstructs the watermark region based on surrounding pixels
   - Better visual quality
   - May take slightly longer to process

## Libraries Used

### Frontend

- expo-image-picker
- expo-video
- expo-file-system
- expo-media-library

### Backend

- Flask
- OpenCV (opencv-python)
- NumPy
- MoviePy
- Flask-CORS

## Trade-offs and Assumptions

1. Watermark Detection:

   - Uses a hybrid detection approach (template matching + statistical analysis)
   - Assumes watermark switches location at 3.47 seconds
   - Could be further improved with ML-based detection

2. Processing Method:

   - Uses blurring or inpainting by generating its own mask
   - Assumes watermark size is static throughout the video

3. Performance:
   - Video processing happens on the backend to avoid mobile device limitations
   - Automatic cleanup reduces storage issues

## Future Improvements

- Machine learning-based watermark detection using a trained model
- Support for watermarks from other platforms (Instagram, YouTube, etc.)
- More sophisticated inpainting techniques
- Video compression options to reduce file size
