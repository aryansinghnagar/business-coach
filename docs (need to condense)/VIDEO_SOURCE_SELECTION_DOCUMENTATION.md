# Video Source Selection - Documentation

## Overview

The Business Meeting Copilot now includes a user-friendly modal interface for selecting the video source for engagement detection. Users can choose from three options: Webcam, Meeting Partner Video, or Local Video File.

## Features

### 1. **Modal Interface**
- Clean, modern modal design
- Three video source options with clear descriptions
- **Face Detection Method**: Choose **MediaPipe (Default)** or **Azure Face API** before starting. MediaPipe is default and recommended; Azure is shown only when configured.
- File upload support for local video files
- Keyboard navigation (Escape to close)
- Click outside to cancel

### 2. **Video Source Options**

#### Webcam
- Uses your default webcam (usually index 0)
- Best for live meetings where you want to analyze your own engagement
- No additional configuration needed

#### Meeting Partner Video
- Uses the meeting partner's video stream from the avatar session
- Best for analyzing the engagement of the person you're meeting with
- Automatically uses the WebRTC video stream

#### Local Video File
- Upload and analyze a recorded video file
- Supports common video formats: MP4, AVI, MOV, WebM
- File is uploaded to the server before analysis begins
- Useful for reviewing past meetings

## User Flow

### 1. **Automatic Prompt**
When the avatar session initializes and video becomes available, the video source selection modal automatically appears.

### 2. **Selection Process**
1. User sees the modal with three options
2. User selects their preferred source:
   - Click on an option card
   - Or use the radio buttons
3. If "Local Video File" is selected:
   - File input appears
   - User clicks "Choose Video File"
   - User selects a video file
   - File name is displayed
4. User clicks "Start Detection"

### 3. **File Upload** (if applicable)
- If a file is selected, it's uploaded to the server
- Upload progress is shown (button changes to "Uploading...")
- Server saves file to `uploads/` directory
- Server path is returned for engagement detection

### 4. **Engagement Detection Starts**
- Session manager starts with selected source
- Engagement detection begins immediately
- Modal closes automatically

## Implementation Details

### Frontend Components

#### `VideoSourceSelector` Class
Located in `static/js/video-source-selector.js`

**Key Methods:**
- `show()` - Display the modal
- `hide()` - Hide the modal
- `handleConfirm()` - Process user selection
- `createModal()` - Create modal DOM structure

**Usage:**
```javascript
const selector = new VideoSourceSelector({
    onSelect: (selection) => {
        // Handle selection
        const { sourceType, sourcePath, file, detectionMethod } = selection;
        // detectionMethod: 'mediapipe' (default) or 'azure_face_api'
    },
    onCancel: () => {
        // Handle cancellation
    }
});

selector.show();
```

### Backend Endpoints

#### `POST /engagement/upload-video`
Upload a video file for engagement detection.

**Request:**
- Content-Type: `multipart/form-data`
- Form field: `video` (file)

**Response:**
```json
{
    "success": true,
    "filePath": "/path/to/uploads/video.mp4",
    "message": "File uploaded successfully"
}
```

**Error Response:**
```json
{
    "error": "Error message",
    "details": "Detailed error information"
}
```

#### `POST /engagement/start`
Start engagement detection with selected source.

**Request:**
```json
{
    "sourceType": "webcam" | "file" | "stream",
    "sourcePath": "optional path for file/stream",
    "detectionMethod": "mediapipe" | "azure_face_api" (optional)
}
```

## Integration Points

### Session Initialization
The modal is triggered automatically when:
1. Avatar session initializes
2. Video track becomes available
3. `promptVideoSourceSelection()` is called

### Source Selection Handler
`handleVideoSourceSelected()` function:
1. Receives selection object
2. Uploads file if needed
3. Starts session with selected source
4. Handles errors gracefully

## Styling

### Modal Design
- Dark theme with glassmorphism effect
- Smooth animations and transitions
- Responsive layout
- Accessible keyboard navigation

### Option Cards
- Hover effects
- Selected state highlighting
- Clear icons and descriptions
- Radio button selection

### File Input
- Custom styled file picker
- File name display
- Format hints
- Upload progress indication

## Error Handling

### File Upload Errors
- Network errors are caught and displayed
- Invalid file types are rejected
- Upload failures show user-friendly messages

### Source Selection Errors
- Missing file selection shows alert
- Invalid source types are rejected
- Backend errors are displayed to user

## Best Practices

### 1. **User Experience**
- Modal appears automatically when needed
- Clear descriptions for each option
- Visual feedback during file upload
- Graceful error handling

### 2. **File Handling**
- Files are uploaded to dedicated `uploads/` directory
- Original filenames are preserved
- Server validates file types
- Clean error messages for failures

### 3. **Accessibility**
- Keyboard navigation (Escape to close)
- Clear labels and descriptions
- Visual feedback for interactions
- Screen reader friendly

## Troubleshooting

### Modal Not Appearing
1. Check browser console for JavaScript errors
2. Verify `videoSourceSelector` is initialized
3. Check that `promptVideoSourceSelection()` is called

### File Upload Failing
1. Check server logs for errors
2. Verify `uploads/` directory exists and is writable
3. Check file size limits
4. Verify file format is supported

### Source Selection Not Working
1. Check network tab for API calls
2. Verify backend endpoints are accessible
3. Check browser console for errors
4. Verify session manager is initialized

## Future Enhancements

Potential improvements:
1. **Video Preview**: Show preview of selected file
2. **Multiple Files**: Support batch analysis
3. **Stream URL Input**: Direct URL input for stream sources
4. **Source History**: Remember last selected source
5. **Advanced Options**: Frame rate, resolution settings

## Code Structure

```
business-meeting-copilot/
├── static/
│   └── js/
│       └── video-source-selector.js    # Modal component
├── index.html                          # Main HTML (includes modal styles)
├── routes.py                           # Backend API (upload endpoint)
└── uploads/                            # Uploaded video files (created automatically)
```

## License

See main project LICENSE file.

## Support

For issues, questions, or contributions, please refer to the main project repository.
