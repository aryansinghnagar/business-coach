# Engagement Detection Debugging Guide

## Issue: Engagement Bar Stuck at ~18/100

This guide helps diagnose and fix issues with engagement detection getting stuck at low scores.

## Debugging Steps

### 1. **Check Console Logs**

Look for these debug messages in the backend console:

```
Debug - Metrics: attention=X, eye_contact=Y, expressiveness=Z...
Debug - Features[0-2] (EAR): left_ear, right_ear, avg_ear
Debug - Raw score: X, Smoothed: Y, History size: Z
```

**What to look for:**
- **EAR values**: Should be 0.15-0.4 for normal eyes open
  - If < 0.1: Eyes might be closed or calculation error
  - If > 0.5: Calculation might be wrong
- **Metrics**: Should be 30-100 range
  - If all < 20: Feature extraction might be failing
- **Raw vs Smoothed**: Should be updating
  - If both stuck: Issue in scoring calculation
  - If raw updates but smoothed doesn't: Issue in smoothing

### 2. **Check Video Source**

Verify video source is providing frames:

```python
# In engagement_state_detector.py, check:
# - Video source initialized successfully
# - Frames are being read (ret=True)
# - Frame is not None
```

**Common issues:**
- Webcam not accessible
- File path incorrect
- Stream URL not working
- Video source returns empty frames

### 3. **Check Face Detection**

Verify faces are being detected:

```python
# Look for:
# - face_results list is not empty
# - landmarks array has correct shape (468 for MediaPipe, 27 for Azure)
# - Landmarks are not all zeros
```

**Common issues:**
- Face not in frame
- Face too small/distant
- Poor lighting
- Face detection confidence too low

### 4. **Check Feature Extraction**

Verify features are being extracted correctly:

```python
# Check:
# - blendshape_features array has 100 elements
# - Not all features are zero
# - EAR features (0-2) are in reasonable range (0.1-0.5)
# - Features are finite (not NaN/Inf)
```

**Common issues:**
- Landmark indices out of bounds
- EAR calculation returning 0
- All features defaulting to 0
- Feature extraction method mismatch

### 5. **Check Score Calculation**

Verify scores are being calculated:

```python
# Check:
# - All metrics are finite and in 0-100 range
# - Score calculation uses correct weights
# - No division by zero errors
# - Score is not always falling back to 50.0
```

## Common Issues and Fixes

### Issue 1: EAR Always Zero

**Symptoms:**
- Features[0-2] are all 0.0
- Attention score is very low (< 20)

**Causes:**
- Eye landmark indices incorrect
- EAR calculation using wrong points
- Landmarks not normalized correctly

**Fix Applied:**
- Improved EAR calculation to use bounding box method
- Added fallback for insufficient landmarks
- Better handling of edge cases

### Issue 2: All Metrics Low

**Symptoms:**
- All metrics < 30
- Score stuck at low value

**Causes:**
- Feature extraction returning zeros
- Scoring normalization too aggressive
- Minimum thresholds too high

**Fix Applied:**
- Added minimum scores for detected faces (20-40 range)
- Improved fallback calculations
- Better normalization ranges

### Issue 3: Smoothing Keeping Low Score

**Symptoms:**
- Raw score updates but smoothed doesn't
- Score history contains low initial values

**Causes:**
- Simple moving average keeps initial low scores
- No mechanism to reset on improvement

**Fix Applied:**
- Changed to exponential moving average (alpha=0.3)
- More weight to recent scores
- Faster response to changes

### Issue 4: Video Source Not Providing Frames

**Symptoms:**
- No debug messages
- Score stuck at last value
- "No frame available" in logs

**Causes:**
- Video source not initialized
- Source exhausted (file ended)
- Connection lost (stream)

**Fix Applied:**
- Added frame read validation on start
- Better error messages
- Automatic recovery for webcam

## Testing Checklist

- [ ] Video source provides frames (check FPS counter)
- [ ] Faces are detected (check face_detected flag)
- [ ] Features are extracted (check feature array not all zeros)
- [ ] EAR values are reasonable (0.1-0.5 range)
- [ ] Metrics are calculated (all > 0)
- [ ] Score updates over time
- [ ] Smoothed score responds to changes

## Expected Values

### Normal Operation:
- **EAR**: 0.2-0.35 (eyes open, engaged)
- **Attention**: 50-80 (moderate to high)
- **Eye Contact**: 60-90 (looking at camera)
- **Expressiveness**: 40-70 (moderate expression)
- **Head Movement**: 70-90 (stable head)
- **Symmetry**: 70-90 (symmetric face)
- **Mouth Activity**: 50-75 (moderate activity)
- **Overall Score**: 55-85 (engaged)

### Low Engagement:
- **EAR**: < 0.15 (eyes closed/drowsy)
- **Attention**: < 40
- **Eye Contact**: < 50
- **Overall Score**: < 45

## Manual Testing

### Test 1: Webcam with Face
1. Select "Webcam" as source
2. Look at camera
3. Check console for debug messages
4. Verify score updates (should be 50-80)

### Test 2: No Face
1. Cover camera or look away
2. Score should reset after ~1 second
3. Should show "No Face" or reset to 0

### Test 3: Different Expressions
1. Smile → Mouth activity should increase
2. Look away → Eye contact should decrease
3. Nod head → Head movement should change
4. Score should reflect changes

## Debugging Commands

### Check Current State
```python
# In Python console or add endpoint
state = engagement_detector.get_current_state()
print(f"Score: {state.score}, Face: {state.face_detected}")
print(f"Metrics: {state.metrics}")
```

### Check Feature Extraction
```python
# Add temporary logging
print(f"Features shape: {blendshape_features.shape}")
print(f"Features[0-5]: {blendshape_features[:5]}")
print(f"Non-zero features: {np.count_nonzero(blendshape_features)}")
```

### Check Video Source
```python
# Check if video source is working
ret, frame = video_handler.read_frame()
print(f"Frame read: {ret}, Frame shape: {frame.shape if frame is not None else None}")
```

## Files Modified for Debugging

1. `engagement_state_detector.py`
   - Added debug logging every ~1 second
   - Logs metrics, features, and scores
   - Better error messages

2. `utils/engagement_scorer.py`
   - Improved metric calculations with fallbacks
   - Better normalization ranges
   - Minimum scores for detected faces

3. `utils/business_meeting_feature_extractor.py`
   - Improved EAR calculation
   - Better handling of edge cases
   - Feature validation

## Next Steps

If issue persists after fixes:

1. **Enable verbose logging**: Set logging level to DEBUG
2. **Check specific metric**: Identify which metric is causing low score
3. **Test with known good video**: Use a test video with clear face
4. **Compare with baseline**: Test with simple scoring to verify detection works
5. **Profile performance**: Check if processing is too slow

## Contact

If issues persist, check:
- Backend console logs
- Browser console for frontend errors
- Network tab for API responses
- Video source accessibility
