# Ski Video Editor

An AI-assisted video processing application that automatically converts traditional landscape waterski videos into social media optimized vertical (9:16) videos by tracking the skier throughout the clip and dynamically reframing the camera view.

## Overview

Modern social media platforms prioritize vertical video formats, but most fact-paced sports footage is captured in traditional landscape orientation. This project solves that by using computer vision techniques to identify and track the athlete, keeping the athlete centered while generating a cropped vertical video suitable for platforms such as Instagram Reels, TikTok, and YouTube Shorts.

The application takes a standard widescreen video input and automatically:
- Detects the waterskier throughout the video
- Tracks skier movement frame-by-frame
- Calculates dynamic crop positioning
- Generates a smooth 9:16 vertical output
- Maintains focus on the athlete during high-speed motion

## Demo
https://ski-video-editor.vercel.app
### Original Landscape Videos

Traditional widescreen waterski footage:

![Original Landscape](images/original.png)
![Original Landscape](images/original2.png)

### Generated Social Media Format

Automatically reframed vertical video with skier tracking:

![Vertical Output](images/output.png)
![Vertical Output](images/output2.png)

## Features

### Automated Subject Tracking
Uses computer vision-based tracking to follow the skier as they move across the frame, reducing the need for manual editing.

### Dynamic Video Reframing
Instead of applying a static crop, the application adjusts the crop window throughout the video to keep the athlete visible.

### Social Media Optimization
Outputs videos in a 9:16 aspect ratio optimized for:
- Instagram Reels
- TikTok
- YouTube Shorts

### Web-Based Interface
Provides a simple interface for uploading videos and generating processed clips.

## Technology Stack

### Backend
- Python
- OpenCV
- Modal Serverless Compute
- GPU-accelerated video processing
- Video encoding and processing pipeline

### Frontend
- HTML
- CSS
- JavaScript

### Infrastructure
- Vercel Deployment (Frontend)
- Modal Cloud Functions (Backend Processing)

### Core Concepts
- Computer Vision
- Object Tracking
- Frame-by-frame Video Processing
- GPU Accelerated Workflows
- Serverless Architecture

## Project Architecture

The application uses a serverless video processing pipeline:

1. User uploads a landscape video through the web interface
2. Video is transferred to backend processing services
3. Modal GPU compute environment processes the video
4. Frames are extracted and analyzed
5. Athlete movement is tracked throughout the video
6. Dynamic crop positions are calculated
7. Frames are reconstructed into a 9:16 portrait video
8. Original audio is preserved and synchronized
9. Processed video is returned to the user

### Backend Functions

The backend is separated into dedicated processing functions:

- `upload`
  - Handles video upload workflow

- `process_video`
  - GPU accelerated video processing
  - Performs tracking, reframing, and rendering

- `status`
  - Tracks processing progress

- `result`
  - Handles completed video retrieval

## Future Improvements

Potential enhancements:
- Improved object detection using deep learning models
- Automated video type selection (Slalom or Jump Video)
- Multiple video submissions
- Motion prediction for smoother camera movement
- Cloud-based video processing pipeline

## Motivation

This project was created to solve a practical problem in waterski media production: converting landscape jump and slalom footage originally used for coaching and self-coaching into engaging short-form social-media-ready content without requiring manual video editing.

As an athlete and engineer, this project combines my interests in sports, software development, and applying engineering concepts to real-world problems.

## Author

Ben Leutz  
Aerospace Engineer | Controls & Simulation 
