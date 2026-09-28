# 🚗 Vehicle Parking Slot Detection System

A computer vision-based intelligent parking management system that uses **YOLO-based object detection** to detect vehicles, identify parking slots, determine slot occupancy, count available parking spaces, and generate annotated parking-lot results.

The project provides both **manual parking-slot layout detection** for fixed cameras and an **optional custom-trained parking-slot detection model** for automated slot identification.

---

## 📖 Overview

Finding available parking spaces in large parking areas can be difficult when monitoring is performed manually. This project automates the parking-space analysis process using computer vision and deep learning.

The system accepts a parking-lot image through a **Flask web application**, detects vehicles using **YOLOv8**, identifies parking slots, analyzes vehicle-slot overlap, and determines which parking spaces are occupied or available.

The application also stores uploaded images, annotated results, and parking information locally using **SQLite**, allowing previously processed uploads to be viewed through the web interface.

---

## ❓ Problem Statement

Traditional parking management systems often depend on manual monitoring to determine whether parking spaces are available.

This approach can be inefficient for large parking areas and may result in inaccurate or delayed information.

The objective of this project is to develop an automated computer vision-based system that can:

- Detect vehicles in parking-lot images
- Count detected vehicles
- Identify individual parking slots
- Determine whether each slot is occupied or empty
- Display available parking slots
- Store processed images and detection results

---

## 🎯 Project Objectives

- Develop an automated parking-slot detection system
- Detect vehicles using YOLO-based object detection
- Count vehicles present in a parking area
- Identify occupied and vacant parking spaces
- Support manually defined parking layouts
- Support custom-trained parking-slot detection models
- Generate annotated detection results
- Store uploads and results for future reference
- Provide a simple browser-based interface
- Create a foundation for future real-time smart parking applications

---

## ✨ Key Features

- Vehicle detection using YOLOv8
- Automatic vehicle counting
- Parking-slot detection
- Empty/available slot identification
- Occupied slot identification
- Manual parking-slot layout creation
- Custom parking-slot model training
- Vehicle-slot overlap analysis
- High-accuracy tiled inference
- Annotated output generation
- Flask-based web interface
- SQLite database storage
- Saved uploads history
- Configurable detection threshold
- Support for fixed-camera parking environments

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| Flask | Web application framework |
| YOLOv8 | Object detection |
| Ultralytics | YOLO training and inference |
| OpenCV | Image processing |
| NumPy | Numerical computation |
| SQLite | Local database storage |
| HTML/CSS | Web interface |
| PowerShell | Environment setup on Windows |

---

## 📁 Repository Structure

```text
parking_app/
│
├── app.py
├── train_slots.py
├── requirements.txt
├── README.md
│
├── models/
│   └── slots.pt
│
├── uploads/
├── results/
├── parking.db
└── .gitignore
```

### File Description

| File / Directory | Description |
|------------------|-------------|
| `app.py` | Main Flask application, vehicle detection, parking analysis and storage |
| `train_slots.py` | Optional custom parking-slot model training script |
| `requirements.txt` | Python dependencies |
| `models/slots.pt` | Custom trained parking-slot detection model |
| `uploads/` | Stores uploaded parking images |
| `results/` | Stores annotated detection results |
| `parking.db` | SQLite database |
| `.gitignore` | Prevents unnecessary/generated files from being uploaded |

---

## 🔄 Project Workflow

The complete system follows this workflow:

```text
Upload Image
      ↓
Image Processing
      ↓
Vehicle Detection
      ↓
Vehicle Counting
      ↓
Parking Slot Detection
      ↓
Vehicle-Slot Overlap Analysis
      ↓
Occupied / Empty Classification
      ↓
Available Slot Identification
      ↓
Annotated Result Generation
      ↓
Database Storage
      ↓
Display Result
```

### 1. Image Upload

The user uploads a parking-lot image through the Flask web interface.

The uploaded image is stored locally in:

```text
uploads/
```

The image is then passed to the computer vision pipeline for processing.

### 2. Vehicle Detection

The system uses **YOLOv8** for detecting vehicles in the parking image.

The default vehicle detection model is:

```text
yolov8m.pt
```

The YOLO model generates bounding boxes around detected vehicles.

The detection process provides information such as:

- Vehicle location
- Bounding-box coordinates
- Object class
- Detection confidence

The detected vehicles are then used for vehicle counting and parking-slot occupancy analysis.

### 3. Vehicle Counting

After vehicle detection, the system counts the detected vehicles.

For example:

```text
Detected Vehicles: 18
```

This count provides an overview of the number of vehicles currently present in the parking area.

### 4. Parking Slot Detection

The system supports two approaches for detecting parking slots.

#### Method 1 — Drawn Parking Layout

The user can manually define parking spaces by clicking the four corners of each parking slot.

This method is particularly suitable for **fixed parking cameras**.

#### Method 2 — Trained Parking-Slot Model

A custom YOLO-based parking-slot detection model can be trained using `train_slots.py`.

The trained model is stored as:

```text
models/slots.pt
```

The model can then automatically detect parking spaces when a manually saved layout is not available.

### 5. Manual Parking Layout

For fixed-camera environments, the user can create a parking layout manually.

**Process**

1. Select a parking-lot image.
2. Enable **Draw Slots**.
3. Click the four corners of a parking slot.
4. Repeat for all parking slots.
5. Select **Save Layout**.

The parking layout is stored in the database and reused for subsequent images from the same camera view.

This approach provides reliable slot boundaries because the geometry of a fixed parking camera generally remains consistent.

### 6. Vehicle-Slot Overlap Analysis

After detecting vehicles and parking slots, the system determines whether each parking slot is occupied.

The application compares the detected vehicle bounding boxes with the parking-slot regions.

The default occupancy threshold is:

```python
OVERLAP = 0.30
```

A parking slot is considered **occupied** when the detected vehicle covers more than approximately **30%** of the slot area.

```text
Vehicle overlaps slot
          ↓
  Calculate overlap
          ↓
 Compare with threshold
          ↓
   ┌──────┴──────┐
   ↓             ↓
 > 30%          ≤ 30%
   ↓             ↓
Occupied        Empty
```

The `OVERLAP` value can be adjusted according to the camera angle, vehicle size, and parking layout.

A practical tuning range is approximately:

```text
0.20 – 0.50
```

### 7. Parking Slot Classification

Each detected parking space is classified as:

- **Occupied**
- **Available / Empty**

The final output provides the overall parking status.

Example:

```text
Total Parking Slots : 25
Occupied Slots      : 18
Available Slots     : 7
```

### 8. Custom Parking-Slot Model

The project includes an optional training pipeline:

```text
train_slots.py
```

This script can be used to train a custom parking-slot detection model.

The trained weights are saved as:

```text
models/slots.pt
```

If a manually drawn parking layout is not available, the application can use this model to automatically detect parking slots.

### 9. Parking-Slot Detection Priority

When both methods are available, the **manually drawn layout has priority**.

```text
Parking Image
      ↓
Saved Layout Available?
      │
   ┌──┴──┐
  YES    NO
   │      │
   ▼      ▼
Drawn   Slot Model
Layout  Available?
          │
       ┌──┴──┐
      YES    NO
       │      │
       ▼      ▼
    Custom   No automatic
     Model   slot detection
```

For a fixed camera, the drawn layout is generally the preferred approach.

> **Note:** To use the trained model instead, the previously saved layout must be cleared.

### 10. High Accuracy Tiled Detection

Large parking-lot images may contain vehicles that occupy a small portion of the overall image.

To improve detection of small vehicles, the application provides:

```text
High Accuracy (tiled)
```

Tiled inference divides the image into smaller sections and performs detection on individual regions.

```text
Large Parking Image
        ↓
 ┌─────────┬─────────┐
 │ Tile 1  │ Tile 2  │
 ├─────────┼─────────┤
 │ Tile 3  │ Tile 4  │
 └─────────┴─────────┘
        ↓
Individual Detection
        ↓
Combined Results
```

This approach can improve detection performance when vehicles appear small in large images.

### 11. YOLO Model Configuration

The default vehicle detection model is:

```text
yolov8m.pt
```

For systems with a capable GPU, a larger model such as:

```text
yolov8x.pt
```

can be used through the `VEHICLE_WEIGHTS` configuration.

A larger model may provide improved detection accuracy but requires more computational resources.

### 12. Accuracy Optimization

**Fixed Camera**

For fixed-camera parking environments, carefully drawing the parking layout is generally recommended.

**Large Parking Areas**

Enable **High Accuracy (tiled)** to improve detection of smaller vehicles.

**GPU Systems**

Use a larger YOLO model such as `yolov8x.pt` when sufficient GPU resources are available.

**Top-View Images**

For top-down or drone-based parking images, generic COCO-trained vehicle detection models may provide lower accuracy.

The vehicle detector can be fine-tuned using domain-specific datasets such as:

- VisDrone
- PKLot
- Other parking/vehicle datasets

---

## 📦 Dataset

The project supports training a custom parking-slot model using a parking dataset.

The dataset should be placed in a folder named:

```text
dataset/
```

next to `app.py`.

Example:

```text
parking_app/
│
├── app.py
├── train_slots.py
├── dataset/
│   └── parking dataset
```

The training script supports datasets containing:

- YOLO `.txt` annotations
- Pascal VOC `.xml` annotations

---

## 🧠 Custom Model Training

Activate the virtual environment and run:

```bash
python train_slots.py
```

The training pipeline performs the following operations:

```text
Parking Dataset
      ↓
Dataset Detection
      ↓
Annotation Processing
      ↓
YOLO Dataset Creation
      ↓
Model Training
      ↓
Custom Slot Model
      ↓
models/slots.pt
```

The generated model is saved to:

```text
models/slots.pt
```

After training, restart the application:

```bash
python app.py
```

### Training on CPU vs GPU

Training a custom YOLO model on a CPU can take several hours depending on the dataset and hardware.

For faster training, the same training process can be executed using:

- Google Colab GPU
- Kaggle GPU
- Local NVIDIA GPU

After training, download:

```text
models/slots.pt
```

and place it inside the project's `models/` directory.

---

## 💾 Data Storage

The system uses local storage for processed data.

| Data | Location |
|------|----------|
| Uploaded Images | `uploads/` |
| Annotated Results | `results/` |
| Database | `parking.db` |

The SQLite database is used to maintain application records and parking-related information.

### Saved Uploads

The web application provides a **Saved Uploads** section where processed images and corresponding results can be viewed.

The storage workflow is:

```text
Uploaded Image
      ↓
uploads/
      ↓
Vehicle + Parking Detection
      ↓
Annotated Image
      ↓
results/
      ↓
Database Record
      ↓
parking.db
```

This allows the system to maintain a history of processed parking images.

---

## 🌐 Web Application

The application is built using **Flask**.

Flask provides the interface between the user and the machine-learning backend.

The application workflow is:

```text
User
 ↓
Web Browser
 ↓
Flask Application
 ↓
Image Processing
 ↓
YOLO Detection
 ↓
Parking Analysis
 ↓
SQLite Storage
 ↓
Result Display
```

### Application Output

The system provides information such as:

- Total Vehicles
- Total Parking Slots
- Occupied Slots
- Available Slots

Example:

```text
Total Vehicles    : 18
Total Slots       : 25
Occupied Slots    : 18
Available Slots   : 7
```

The processed image is also annotated to visually represent the detection results.

---

## ⚙️ Installation

**Clone the repository:**

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

**Navigate to the project directory:**

```bash
cd YOUR_REPOSITORY
```

**Create a virtual environment:**

```bash
python -m venv venv
```

**Activate the environment on Windows PowerShell:**

```powershell
Set-ExecutionPolicy -Scope Process Bypass
venv\Scripts\Activate.ps1
```

**Install the required dependencies:**

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Flask application:

```bash
python app.py
```

Open the application in a browser:

```text
http://127.0.0.1:5000
```

> The `yolov8m.pt` model is downloaded automatically on the first run if it is not already available.

---

## 💻 System Requirements

### Software

- Python 3.x
- Windows / Linux / macOS
- pip
- Modern web browser

### Hardware

The application can operate on CPU hardware for development and testing.

A GPU is recommended for:

- Faster YOLO inference
- High-resolution images
- Tiled inference
- Custom model training
- Larger YOLO models

---

## ✅ Advantages

- Automated parking-space monitoring
- YOLO-based vehicle detection
- Automatic vehicle counting
- Individual parking-slot analysis
- Manual and automatic slot detection
- Suitable for fixed-camera environments
- Local image and database storage
- Browser-based interface
- Configurable occupancy threshold
- Custom model training support
- Expandable architecture

---

## ⚠️ Limitations

The current implementation has several practical limitations:

**Camera Position**
Manually defined parking layouts work best when the camera position remains fixed.

**Occlusion**
Vehicles can partially hide other vehicles or parking slots.

**Lighting Conditions**
Performance can be affected by:

- Low light
- Shadows
- Glare
- Weather conditions

**Top-View Images**
Generic vehicle detection models may perform poorly on extreme top-down images.

**Small Vehicles**
Very small vehicles may be difficult to detect without high-resolution or tiled inference.

**Computational Requirements**
Custom model training and high-accuracy inference may require significant computational resources.

---

## 🌍 Real-World Applications

The system can be adapted for:

- Shopping mall parking
- College and university parking
- Office parking
- Residential societies
- Hospitals
- Airports
- Railway stations
- Commercial parking facilities
- Smart-city parking
- Event parking areas

---

## 🚀 Future Enhancements

Future versions of the system can include:

- Real-time CCTV integration
- RTSP/IP camera support
- Multi-camera parking monitoring
- Real-time parking dashboard
- License plate recognition
- Vehicle entry and exit tracking
- Parking duration monitoring
- Parking reservation system
- Mobile application
- Cloud deployment
- Occupancy analytics
- Historical parking reports
- Real-time notifications
- Unauthorized parking detection
- Automatic camera calibration
- IoT-based smart parking integration

---

## 🏭 Production Deployment Considerations

For a production parking facility, the system can be extended from image-based processing to continuous camera monitoring.

A possible production architecture would be:

```text
CCTV / IP Camera
       ↓
Video Stream
       ↓
Frame Extraction
       ↓
Vehicle Detection
       ↓
Parking-Slot Analysis
       ↓
Occupancy Database
       ↓
Real-Time Dashboard
       ↓
Parking Availability
```

Additional production components could include:

- Authentication
- Role-based access control
- Secure image storage
- PostgreSQL/MySQL database
- Cloud infrastructure
- API integration
- Monitoring and logging
- Model version management
- Automated model retraining

---

## 🎓 Skills Demonstrated

- Computer Vision
- Deep Learning
- Object Detection
- YOLOv8
- Image Processing
- OpenCV
- Python Programming
- Flask Web Development
- Machine Learning Model Training
- Custom Dataset Processing
- SQLite Database Management
- Data Persistence
- Model Optimization
- Parking Analytics

---

## 🏗️ Project Architecture

The project integrates machine learning and software engineering components:

```text
                    Vehicle Parking System
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
    Web Interface      Computer Vision       Storage
       Flask             YOLO/OpenCV         SQLite
          │                  │                  │
          ▼                  ▼                  ▼
     Image Upload      Vehicle Detection   parking.db
                             │
                             ▼
                     Parking Slot Detection
                       /              \
                      /                \
                     ▼                  ▼
             Drawn Layout         Custom YOLO Model
                     \                /
                      \              /
                       ▼            ▼
                       Occupancy Analysis
                              │
                              ▼
                    Available Slot Detection
```

---

## 📌 Project Status

**Status:** Functional Computer Vision Prototype

### Implemented

- [x] Image upload
- [x] YOLO vehicle detection
- [x] Vehicle counting
- [x] Manual parking-slot layout
- [x] Automatic parking-slot model
- [x] Vehicle-slot overlap analysis
- [x] Occupied/empty classification
- [x] Annotated result generation
- [x] SQLite data storage
- [x] Saved uploads
- [x] Flask web interface
- [x] Custom parking-slot model training
- [x] Tiled/high-accuracy inference

### Planned

- [ ] Real-time CCTV support
- [ ] Multi-camera support
- [ ] Parking analytics dashboard
- [ ] License plate recognition
- [ ] Cloud deployment
- [ ] Mobile application
- [ ] Real-time notifications

---

## 📄 License

This project is developed for educational, research, and development purposes.

If external datasets or pretrained models are used, their respective licenses and usage conditions should be followed.

---

## 👨‍💻 Author

**Dipanshu Shende**

Artificial Intelligence & Machine Learning Student

GitHub: [https://github.com/dipanshushende](https://github.com/dipanshushende)

---

## 🙏 Acknowledgements

- Ultralytics YOLO
- OpenCV
- Flask
- NumPy
- SQLite
- Parking-slot dataset providers
- Open-source computer vision community

---

⭐ If you found this project useful, consider giving the repository a star!
