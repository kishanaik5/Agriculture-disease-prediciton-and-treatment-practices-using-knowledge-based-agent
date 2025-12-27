# Kisaan Sampurna CV Service 🌾🤖

A high-performance **Computer Vision & AI Service** for detecting crop diseases, providing treatment recommendations, and generating comprehensive health reports. Built with **FastAPI**, **Google Gemini 2.5**, and **AWS S3**.

## 🏗️ Architecture & Backend Flow

This service acts as the intelligent core of the Kisaan Sampurna ecosystem. Here is the step-by-step lifecycle of a single scan request:

### 1. Request Ingestion (`POST /api/v1/crop_scan`)
*   **Input**: User sends an image file, crop name (e.g., "Tomato"), and metadata (language, acres).
*   **Pre-Processing**: The image is validated and prepared for AI analysis.

### 2. Intelligent AI Analysis (Gemini 2.5)
The system performs a **Multi-Stage Analysis**:
1.  **Crop Validation**: Verifies if the image actually contains the claimed crop (e.g., checks if "Tomato" image is actually a tomato plant).
    *   *If Mismatch*: Returns immediate 400 Error (No cost, no storage).
2.  **Disease Detection**: Identifies pathogens, severity, and visual symptoms using `gemini-2.5-flash` (or `pro` fallback).
3.  **Bounding Box Generation**: Uses `gemini-3-pro-preview` to draw precise bounding boxes around diseased areas.

### 3. Knowledge Base Retrieval (Local/Hybrid)
*   **Source**: Local CSVs (`knowledge_based_folder/`) for 0-latency validation.
*   **Process**:
    *   Fuzzy matches the detected disease name against a curated agricultural database.
    *   Retrieves verified **Treatment Methods** (Chemical/Biological) in the requested language (English/Kannada).

### 4. Data Persistence & Storage
*   **AWS S3**:
    *   *Private Bucket*: Stores the raw, original user upload.
    *   *Public Bucket*: Stores the processed image with bounding boxes overlay.
*   **PostgreSQL**:
    *   Saves the full `AnalysisReport` with `JSONB` for the raw AI response.
    *   *Optimization*: Healthy scans are logged but not persisted deep in analytics if configured to skip.

---

## 🛠️ Tech Stack

*   **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Async Python)
*   **AI Models**:
    *   **Logic**: Google Gemini 2.5 Flash / Pro
    *   **Vision**: Google Gemini 3 Pro (Preview)
*   **Database**: PostgreSQL 15+ (with `asyncpg` & SQLAlchemy 2.0)
*   **Storage**: AWS S3 (Boto3)
*   **Containerization**: Docker & Docker Compose

---

## 📂 Project Structure

```
.
├── app
│   ├── api/v1/          # Endpoints (crop_scan, crops, health)
│   ├── core/            # Config & Database setup
│   ├── models/          # SQL Alchemy Database Models
│   ├── schemas/         # Pydantic Response/Request Schemas
│   ├── services/        # Business Logic Layers
│   │   ├── gemini.py    # AI Interaction
│   │   ├── image.py     # Image Processing (Pillow)
│   │   ├── knowledge.py # CSV Knowledge Base Logic
│   │   └── s3.py        # AWS S3 Uploads
│   └── main.py          # App Entrypoint
├── knowledge_based_folder/  # CSV Knowledge Bases (En/Kn)
├── static/              # Frontend Test Interface
├── Dockerfile           # Production Image Definition
└── requirements.txt     # Python Dependencies
```

---

## 🚀 Setup & Installation

### 1. Environment Variables
Create a `.env` file (copy from `.env.example`) and configure:

```ini
# Database
POSTGRES_SERVER=localhost
POSTGRES_USER=postgres
POSTGRES_DB=kisaan_cv_db
POSTGRES_PASSWORD=...

# AI Keys
GEMINI_API_KEY=AIzaSy...

# AWS Config
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
S3_BUCKET_PRIVATE=...
S3_BUCKET_PUBLIC=...
```

### 2. Run Locally (Docker)
The easiest way to run the service:

```bash
docker-compose up --build
```

### 3. Run Manually (Python)
```bash
# Install dependencies
pip install -r requirements.txt

# Start Server
uvicorn app.main:app --reload --port 8000
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check (`{"status": "healthy"}`) |
| `GET` | `/api/v1/crops` | List supported crops (`?language=en` or `kn`) |
| `POST` | `/api/v1/crop_scan` | **Main Analysis**: Upload image & metadata |

---

*Verified for Silo Fortune Deployment* 🚀
