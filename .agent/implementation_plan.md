# Implementation Plan: Async Report Generation

## Objective
Implement an asynchronous version to the report generation endpoint to prevent timeout issues during the compute-intensive analysis and translation phases. This will strictly follow the established `QA -> Payment -> Generate` workflow.

## Current Workflow & Issues
1.  **QA Scan**: Uploads image, checks disease. Stores initial record with `payment_status='PENDING'`. Returns `report_id`.
2.  **Payment**: User pays. Status updates to `SUCCESS`.
3.  **Generate (Current Sync)**: User calls `POST /report/generate`.
    *   **Issue**: This step performs S3 Download + Gemini Analysis (English) + BBox Generation + S3 Upload + DB Update + Optional Translation. This can take 30-60s, leading to client timeouts.

## Proposed Solution
Introduce `POST /report/generate_async` and refactor the logic to be shared with the synchronous endpoint.

### 1. Core Logic Extraction (`_core_process_generation`)
A shared internal function that:
*   Takes `report_id`, `language`, `db_session`.
*   Downloads image from S3.
*   Performs Gemini Analysis (English).
*   Updates `AnalysisReport` (or Fruit/Veg tables) with English data.
*   Sets `payment_status='SUCCESS'`.
*   **Translation**: If `language != 'en'`, triggers `TranslationService` to generate translated report.
*   **Output**: Returns the final JSON response (including `available_lang` metadata).
*   **Progress Tracking**: Updates Redis `task_manager` with steps (e.g., "Analyzing", "Translating") if a `task_id` is provided.

### 2. Endpoints
#### A. Synchronous (Existing, Refactored)
`POST /api/v1/report/generate`
*   **Behavior**: Calls `_core_process_generation` and waits for result.
*   **Use Case**: Quick server responses or clients that can wait.

#### B. Asynchronous (New)
`POST /api/v1/report/generate_async`
*   **Input**: Same as sync (`report_id`, `user_id`, `language`...).
*   **Behavior**: 
    1.  Verifies payment (Fail fast).
    2.  Generates `task_id`.
    3.  Queues background task.
    4.  Returns `{ "task_id": "...", "status": "queued" }`.
*   **Use Case**: Production clients to avoid timeouts.

#### C. Status Polling
`GET /api/v1/report/status/{task_id}`
*   Returns task status from Redis.
*   **Response Stages**:
    *   `queued`
    *   `processing`: Returns `{ "progress": 50, "stage": "Analysis Complete" }`
    *   `completed`: Returns `{ "result": <Final Report JSON> }`
    *   `failed`: Returns `{ "error": "..." }`

## Implementation Steps
1.  **Refactor `scan.py`**:
    *   Extract logic from `generate_full_report` into `_core_process_generation`.
    *   Implement `background_generation_worker` handler.
    *   Update `generate_full_report` to use the core function.
    *   Add `generate_report_async` endpoint.
    *   Add `get_generation_status` endpoint.
2.  **Database & Redis**:
    *   Ensure `task_manager` (Redis) is active and imported.
    *   Ensure `SessionLocal` is used for background tasks (to avoid detached session errors).
3.  **Validation**:
    *   Verify `available_lang` is correctly populated in both Sync and Async results.
    *   Verify `desired_language_output` is respected.

## Verification
*   User will call `POST /report/generate_async` with `kn`.
*   User will poll `/report/status/{task_id}`.
*   Expected result: Task completes, returns Kannada report JSON with `available_lang: {'en': True, 'kn': True}`.
