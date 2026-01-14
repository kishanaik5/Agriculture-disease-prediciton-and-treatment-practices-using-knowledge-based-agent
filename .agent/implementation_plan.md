# Implementation Plan - Refine Analysis & Master Data

This conversation focused on refining the analysis report structure, translation logic, and updating the master icons database.

## User Review Required

> [!IMPORTANT]
> The `master_icons` table has been truncated and re-seeded with 341 records from the merged CSV. Please verify that the icons are appearing correctly in the frontend application.

- **Translation Response Structure:** The translation endpoint now returns a JSON structure identical to the English report (`ScanResponse` schema), with `report_uid` mapped to `id` and `item_name` to `user_input_crop`.
- **Category Filtering:** Knowledge base lookups (`get_treatment`) now strictly filter by `category` (crop/fruit/vegetable) to prevent ambiguity.

## Proposed Changes

### Backend Logic

#### `app/routers/v1/scan.py`
- [x] **Hardcoded Language:** Removed `language` form parameter from `/qa/scan` and hardcoded it to "en".
- [x] **Category Field:** Added `category` field (crop, fruit, vegetable) to the `ScanResponse` object and populated it in all analysis endpoints.

#### `app/routers/v1/translation.py`
- [x] **Standardized Response:**  Manually constructed the return dictionary in `translate_report` and `get_report` to match the exact JSON structure of the English report (removing internal DB fields like `payment_status`).
- [x] **Localized Treatment:** Implemented a call to `knowledge_service.get_treatment` using the *translated* crop and disease names to fetch localized treatment methods from the KB.

#### `app/services/knowledge.py`
- [x] **Enhanced Lookup:** Updated `get_treatment` to accept `category` and prioritize `scientific_name` for more accurate matches. Added case-insensitive matching for all fields.

### Database Updates

#### `master_icons`
- [x] **Data Merge:** Merged updated icon data (with Kannada/Hindi names) into `master_data.csv`, adding new items and resolving duplicate IDs.
- [x] **Re-seeding:** Created and ran scripts (`merge_csvs.py`, `fix_and_merge_csv.py`, `reset_icons_db.py`) to truncate the `master_icons` table and populate it with the cleaned dataset (341 records).

### Deployment
- [x] **Git Push:** Pushed all backend changes and migration scripts to `origin/dev`. Excluded large asset folders (`s3_file_upload`, `icons_folder`, `knowledge_based_folder`) from version control.

## Verification Plan

### Automated Tests
- [ ] **Translation Flow:** Run `verify_translation_flow.py` (if available or created) to ensure that translated reports contain the correct `category` and `kb_treatment`.
- [ ] **Icon Data:** Query the `master_icons` table to confirm unique `category_id`s and presence of localized names.

### Manual Verification
- [x] **QA Scan:** Verified `/qa/scan` accepts request without `language` optional param (default logic handled internally) and returns `category`.
- [ ] **Treatment Lookup:** Check a known disease (e.g., "Early Blight") in a translated report to see if the Kannada/Hindi treatment text is correctly appended.
