-- Migration: Drop report_url column from all analysis tables
-- Run this after deploying the updated code

-- Drop from crop_analysis_report
ALTER TABLE kissan_cv.crop_analysis_report DROP COLUMN IF EXISTS report_url;

-- Drop from fruit_analysis_report
ALTER TABLE kissan_cv.fruit_analysis_report DROP COLUMN IF EXISTS report_url;

-- Drop from vegetable_analysis_report
ALTER TABLE kissan_cv.vegetable_analysis_report DROP COLUMN IF EXISTS report_url;
