-- Kisaan CV Service Database Schema
-- Creates dedicated schema for service isolation

-- Create schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS kissan_cv;

-- Grant usage on schema
GRANT USAGE ON SCHEMA kissan_cv TO kissan_cv_dev;
GRANT CREATE ON SCHEMA kissan_cv TO kissan_cv_dev;

-- Set search path (optional, but recommended)
-- ALTER ROLE kissan_cv_dev SET search_path TO kissan_cv, public;

-- Table: kissan_cv.crop_analysis_report
-- Main table for storing crop disease analysis results
CREATE TABLE IF NOT EXISTS kissan_cv.crop_analysis_report (
    uid VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::VARCHAR,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE,
    deleted_at TIMESTAMP WITH TIME ZONE,
    
    -- User Information
    user_id VARCHAR(50) NOT NULL,
    
    -- Input Data
    user_input_crop VARCHAR(255),
    language VARCHAR(10) DEFAULT 'en',
    is_mixed_cropping BOOLEAN,
    acres_of_land VARCHAR(50),
    
    -- Detection Results
    detected_crop VARCHAR(255),
    detected_disease VARCHAR(255),
    pathogen_type VARCHAR(100),
    severity TEXT,
    treatment TEXT,
    
    -- Raw Analysis Data (JSON)
    analysis_raw JSONB,
    
    -- Image URLs
    original_image_url TEXT,
    bbox_image_url TEXT,
    report_url TEXT
);

-- Create indexes on user_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_crop_analysis_report_user_id ON kissan_cv.crop_analysis_report(user_id);
CREATE INDEX IF NOT EXISTS idx_crop_analysis_report_created_at ON kissan_cv.crop_analysis_report(created_at);

-- Grant privileges to kissan_cv_dev user
GRANT SELECT, INSERT, UPDATE, DELETE ON kissan_cv.crop_analysis_report TO kissan_cv_dev;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA kissan_cv TO kissan_cv_dev;

-- Optional: If SharedBackend tables are needed in kissan_cv schema, uncomment below:

-- Table: kissan_cv.entities (for managing upstream/downstream relationships)
-- CREATE TABLE IF NOT EXISTS kissan_cv.entities (
--     uid VARCHAR PRIMARY KEY DEFAULT uuid_generate_v4()::VARCHAR,
--     created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
--     updated_at TIMESTAMP WITH TIME ZONE,
--     deleted_at TIMESTAMP WITH TIME ZONE,
--     "upstreamId" VARCHAR NOT NULL,
--     tablename VARCHAR NOT NULL,
--     "downstreamId" VARCHAR NOT NULL,
--     UNIQUE(tablename, "downstreamId")
-- );
-- GRANT SELECT, INSERT, UPDATE, DELETE ON kissan_cv.entities TO kissan_cv_dev;

-- Verify tables created successfully
SELECT schemaname, tablename 
FROM pg_tables 
WHERE schemaname = 'kissan_cv' AND tablename IN ('crop_analysis_report');

