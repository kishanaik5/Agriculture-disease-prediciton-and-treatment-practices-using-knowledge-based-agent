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

-- Table: kissan_cv.fruit_analysis_report
CREATE TABLE IF NOT EXISTS kissan_cv.fruit_analysis_report (
    uid VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::VARCHAR,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE,
    deleted_at TIMESTAMP WITH TIME ZONE,

    user_id VARCHAR(50) NOT NULL,
    fruit_name VARCHAR(255),
    
    -- Diagnosis
    disease_name VARCHAR(255),
    pathogen_name VARCHAR(255),
    condition_type VARCHAR(100),
    symptoms TEXT,
    severity_stage VARCHAR(100),
    affected_percentage VARCHAR(50),
    
    -- Market & Quality
    grade VARCHAR(100),
    suitability VARCHAR(100),
    reasoning TEXT,
    
    -- Shelf Life
    stability_score INTEGER,
    days_to_rot_room INTEGER,
    days_to_rot_cold INTEGER,
    
    -- Management
    organic_practices JSONB,
    chemical_practices JSONB,
    consumption_safety TEXT,
    
    -- Raw JSON
    analysis_json JSONB,
    
    -- Media
    original_image_url TEXT,
    bbox_image_url TEXT,
    report_url TEXT
);

-- Table: kissan_cv.vegetable_analysis_report
CREATE TABLE IF NOT EXISTS kissan_cv.vegetable_analysis_report (
    uid VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::VARCHAR,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE,
    deleted_at TIMESTAMP WITH TIME ZONE,

    user_id VARCHAR(50) NOT NULL,
    vegetable_name VARCHAR(255),
    
    -- Diagnosis
    disease_name VARCHAR(255),
    pathogen_type VARCHAR(100),
    cause VARCHAR(255),
    symptoms TEXT,
    severity_stage VARCHAR(100),
    affected_percentage VARCHAR(50),
    
    -- Market & Quality
    export_grade VARCHAR(100),
    market_suitability VARCHAR(100),
    
    -- Shelf Life
    stability_score INTEGER,
    days_to_rot_10c INTEGER,
    days_to_rot_25c INTEGER,
    
    -- Management
    organic_practices JSONB,
    chemical_practices JSONB,
    consumption_safety TEXT,
    
    -- Raw JSON
    analysis_json JSONB,
    
    -- Media
    original_image_url TEXT,
    bbox_image_url TEXT,
    report_url TEXT
);


-- Create indexes on user_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_crop_analysis_report_user_id ON kissan_cv.crop_analysis_report(user_id);
CREATE INDEX IF NOT EXISTS idx_crop_analysis_report_created_at ON kissan_cv.crop_analysis_report(created_at);

CREATE INDEX IF NOT EXISTS idx_fruit_analysis_report_user_id ON kissan_cv.fruit_analysis_report(user_id);
CREATE INDEX IF NOT EXISTS idx_fruit_analysis_report_created_at ON kissan_cv.fruit_analysis_report(created_at);

CREATE INDEX IF NOT EXISTS idx_vegetable_analysis_report_user_id ON kissan_cv.vegetable_analysis_report(user_id);
CREATE INDEX IF NOT EXISTS idx_vegetable_analysis_report_created_at ON kissan_cv.vegetable_analysis_report(created_at);

-- Grant privileges to kissan_cv_dev user
GRANT SELECT, INSERT, UPDATE, DELETE ON kissan_cv.crop_analysis_report TO kissan_cv_dev;
GRANT SELECT, INSERT, UPDATE, DELETE ON kissan_cv.fruit_analysis_report TO kissan_cv_dev;
GRANT SELECT, INSERT, UPDATE, DELETE ON kissan_cv.vegetable_analysis_report TO kissan_cv_dev;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA kissan_cv TO kissan_cv_dev;

-- Verify tables created successfully (Optional check)
SELECT schemaname, tablename 
FROM pg_tables 
WHERE schemaname = 'kissan_cv' AND tablename IN ('crop_analysis_report', 'fruit_analysis_report', 'vegetable_analysis_report');
