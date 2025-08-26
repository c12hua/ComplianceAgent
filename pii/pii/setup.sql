-- Create database and table for PII person_information
CREATE DATABASE IF NOT EXISTS ComplianceWizard CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
USE ComplianceWizard;

CREATE TABLE IF NOT EXISTS person_information (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  source_id VARCHAR(255) NULL,
  source_path VARCHAR(1024) NULL,
  text LONGTEXT NOT NULL,
  entities JSON NOT NULL,
  model VARCHAR(255) NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;
