-- Nullable: most alerts never get a keypoint file, and the key moves once triage labels it.
ALTER TABLE alerts
    ADD COLUMN keypoints_storage_key VARCHAR(512),
    ADD COLUMN keypoints_stored_at   TIMESTAMPTZ;
