-- Existing rows are all past Pending, so neither column needs a backfill.
ALTER TABLE alerts
    ADD COLUMN dispatch_deadline      TIMESTAMPTZ,
    ADD COLUMN dispatch_carries_clip  BOOLEAN NOT NULL DEFAULT FALSE;
