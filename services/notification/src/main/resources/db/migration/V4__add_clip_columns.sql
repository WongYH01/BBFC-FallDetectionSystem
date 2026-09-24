-- All four nullable: storage and Telegram delivery are independent, so a clip that is
-- stored but not delivered is a valid recorded state, not a failure.
ALTER TABLE alerts
    ADD COLUMN clip_storage_key       VARCHAR(512),
    ADD COLUMN clip_stored_at         TIMESTAMPTZ,
    ADD COLUMN clip_telegram_file_id  VARCHAR(256),
    ADD COLUMN clip_message_id        BIGINT;
