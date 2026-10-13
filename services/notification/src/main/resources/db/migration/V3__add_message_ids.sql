ALTER TABLE alerts
    ADD COLUMN dispatch_message_id    BIGINT,
    ADD COLUMN escalation_message_ids BIGINT[]    NOT NULL DEFAULT '{}',
    ADD COLUMN follow_up_message_id   BIGINT,
    ADD COLUMN created_at             TIMESTAMPTZ NOT NULL DEFAULT now();
