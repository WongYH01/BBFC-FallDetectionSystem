ALTER TABLE alerts
    ADD COLUMN repeat_count INT NOT NULL DEFAULT 0,
    ADD COLUMN next_escalation_at TIMESTAMPTZ;

CREATE INDEX idx_alerts_due ON alerts (state, next_escalation_at)
    WHERE next_escalation_at IS NOT NULL;
