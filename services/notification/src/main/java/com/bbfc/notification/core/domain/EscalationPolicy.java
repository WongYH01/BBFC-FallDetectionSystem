package com.bbfc.notification.core.domain;

import java.time.Duration;

public interface EscalationPolicy {
    Duration window();
    int maxRepeats();
}
