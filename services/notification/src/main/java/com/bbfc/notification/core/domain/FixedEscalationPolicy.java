package com.bbfc.notification.core.domain;

import java.time.Duration;

public record FixedEscalationPolicy(Duration window, int maxRepeats) implements EscalationPolicy{
}
