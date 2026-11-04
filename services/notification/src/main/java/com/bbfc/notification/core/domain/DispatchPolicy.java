package com.bbfc.notification.core.domain;

import java.time.Duration;

/** How long a received alert is held for its skeleton clip before it is dispatched as text. */
public record DispatchPolicy(Duration clipWait) {
}
