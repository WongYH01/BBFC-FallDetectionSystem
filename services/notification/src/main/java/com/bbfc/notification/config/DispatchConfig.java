package com.bbfc.notification.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

import java.time.Duration;

@ConfigurationProperties(prefix = "alerting.dispatch")
public record DispatchConfig(Duration clipWait) {
    public DispatchConfig {
        if (clipWait == null) {
            clipWait = Duration.ofSeconds(20);
        }
    }
}
