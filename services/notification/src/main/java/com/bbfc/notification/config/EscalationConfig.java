package com.bbfc.notification.config;

import jakarta.validation.constraints.Min;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

import java.time.Duration;

@Validated
@ConfigurationProperties(prefix = "alerting.escalation")
public record EscalationConfig(
        Duration window,
        @Min(1) int maxRepeats
) {
    public EscalationConfig{
        if (window == null){
            window = Duration.ofSeconds(60);
        }
        if (maxRepeats <= 0 ){
            maxRepeats = 3;
        }
    }
}
