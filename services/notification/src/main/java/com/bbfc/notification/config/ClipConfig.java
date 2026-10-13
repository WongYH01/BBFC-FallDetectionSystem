package com.bbfc.notification.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "clips")
public record ClipConfig(long maxBytes) {

    public ClipConfig {
        if (maxBytes <= 0) {
            maxBytes = 8_388_608L;
        }
    }
}
