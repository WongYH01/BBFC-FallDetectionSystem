package com.bbfc.notification.config;

import jakarta.validation.constraints.NotBlank;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@Validated
@ConditionalOnProperty(prefix = "clips", name = "store", havingValue = "gcs")
@ConfigurationProperties(prefix = "clips.gcs")
public record GcsStorageConfig(@NotBlank String bucket) {
}
