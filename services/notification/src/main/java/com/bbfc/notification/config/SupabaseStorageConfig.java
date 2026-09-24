package com.bbfc.notification.config;

import jakarta.validation.constraints.NotBlank;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@Validated
@ConfigurationProperties(prefix = "supabase.storage")
public record SupabaseStorageConfig(
        @NotBlank String url,
        @NotBlank String serviceKey,
        String bucket
) {
    public SupabaseStorageConfig {
        if (bucket == null || bucket.isBlank()) {
            bucket = "skeleton-clips";
        }
    }
}
