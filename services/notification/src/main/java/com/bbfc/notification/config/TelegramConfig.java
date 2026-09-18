package com.bbfc.notification.config;

import jakarta.validation.constraints.NotBlank;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@Validated
@ConfigurationProperties(prefix = "telegram")
public record TelegramConfig(
        @NotBlank String botToken,
        @NotBlank String chatGroupId,
        String baseUrl,
        @NotBlank String messageTemplate,
        @NotBlank String triageQuestionTemplate
) {
    public TelegramConfig {
        if (baseUrl == null || baseUrl.isBlank()){
            baseUrl = "https://api.telegram.org";
        }
    }
}
