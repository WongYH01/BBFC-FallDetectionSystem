package com.bbfc.notification.config;

import org.springframework.boot.ApplicationRunner;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import com.bbfc.notification.in.telegram.UpdateSource;

/**
 * Defaults on: a missing property must not silently leave the service unable to receive
 * acknowledgements. Tests switch it off in application-test.yml.
 */
@Configuration
@ConditionalOnProperty(
        prefix = "telegram",
        name = "polling-enabled",
        havingValue = "true",
        matchIfMissing = true)
public class TelegramPollingConfig {

    // Anything thrown from ApplicationRunner.run fails startup, so this only kicks off the thread.
    @Bean
    public ApplicationRunner telegramPollingRunner(UpdateSource updateSource) {
        return args -> updateSource.start();
    }
}
