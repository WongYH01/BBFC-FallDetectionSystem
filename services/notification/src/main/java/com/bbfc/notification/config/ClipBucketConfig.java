package com.bbfc.notification.config;

import com.bbfc.notification.core.port.ClipStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationRunner;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
@ConditionalOnProperty(
        prefix = "clips",
        name = "ensure-bucket",
        havingValue = "true",
        matchIfMissing = true)
public class ClipBucketConfig {

    private static final Logger log = LoggerFactory.getLogger(ClipBucketConfig.class);

    // Text alerts never touch Storage, so the service must still boot when Storage is down —
    // and anything thrown from ApplicationRunner.run would abort startup.
    @Bean
    public ApplicationRunner clipBucketRunner(ClipStore clipStore) {
        return args -> {
            try {
                clipStore.ensureBucket();
            } catch (RuntimeException e) {
                log.warn("Could not ensure the clip bucket exists; clip uploads will fail until it does", e);
            }
        };
    }
}
