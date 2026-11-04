package com.bbfc.notification.config;

import com.bbfc.notification.core.domain.DispatchPolicy;
import com.bbfc.notification.core.domain.EscalationPolicy;
import com.bbfc.notification.core.domain.FixedEscalationPolicy;
import com.bbfc.notification.core.domain.UploadLimit;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/** Hands the core its tunables as domain types, so core code never imports config. */
@Configuration
public class PolicyConfig {
    @Bean
    public EscalationPolicy escalationPolicy(EscalationConfig escalationConfig){
        return new FixedEscalationPolicy(escalationConfig.window(), escalationConfig.maxRepeats());
    }

    @Bean
    public DispatchPolicy dispatchPolicy(DispatchConfig dispatchConfig) {
        return new DispatchPolicy(dispatchConfig.clipWait());
    }

    @Bean
    public UploadLimit uploadLimit(ClipConfig clipConfig) {
        return new UploadLimit(clipConfig.maxBytes());
    }
}
